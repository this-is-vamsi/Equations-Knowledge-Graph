"""
pipeline.py
-----------
Main pipeline: fetch arXiv HTML → extract equations → write dataset JSON.

Follows the project spec exactly:
- Paper list processed in exact order
- Max 7 equations per paper
- Stop adding papers once dataset reaches 350 equations (always finish current paper)
- Every paper gets a key in the JSON (empty dict if no enumerated equations)
- Crash-safe save after every paper
- Auto-numbered output: dataset_001.json, dataset_002.json, ...
- HTML cached to disk so each paper is only downloaded once

Usage
-----
    python pipeline.py --paper-list paper_list_49.txt
    python pipeline.py --paper-list paper_list_49.txt --limit 5   # test mode
    python pipeline.py --paper-list paper_list_49.txt --resume    # after crash
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional

from html_fetcher import load_paper_list, fetch_paper_html
from equations_extractor import process_paper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_EQ_COUNT  = 350   # stop adding new papers once we have this many
MAX_EQ_PER_PAPER = 7     # enforced inside process_paper()
DEFAULT_OUT_BASE = "dataset"
DEFAULT_OUT_DIR  = "."
LOG_FILE         = "pipeline.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output file naming
# ---------------------------------------------------------------------------

def _resolve_output_path(
    out_base: str,
    out_dir: str,
    no_autonumber: bool,
    resume: bool,
) -> Path:
    """Return the output JSON path for this run.

    Auto-numbering: dataset_001.json, dataset_002.json, ...
    Resume: re-use the most recent existing numbered file.
    no_autonumber: use out_base as the exact filename.

    Parameters
    ----------
    out_base : str
        Base filename without extension, e.g. 'dataset'.
    out_dir : str
        Output directory.
    no_autonumber : bool
        If True, write to out_base directly.
    resume : bool
        If True and auto-numbering, return the most recent existing file.

    Returns
    -------
    Path
        Full path to the output file.
    """
    dir_path = Path(out_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    if no_autonumber:
        name = out_base if out_base.endswith(".json") else out_base + ".json"
        return dir_path / name

    base = re.sub(r"\.json$", "", out_base)
    highest = 0
    for p in dir_path.glob(f"{base}_*.json"):
        m = re.search(r"_(\d+)\.json$", p.name)
        if m:
            highest = max(highest, int(m.group(1)))

    if resume and highest > 0:
        path = dir_path / f"{base}_{highest:03d}.json"
        logger.info("Resume: loading %s", path)
        return path

    return dir_path / f"{base}_{highest + 1:03d}.json"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _count_equations(dataset: Dict) -> int:
    """Count total equations across all papers in the dataset.

    Parameters
    ----------
    dataset : dict
        The full dataset dict.

    Returns
    -------
    int
        Total number of equations.
    """
    return sum(len(eqs) for eqs in dataset.values())


def _load_existing(path: Path) -> Dict:
    """Load a JSON dataset file, or return empty dict if not found.

    Parameters
    ----------
    path : Path
        Path to a JSON file.

    Returns
    -------
    dict
        Loaded dataset or {}.
    """
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded existing dataset: %d papers, %d equations",
                    len(data), _count_equations(data))
        return data
    return {}


def _save(dataset: Dict, path: Path) -> None:
    """Write the dataset to JSON (pretty-printed, UTF-8).

    Parameters
    ----------
    dataset : dict
        The dataset to serialise.
    path : Path
        Destination file.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    paper_list_path: str,
    out_base: str,
    out_dir: str,
    no_autonumber: bool,
    resume: bool,
    limit: Optional[int],
) -> None:
    """Run the full extraction pipeline from paper list to dataset JSON.

    Processes papers in list order. Stops adding new papers once the dataset
    reaches TARGET_EQ_COUNT equations, but always completes the current paper.
    Saves the dataset after every paper so a crash loses at most one paper.

    Parameters
    ----------
    paper_list_path : str
        Path to paper_list_<exam_id>.txt.
    out_base : str
        Base name for output files, e.g. 'dataset'.
    out_dir : str
        Directory to write output files into.
    no_autonumber : bool
        Write to the exact out_base filename (no run-number suffix).
    resume : bool
        Load the most recent run file and skip already-processed papers.
    limit : int or None
        Process at most this many papers (test mode).
    """
    output_path = _resolve_output_path(out_base, out_dir, no_autonumber, resume)

    logger.info("=" * 60)
    logger.info("OUTPUT  : %s", output_path)
    logger.info("TARGET  : %d equations | max %d per paper",
                TARGET_EQ_COUNT, MAX_EQ_PER_PAPER)
    logger.info("=" * 60)

    # Load paper list in exact order (spec requirement)
    paper_ids = load_paper_list(paper_list_path)
    logger.info("Paper list: %d entries", len(paper_ids))

    if limit is not None:
        paper_ids = paper_ids[:limit]
        logger.info("TEST MODE: limiting to first %d papers", limit)

    # Load existing dataset if resuming
    dataset: Dict = _load_existing(output_path) if resume else {}
    already_done = set(dataset.keys()) if resume else set()
    if already_done:
        logger.info("Resume: %d papers already done", len(already_done))

    papers_processed = 0

    for idx, arxiv_id in enumerate(paper_ids):

        current_total = _count_equations(dataset)

        # Stopping condition: checked BEFORE starting a new paper
        # (spec: stop when we have 350, but always complete the last paper)
        if current_total >= TARGET_EQ_COUNT:
            logger.info(
                "Reached %d equations (≥ %d). Stopping after %d papers.",
                current_total, TARGET_EQ_COUNT, papers_processed,
            )
            break

        # Skip already-done papers in resume mode
        if arxiv_id in already_done:
            continue

        logger.info(
            "[%d/%d] %s  (total so far: %d equations)",
            idx + 1, len(paper_ids), arxiv_id, current_total,
        )

        # Step 1: fetch HTML (cache-first, 3-second polite delay)
        html = fetch_paper_html(arxiv_id, use_cache=True)
        if html is None:
            logger.warning("%s: fetch failed — recording empty entry.", arxiv_id)
            dataset[arxiv_id] = {}
            papers_processed += 1
            _save(dataset, output_path)
            continue

        # Step 2: extract equations
        try:
            eq_dict = process_paper(html, arxiv_id)
        except Exception as exc:
            logger.error("%s: extraction error — %s", arxiv_id, exc)
            dataset[arxiv_id] = {}
            papers_processed += 1
            _save(dataset, output_path)
            continue

        n_eq = len(eq_dict)
        logger.info("%s: extracted %d equation(s).", arxiv_id, n_eq)

        dataset[arxiv_id] = eq_dict
        papers_processed += 1

        new_total = _count_equations(dataset)
        logger.info("Running total: %d equations across %d papers.",
                    new_total, papers_processed)

        # Crash-safe save after every paper
        _save(dataset, output_path)

    # Final summary
    _print_summary(dataset, output_path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(dataset: Dict, output_path: Path) -> None:
    """Print a final run summary to the log.

    Parameters
    ----------
    dataset : dict
        Completed dataset.
    output_path : Path
        Where the dataset was written.
    """
    total_papers    = len(dataset)
    total_eqs       = _count_equations(dataset)
    papers_with_eqs = sum(1 for v in dataset.values() if v)

    total_syms   = sum(len(eq.get("symbols", {}))
                       for p in dataset.values() for eq in p.values())
    defined_syms = sum(1 for p in dataset.values() for eq in p.values()
                       for v in eq.get("symbols", {}).values() if v)
    with_meaning  = sum(1 for p in dataset.values() for eq in p.values()
                        if eq.get("meaning"))
    with_strong   = sum(1 for p in dataset.values() for eq in p.values()
                        if any(r.get("grade") == "strong"
                               for r in eq.get("relations", {}).values()))

    bar = "=" * 60
    logger.info(bar)
    logger.info("DONE")
    logger.info("Output file    : %s", output_path)
    logger.info("Papers total   : %d  (with equations: %d)",
                total_papers, papers_with_eqs)
    logger.info("Equations      : %d  (target: %d–%d)",
                total_eqs, TARGET_EQ_COUNT, TARGET_EQ_COUNT + MAX_EQ_PER_PAPER - 1)
    logger.info("Symbols def'd  : %d / %d  (%d%%)",
                defined_syms, max(total_syms, 1),
                100 * defined_syms // max(total_syms, 1))
    logger.info("Meanings found : %d / %d  (%d%%)",
                with_meaning, max(total_eqs, 1),
                100 * with_meaning // max(total_eqs, 1))
    logger.info("Eqs with strong relation: %d / %d",
                with_strong, max(total_eqs, 1))
    logger.info(bar)

    logger.info("--- Per-paper summary ---")
    for arxiv_id, eqs in dataset.items():
        n = len(eqs)
        logger.info("  %-24s : %s", arxiv_id,
                    f"{n} equation(s)" if n else "no enumerated equations")
    logger.info("--- End ---")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "NLP Project — Equations Knowledge Graph Pipeline.\n"
            "Produces dataset_001.json, dataset_002.json, ... per run.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py --paper-list paper_list_49.txt\n"
            "  python pipeline.py --paper-list paper_list_49.txt --limit 5\n"
            "  python pipeline.py --paper-list paper_list_49.txt --resume\n"
        ),
    )
    parser.add_argument(
        "--paper-list", required=True, metavar="FILE",
        help="Path to paper_list_<exam_id>.txt",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT_BASE, metavar="NAME",
        help=f"Output base name (default: '{DEFAULT_OUT_BASE}').",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUT_DIR, metavar="DIR",
        help=f"Directory for output files (default: '{DEFAULT_OUT_DIR}').",
    )
    parser.add_argument(
        "--no-autonumber", action="store_true",
        help="Write to --out directly (no run-number suffix).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N papers (test mode).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load the most-recent existing output and skip done papers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        paper_list_path=args.paper_list,
        out_base=args.out,
        out_dir=args.output_dir,
        no_autonumber=args.no_autonumber,
        resume=args.resume,
        limit=args.limit,
    )