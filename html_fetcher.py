"""
fetcher.py
----------
Fetches arXiv paper HTML from ar5iv.org with disk-based caching
and polite sleep times to respect robots.txt.

arXiv robots.txt allows crawling but requests politeness delays.
We use ar5iv.org (HTML version of arXiv) which is optimised for
programmatic access and has clean MathML/LaTeX equations.
"""

import os
import time
import logging
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ar5iv serves rendered HTML of arXiv papers
AR5IV_BASE = "https://ar5iv.org/abs/{arxiv_id}"

# Fallback: arxiv HTML endpoint (newer papers)
ARXIV_HTML_BASE = "https://arxiv.org/html/{arxiv_id}"

# Cache directory on disk so re-runs never re-download
CACHE_DIR = Path("cache/html")

# Polite crawl delay in seconds (ar5iv robots.txt: Crawl-delay not specified,
# but we use 3 s as a safe value to avoid hammering the server)
CRAWL_DELAY = 3.0

# HTTP request timeout
REQUEST_TIMEOUT = 30

# User-Agent identifying our academic project (good practice)
HEADERS = {
    "User-Agent": (
        "OTH-NLP-Project/1.0 (academic; OTH Amberg-Weiden; "
        "contact: student@oth-aw.de)"
    )
}

# Set up module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_paper_list(filepath: str) -> list[str]:
    """
    Read the paper list file and return a list of clean arXiv IDs.

    Parameters
    ----------
    filepath : str
        Path to the paper_list_<exam_id>.txt file.

    Returns
    -------
    list of str
        Ordered list of arXiv IDs, e.g. ['2506.23039', '2401.12877', ...].
    """
    ids = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format in file: "arXiv:2506.23039" or just the ID
            if line.lower().startswith("arxiv:"):
                arxiv_id = line.split(":", 1)[1].strip()
            else:
                arxiv_id = line
            ids.append(arxiv_id)
    logger.info("Loaded %d paper IDs from %s", len(ids), filepath)
    return ids


def fetch_paper_html(arxiv_id: str, use_cache: bool = True) -> str | None:
    """
    Fetch the HTML content of an arXiv paper, using local disk cache.

    Tries ar5iv.org first (better equation rendering), falls back to
    arxiv.org/html if ar5iv returns a 404 or error.

    Parameters
    ----------
    arxiv_id : str
        The arXiv paper ID, e.g. '2506.23039'.
    use_cache : bool, optional
        If True (default), serve from cache when available and save new
        fetches to cache. Set False to force a live download.

    Returns
    -------
    str or None
        Raw HTML string, or None if both sources failed.
    """
    cache_path = _cache_path(arxiv_id)

    # --- Cache hit ---
    if use_cache and cache_path.exists():
        logger.info("[CACHE HIT] %s", arxiv_id)
        return cache_path.read_text(encoding="utf-8")

    # --- Live fetch ---
    html = _fetch_from_ar5iv(arxiv_id)

    if html is None:
        logger.warning(
            "ar5iv failed for %s, trying arxiv.org/html", arxiv_id
        )
        html = _fetch_from_arxiv_html(arxiv_id)

    if html is None:
        logger.error("Both sources failed for %s", arxiv_id)
        return None

    # --- Save to cache ---
    if use_cache:
        _save_to_cache(cache_path, html)

    return html


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_from_ar5iv(arxiv_id: str) -> str | None:
    """
    Download HTML from ar5iv.org with polite delay.

    Parameters
    ----------
    arxiv_id : str
        The arXiv paper ID.

    Returns
    -------
    str or None
        HTML content or None on failure.
    """
    url = AR5IV_BASE.format(arxiv_id=arxiv_id)
    return _get_url(url, arxiv_id, source="ar5iv")


def _fetch_from_arxiv_html(arxiv_id: str) -> str | None:
    """
    Download HTML from arxiv.org/html (fallback).

    Parameters
    ----------
    arxiv_id : str
        The arXiv paper ID.

    Returns
    -------
    str or None
        HTML content or None on failure.
    """
    url = ARXIV_HTML_BASE.format(arxiv_id=arxiv_id)
    return _get_url(url, arxiv_id, source="arxiv-html")


def _get_url(url: str, arxiv_id: str, source: str) -> str | None:
    """
    Perform a GET request with polite delay and error handling.

    Parameters
    ----------
    url : str
        Full URL to fetch.
    arxiv_id : str
        Paper ID (used for logging).
    source : str
        Human-readable source name for logs.

    Returns
    -------
    str or None
        Response text or None on any failure.
    """
    logger.info("[FETCH] %s from %s -> %s", arxiv_id, source, url)

    # Polite delay before every live request
    time.sleep(CRAWL_DELAY)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )
    except requests.exceptions.RequestException as exc:
        logger.error("Network error fetching %s: %s", url, exc)
        return None

    if response.status_code == 200:
        logger.info(
            "[OK] %s from %s (%d bytes)",
            arxiv_id, source, len(response.text)
        )
        return response.text

    logger.warning(
        "[HTTP %d] %s from %s", response.status_code, arxiv_id, source
    )
    return None


def _cache_path(arxiv_id: str) -> Path:
    """
    Build the local cache file path for a given arXiv ID.

    Parameters
    ----------
    arxiv_id : str
        The arXiv paper ID.

    Returns
    -------
    Path
        Local path like cache/html/2506.23039.html
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Replace slashes in older IDs (e.g. hep-th/0001234) with underscores
    safe_id = arxiv_id.replace("/", "_")
    return CACHE_DIR / f"{safe_id}.html"


def _save_to_cache(cache_path: Path, html: str) -> None:
    """
    Write HTML content to the cache file.

    Parameters
    ----------
    cache_path : Path
        Destination file path.
    html : str
        HTML content to write.
    """
    cache_path.write_text(html, encoding="utf-8")
    logger.info("[CACHED] Saved to %s", cache_path)
