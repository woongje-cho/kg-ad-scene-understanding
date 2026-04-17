# Shared helpers for scene tool modules (KG / DSG / Hybrid).
import logging
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("scene_tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Base URL for the FastAPI scene/ontology server
SCENE_SERVER_URL = os.getenv("SCENE_SERVER_URL", "http://localhost:8000")


def _request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: float = 6000.0,
) -> Any:
    """Lightweight HTTP helper used by tool modules."""
    url = f"{SCENE_SERVER_URL.rstrip('/')}{path}"
    logger.info("[Tool] %s %s params=%s json=%s", method, url, params, json)

    try:
        resp = requests.request(method, url, params=params, json=json, timeout=timeout)
    except Exception as exc:
        logger.exception("[Tool] request failed: %s", exc)
        raise

    if not resp.ok:
        logger.error(
            "[Tool] %s failed (status=%s, body=%s)", url, resp.status_code, resp.text
        )
        resp.raise_for_status()

    try:
        data = resp.json()
    except Exception:
        logger.error("[Tool] JSON parsing failed, raw body=%r", resp.text)
        raise

    logger.info("[Tool] response: %s", data)
    return data
