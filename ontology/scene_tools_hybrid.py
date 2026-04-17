# Hybrid (KG + 3DSG) tool definitions.
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from ontology_server.scene_tools_common import _request


@tool
def locate_semantic(query: str, room_id: Optional[str] = None, top_k: int = 5) -> Any:
    """Semantic locate with optional room filter using hybrid backend."""
    payload: Dict[str, Any] = {"query": query, "top_k": top_k}
    if room_id:
        payload["room_id"] = room_id
    return _request("POST", "/hybrid/locate_semantic", json=payload)


@tool
def find_affordance_target(
    affordance: str,
    room_id: Optional[str] = None,
    top_k: int = 5,
) -> Any:
    """Find objects satisfying a specific affordance (hybrid)."""
    payload: Dict[str, Any] = {"affordance": affordance, "top_k": top_k}
    if room_id:
        payload["room_id"] = room_id
    return _request("POST", "/hybrid/find_affordance_target", json=payload)
