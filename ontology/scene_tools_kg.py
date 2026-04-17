# KG-only tool definitions (ontology/Neo4j side).
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from ontology_server.scene_tools_common import _request


@tool
def semantic_search(query: str, top_k: int = 5) -> Any:
    """Semantic search over the KG (vector index)."""
    params = {"query": query, "top_k": top_k}
    return _request("POST", "/semantic_search", params=params)


@tool
def search_by_name(name: str) -> Any:
    """Find a KG node by id/name."""
    params = {"name": name}
    return _request("GET", "/kg/search_by_name", params=params)


@tool
def list_in_category(category: str, room_id: Optional[str] = None) -> Any:
    """List KG nodes that have the given category (label)."""
    params: Dict[str, Any] = {"category": category}
    if room_id:
        params["room_id"] = room_id
    return _request("GET", "/kg/list_in_category", params=params)


@tool
def get_room(object_id: str) -> Any:
    """Get the room/space for a KG object."""
    params = {"object_id": object_id}
    return _request("GET", "/kg/get_room", params=params)


@tool
def get_node_info(node_id: str) -> Any:
    """Unified node info from KG (and DSG if available)."""
    params = {"node_id": node_id}
    return _request("GET", "/graph/get_node_info", params=params)


@tool
def shortest_path(
    start_id: str,
    goal_id: str,
    max_hops: int = 10,
    allow_closed: bool = True,
) -> Any:
    """Shortest path between rooms/spaces in KG; optional closed-door filtering."""
    params = {
        "start_id": start_id,
        "goal_id": goal_id,
        "max_hops": max_hops,
        "allow_closed": allow_closed,
    }
    return _request("GET", "/kg/shortest_path", params=params)
