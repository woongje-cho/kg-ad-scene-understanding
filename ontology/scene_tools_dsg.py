from langchain_core.tools import tool

from ontology_server.scene_tools_common import _request


@tool
def get_position(object_id: str):
    """Return 3DSG position; accepts id ('29', 'chair_29') or class name."""
    params = {"object_id": object_id}
    return _request("GET", "/scene/get_position_fuzzy", params=params)


@tool
def get_room(object_id: str):
    """Return 3DSG room for an object id or id-with-class."""
    params = {"object_id": object_id}
    return _request("GET", "/scene/get_room", params=params)


@tool
def list_in_category(category: str, top_k: int = 20):
    """List 3DSG objects by class (category)."""
    params = {"category": category, "top_k": top_k}
    return _request("GET", "/scene/list_in_category", params=params)


@tool
def search_scene(query: str, top_k: int = 5):
    """Simple class-name search on 3DSG objects."""
    params = {"query": query, "top_k": top_k}
    return _request("GET", "/scene/search", params=params)


@tool
def shortest_path(start_room_id: str, goal_room_id: str):
    """Shortest path between two rooms using 3DSG connections."""
    params = {"start_room_id": start_room_id, "goal_room_id": goal_room_id}
    return _request("GET", "/scene/shortest_path", params=params)
