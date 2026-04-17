"""
Legacy aggregate exports. Use scene_tools_kg / scene_tools_dsg / scene_tools_hybrid for mode-specific imports.
"""
from ontology_server.scene_tools_common import SCENE_SERVER_URL
from ontology_server.scene_tools_kg import (
    semantic_search,
    search_by_name,
    list_in_category,
    get_room,
    get_node_info,
)
from ontology_server.scene_tools_dsg import get_position, get_node_info as dsg_get_node_info
from ontology_server.scene_tools_hybrid import locate_semantic, find_affordance_target

__all__ = [
    "SCENE_SERVER_URL",
    "semantic_search",
    "search_by_name",
    "list_in_category",
    "get_room",
    "get_node_info",
    "get_position",
    "dsg_get_node_info",
    "locate_semantic",
    "find_affordance_target",
]
