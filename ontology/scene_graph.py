# scene_graph.py
"""
Scene Manager Graph (LangGraph)
Mode-aware tool loading: KG / DSG(JSON) / Hybrid.
Set SCENE_MODE=kg|dsg|hybrid to choose the toolset.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Ensure package imports work whether run as module or script
ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# --- 1. Tool / LLM configuration ---
SCENE_MODE = os.getenv("SCENE_MODE", "kg").lower()
if SCENE_MODE not in {"kg", "dsg", "hybrid"}:
    SCENE_MODE = "kg"

if SCENE_MODE == "kg":
    from ontology_server.scene_tools_kg import (
        semantic_search,
        search_by_name,
        list_in_category,
        get_room,
        get_node_info,
        shortest_path,
    )

    tools = [
        semantic_search,
        search_by_name,
        list_in_category,
        get_room,
        get_node_info,
        shortest_path,
    ]
    MODE_PROMPT = (
        "KG mode: use semantic_search for descriptions/attributes, search_by_name for exact IDs, "
        "list_in_category for catalog queries, get_room/get_node_info for details/coordinates, "
        "shortest_path for room graph (respects door status: allow_closed=false by default)."
    )

elif SCENE_MODE == "dsg":
    from ontology_server.scene_tools_dsg import (
        get_position,
        get_room,
        list_in_category,
        search_scene,
        shortest_path,
    )

    tools = [
        get_position,
        get_room,
        list_in_category,
        search_scene,
        shortest_path,
    ]
    MODE_PROMPT = (
        "DSG(JSON) mode: use get_position (id or class) for coordinates, get_room for parent room, "
        "list_in_category/search_scene for class-based search, shortest_path for room graph (no door semantics)."
    )

else:
    from ontology_server.scene_tools_kg import (
        semantic_search,
        search_by_name,
        list_in_category as kg_list_in_category,
        get_node_info,
        shortest_path as kg_shortest_path,
    )
    from ontology_server.scene_tools_dsg import (
        get_position,
        get_room,
        list_in_category as dsg_list_in_category,
        search_scene,
    )

    tools = [
        # KG side
        semantic_search,
        search_by_name,
        kg_list_in_category,
        get_node_info,
        kg_shortest_path,
        # DSG side
        get_position,
        get_room,
        dsg_list_in_category,
        search_scene,
    ]
    MODE_PROMPT = (
        "Hybrid mode (no hybrid-only endpoints): combine KG + 3DSG tools. "
        "Decision: text/attributes/IDs -> KG (semantic_search/search_by_name/list_in_category/get_node_info); "
        "coordinates/room from 3D -> DSG (get_position/get_room/list_in_category/search_scene). "
        "Workflow: never finish after a single call—cross-validate. "
        "If you start from DSG, confirm the picked id with KG search_by_name/get_node_info. "
        "If you start from KG, ground coordinates with DSG get_position/get_room. "
        "Prefer KG shortest_path for room navigation (door-aware). "
        "If either side returns empty or mismatched rooms, back off, retry the other side, or ask for room_id/class."
    )

load_dotenv()
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
raw_key = os.getenv("OPENAI_API_KEY") or ""
clean_api_key = raw_key.strip()

if not clean_api_key:
    raise ValueError("OPENAI_API_KEY is missing in .env")

tool_node = ToolNode(tools)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.1, api_key=clean_api_key)
llm_with_tools = llm.bind_tools(tools)

# --- 2. Prompt ---
SYSTEM_PROMPT = f"""
You are the Scene Manager, a smart assistant for a mobile robot.
Your job is to understand a user's command and choose the best tool to get information.

Tool Selection Rules
- If the user gives descriptions, attributes, colors, owners, or phrases (not exact IDs): use semantic_search (KG). After a hit, call get_node_info for details/coordinates.
- Exact IDs only (e.g., "chair_10"): use search_by_name.
- Category lists (e.g., "all chairs"): use list_in_category (KG). If coordinates are needed, call get_node_info on the results.
- Room/space path (KG graph): use shortest_path (allow_closed=false by default; set allow_closed=true only if the user explicitly allows closed doors).
- DSG tools (raw 3D scene):
  - get_position: id or class name to fetch coordinates.
  - get_room: object_id to get its parent room_id.
  - list_in_category: class name (e.g., "chair") to list objects with coordinates.
  - search_scene: free-text class search (substring match on class_).
  - shortest_path: room-to-room path using 3DSG connections (no door-state semantics).
- Hybrid mode (KG+DSG only, no hybrid endpoints):
  - Do not use hybrid-only tools; rely on the KG and DSG tools above.
  - Never stop after a single call: if you start from DSG, validate the candidate with KG search_by_name/get_node_info; if you start from KG, ground with DSG get_position/get_room.
  - Prefer KG shortest_path for navigation. If one side returns empty or mismatched rooms, back off to the other side or ask for room_id/class.
- Room IDs: numeric room_id is preferred (e.g., Darden kitchen=20, living=22, bedroom=9); "room_20" is also accepted.
- When location is requested: follow up with get_node_info (KG) or get_position (DSG) on chosen IDs.
- If a tool returns empty: switch from name → semantic_search; from KG-only → DSG, or ask for a room_id to narrow down.
- If list_in_category returns 0 results, immediately back off to semantic_search with the user’s text.
- Final answers must explicitly mention the IDs of any referenced rooms/objects so the user can see which identifiers were used.
- For navigation, treat all doors as passable (ignore open/closed) unless the user explicitly forbids; when reporting a path, present only the sequence of room IDs (no connection types).
- Hygiene queries (soap/shampoo/handwash): if category lookup is empty, fall back to semantic_search using the user’s phrase; accept containers (e.g., Bottle) whose description mentions the target term and return their IDs/rooms.

- KG mode: if list_in_category returns 0 results, immediately run semantic_search with the original user text (or English keyword variant if needed), then call get_node_info on the hits; do not stop with "not found."
- DSG mode: final answers must include object IDs and parent_room (format id=<object_id>, room=room_<parent_room>); do not omit IDs.
- When reporting a room from DSG, include both the numeric form (room_<id>) and, if known, the semantic name (e.g., kitchen_20); if name is unknown, keep room_<id>.
Mode Emphasis
- KG mode: prioritize semantic_search/search_by_name/list_in_category/get_room/get_node_info/shortest_path. Great for rich text/attributes, room graphs, door-status filtering.
- DSG mode: prioritize get_position/get_room/list_in_category/search_scene/shortest_path (3DSG). Great for raw coordinates; no door semantics.
- Hybrid mode: blend KG + DSG (no hybrid-only tools). Do at least two hops: KG hit → DSG grounding (get_position/get_room), or DSG hit → KG validation (search_by_name/get_node_info). Use KG shortest_path for navigation; if mismatch/empty, pivot to the other side or ask the user.

Active mode: {SCENE_MODE.upper()}
{MODE_PROMPT}

Return your final answer in Korean, but keep IDs/technical fields in English.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
agent_chain = prompt | llm_with_tools

# --- 3. LangGraph state ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]


# --- 4. LangGraph nodes ---
def agent_node(state: AgentState):
    response = agent_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def should_continue(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "continue_to_tool"
    else:
        return "end"


# --- 5. Build graph ---
workflow = StateGraph(AgentState)
workflow.add_node("manager", agent_node)
workflow.add_node("tool", tool_node)
workflow.set_entry_point("manager")
workflow.add_conditional_edges(
    "manager",
    should_continue,
    {
        "continue_to_tool": "tool",
        "end": END,
    },
)
workflow.add_edge("tool", "manager")

graph = workflow.compile()
