#!/usr/bin/env python3
"""
KSAE 2026 Experiment v2: KG vs 3DSG for AD Scene Understanding
LLM-in-the-loop evaluation with GPT-4o-mini (answerer) and GPT-4o (judge)
30 queries x 2 modes x N_TRIALS repetitions
"""

import json
import os
import time
import math
import requests
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from openai import OpenAI
from scipy import stats
import numpy as np

# ============================================================
# Configuration
# ============================================================
GRAPHDB_URL = "http://localhost:7200/repositories/DrivingKG"
DRIVING_NS = "http://www.semanticweb.org/driving-ontology/2026/3#"
SPARQL_PREFIX = (
    f"PREFIX : <{DRIVING_NS}>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)

ANSWER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"
ANSWER_TEMP = 0.3
JUDGE_TEMP = 0.0
N_TRIALS = 5
MAX_CONTEXT_TOKENS = 1500  # Rough limit to keep context comparable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_GRAPH_PATH = os.path.join(SCRIPT_DIR, "scene_graph.json")
GROUND_TRUTH_PATH = os.path.join(SCRIPT_DIR, "ground_truth.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "experiment_results_v2.json")

client = OpenAI()  # Uses OPENAI_API_KEY env var


# ============================================================
# Data Loading
# ============================================================
with open(SCENE_GRAPH_PATH, "r", encoding="utf-8") as f:
    scene_graph = json.load(f)

with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

QUERIES = ground_truth_data["queries"]
SG_OBJECTS = {obj["id"]: obj for obj in scene_graph["objects"]}
SG_ZONES = {z["id"]: z for z in scene_graph["layers"]["zones"]}
SG_EDGES = scene_graph["edges"]


# ============================================================
# SPARQL Helper
# ============================================================
def sparql_query(query: str, infer: bool = True) -> list:
    """Execute SPARQL SELECT on GraphDB DrivingKG."""
    resp = requests.get(
        GRAPHDB_URL,
        params={"query": SPARQL_PREFIX + query, "infer": str(infer).lower()},
        headers={"Accept": "application/sparql-results+json"},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]["bindings"]
    parsed = []
    for b in results:
        row = {}
        for k, v in b.items():
            val = v["value"]
            if val.startswith(DRIVING_NS):
                val = val[len(DRIVING_NS):]
            row[k] = val
        parsed.append(row)
    return parsed


# ============================================================
# Query-to-Entity Mapping (new Q1-30 IDs)
# ============================================================
# Each query maps to a retrieval strategy: entities to fetch, zones to scan, etc.
KG_ENTITY_MAP = {
    1: {"entities": [], "zones": [], "nearby_ego": True},
    2: {"entities": [], "zones": [], "nearby_ego": True},
    3: {"entities": [], "zones": ["intersection_2"], "zone_objects": True, "zone_connections": True},
    4: {"entities": [], "all_vehicles": True},
    5: {"entities": ["traffic_light_1"]},
    6: {"entities": [], "all_pedestrians": True},
    7: {"entities": ["traffic_light_1", "sedan_3"]},
    8: {"entities": ["truck_4", "no_parking_sign_6"]},
    9: {"entities": [], "zones": ["school_zone_1"], "zone_objects": True},
    10: {"entities": ["sedan_3", "ego_car"]},
    11: {"entities": [], "zones": ["intersection_3", "side_road_1"], "zone_connections": True, "extra_entities": ["stop_sign_4"]},
    12: {"entities": ["traffic_light_6", "construction_cone_1", "construction_cone_2", "construction_sign_9"], "zones": ["construction_zone_1"]},
    13: {"entities": ["traffic_light_7", "sedan_15", "yield_sign_5", "sedan_10", "sedan_11"], "zones": ["merge_zone_1"]},
    14: {"entities": [], "all_traffic_control": True},
    15: {"entities": ["crosswalk_1", "crosswalk_2"], "hierarchy_for": ["Crosswalk"]},
    16: {"entities": [], "zone_types": True},
    17: {"entities": ["traffic_light_1", "sedan_3", "emergency_vehicle_12", "ego_car"]},
    18: {"entities": ["bus_6", "pedestrian_1", "motorcycle_8", "crosswalk_1", "traffic_light_2", "bus_stop_1"]},
    19: {"entities": [], "zones": ["school_zone_1"], "zone_objects": True},
    20: {"entities": ["debris_3", "pedestrian_4"], "zones": ["wet_road_section_1"]},
    21: {"entities": ["sedan_2", "sedan_1", "ego_car"]},
    22: {"entities": ["elderly_pedestrian_3", "stop_sign_4", "traffic_light_3"], "zones": ["intersection_3"]},
    23: {"entities": [], "scene_overview": True},
    24: {"entities": ["traffic_light_1", "sedan_3", "emergency_vehicle_12", "ego_car"]},
    25: {"entities": ["speed_limit_sign_3", "sedan_10", "sedan_11", "truck_5"]},
    26: {"entities": ["truck_4", "no_parking_sign_6"]},
    27: {"entities": ["bus_6", "pedestrian_1", "motorcycle_8", "crosswalk_1", "traffic_light_2", "bus_stop_1"]},
    28: {"entities": ["sedan_15", "yield_sign_5", "traffic_light_7", "sedan_10", "sedan_11", "guardrail_2"], "zones": ["merge_zone_1"]},
    29: {"entities": ["emergency_vehicle_12", "traffic_light_1", "ego_car", "sedan_3"]},
    30: {"entities": ["bus_7", "child_pedestrian_2", "child_pedestrian_5", "bicycle_9", "crosswalk_2", "speed_limit_sign_2"]},
}


# ============================================================
# Context Retrieval: KG Mode (SPARQL) — entity-driven
# ============================================================
def _kg_fetch_entity(entity_id: str) -> str:
    """Fetch a single entity's full KG context (most specific type only)."""
    data = sparql_query(
        f"""SELECT ?zone ?x ?y ?speed ?state ?obstacle ?comment ?type WHERE {{
            :{entity_id} :inZone ?zone ; :x ?x ; :y ?y ; rdf:type ?type .
            OPTIONAL {{ :{entity_id} :speed ?speed }}
            OPTIONAL {{ :{entity_id} :state ?state }}
            OPTIONAL {{ :{entity_id} :isObstacle ?obstacle }}
            OPTIONAL {{ :{entity_id} rdfs:comment ?comment }}
            FILTER(?type != owl:NamedIndividual)
            FILTER NOT EXISTS {{ :{entity_id} rdf:type ?sub . ?sub rdfs:subClassOf ?type . FILTER(?sub != ?type) }}
        }} LIMIT 1"""
    )
    if not data:
        return ""
    d = data[0]
    line = f"[{entity_id}] type={d['type']}, zone={d['zone']}, pos=({d['x']},{d['y']})"
    if d.get("speed"):
        line += f", speed={d['speed']}km/h"
    if d.get("state"):
        line += f", state={d['state']}"
    if d.get("obstacle"):
        line += f", obstacle={d['obstacle']}"
    if d.get("comment"):
        line += f"\n  description: {d['comment']}"
    return line


def _kg_fetch_zone_objects(zone_id: str) -> str:
    """Fetch all objects in a zone with descriptions."""
    zone_info = sparql_query(f"SELECT ?limit WHERE {{ :{zone_id} :speedLimit ?limit }}")
    objs = sparql_query(
        f"""SELECT ?obj ?type ?speed ?state ?comment WHERE {{
            ?obj :inZone :{zone_id} ; rdf:type ?type .
            OPTIONAL {{ ?obj :speed ?speed }}
            OPTIONAL {{ ?obj :state ?state }}
            OPTIONAL {{ ?obj rdfs:comment ?comment }}
            FILTER(?type != owl:NamedIndividual)
            FILTER NOT EXISTS {{ ?obj rdf:type ?sub . ?sub rdfs:subClassOf ?type . FILTER(?sub != ?type) }}
        }}"""
    )
    lines = []
    if zone_info:
        lines.append(f"[{zone_id}] speedLimit={zone_info[0]['limit']}km/h")
    lines.append(f"[Objects in {zone_id}: {len(objs)}]")
    for o in objs:
        line = f"  {o['obj']} type={o['type']}"
        if o.get("speed"):
            line += f", speed={o['speed']}km/h"
        if o.get("state"):
            line += f", state={o['state']}"
        if o.get("comment"):
            line += f"\n    desc: {o['comment'][:250]}"
        lines.append(line)
    return "\n".join(lines)


def _kg_fetch_zone_connections(zone_id: str) -> str:
    """Fetch connections for a zone."""
    conns = sparql_query(
        f"""SELECT ?conn ?zone2 ?open WHERE {{
            ?conn :isConnectionOf :{zone_id} ;
                  :isConnectionOf ?zone2 ;
                  :isOpen ?open .
            FILTER(?zone2 != :{zone_id})
        }}"""
    )
    lines = [f"[Connections of {zone_id}]"]
    for c in conns:
        lines.append(f"  {c['conn']} → {c['zone2']} (open={c['open']})")
    return "\n".join(lines)


def retrieve_kg_context(query_id: int, question: str) -> str:
    """Retrieve structured KG context via SPARQL for a given query."""
    parts = []
    config = KG_ENTITY_MAP.get(query_id, {})

    # Always include ego context
    ego = sparql_query(
        """SELECT ?zone ?x ?y ?speed ?state ?comment WHERE {
            :ego_car :inZone ?zone ; :x ?x ; :y ?y ; :speed ?speed ; :state ?state ;
            rdfs:comment ?comment . }"""
    )
    if ego:
        e = ego[0]
        parts.append(
            f"[Ego Vehicle] zone={e['zone']}, pos=({e['x']},{e['y']}), "
            f"speed={e['speed']}km/h, state={e['state']}\n  description: {e['comment']}"
        )

    # Fetch specific entities
    for eid in config.get("entities", []):
        if eid == "ego_car":
            continue  # Already fetched
        ctx = _kg_fetch_entity(eid)
        if ctx:
            parts.append(ctx)

    for eid in config.get("extra_entities", []):
        ctx = _kg_fetch_entity(eid)
        if ctx:
            parts.append(ctx)

    # Nearby objects for ego
    if config.get("nearby_ego"):
        nearby = sparql_query(
            """SELECT ?obj ?type ?zone ?x ?y ?speed ?state ?comment WHERE {
                ?obj :inZone ?zone ; :x ?x ; :y ?y ; rdf:type ?type .
                OPTIONAL { ?obj :speed ?speed }
                OPTIONAL { ?obj :state ?state }
                OPTIONAL { ?obj rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual)
                FILTER NOT EXISTS { ?obj rdf:type ?sub . ?sub rdfs:subClassOf ?type . FILTER(?sub != ?type) }
            } LIMIT 25"""
        )
        lines = [f"[Nearby objects: {len(nearby)}]"]
        for o in nearby:
            line = f"  {o['obj']} type={o['type']}, zone={o['zone']}, pos=({o['x']},{o['y']})"
            if o.get("speed"):
                line += f", speed={o['speed']}km/h"
            if o.get("state"):
                line += f", state={o['state']}"
            if o.get("comment"):
                line += f"\n    desc: {o['comment'][:200]}"
            lines.append(line)
        parts.append("\n".join(lines))

    # Zone objects
    for zone_id in config.get("zones", []):
        if config.get("zone_objects"):
            parts.append(_kg_fetch_zone_objects(zone_id))
        if config.get("zone_connections"):
            parts.append(_kg_fetch_zone_connections(zone_id))
        if not config.get("zone_objects") and not config.get("zone_connections"):
            zone_info = sparql_query(f"SELECT ?limit WHERE {{ :{zone_id} :speedLimit ?limit }}")
            if zone_info:
                parts.append(f"[{zone_id}] speedLimit={zone_info[0]['limit']}km/h")

    # All vehicles
    if config.get("all_vehicles"):
        vehicles = sparql_query(
            """SELECT DISTINCT ?v ?type ?speed ?state ?zone ?comment WHERE {
                ?v rdf:type/rdfs:subClassOf* :Vehicle ; rdf:type ?type ;
                   :speed ?speed ; :state ?state ; :inZone ?zone .
                OPTIONAL { ?v rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual)
                FILTER NOT EXISTS { ?v rdf:type ?sub . ?sub rdfs:subClassOf ?type . FILTER(?sub != ?type) }
            }"""
        )
        lines = [f"[All vehicles: {len(vehicles)}]"]
        for v in vehicles:
            line = f"  {v['v']} type={v['type']}, zone={v['zone']}, speed={v['speed']}km/h, state={v['state']}"
            if v.get("comment"):
                line += f"\n    desc: {v['comment'][:150]}"
            lines.append(line)
        parts.append("\n".join(lines))

    # All pedestrians
    if config.get("all_pedestrians"):
        peds = sparql_query(
            """SELECT DISTINCT ?p ?type ?zone ?speed ?state ?comment WHERE {
                ?p rdf:type/rdfs:subClassOf* :Pedestrian ; rdf:type ?type ;
                   :inZone ?zone ; :speed ?speed ; :state ?state .
                OPTIONAL { ?p rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual)
                FILTER NOT EXISTS { ?p rdf:type ?sub . ?sub rdfs:subClassOf ?type . FILTER(?sub != ?type) }
            }"""
        )
        lines = [f"[All pedestrians: {len(peds)}]"]
        for p in peds:
            line = f"  {p['p']} type={p['type']}, zone={p['zone']}, speed={p['speed']}km/h, state={p['state']}"
            if p.get("comment"):
                line += f"\n    desc: {p['comment'][:200]}"
            lines.append(line)
        parts.append("\n".join(lines))

    # All traffic control
    if config.get("all_traffic_control"):
        tc = sparql_query(
            """SELECT ?obj ?type WHERE {
                ?obj rdf:type/rdfs:subClassOf* :TrafficControl ; rdf:type ?type .
                FILTER(?type != owl:NamedIndividual)
                FILTER NOT EXISTS { ?obj rdf:type ?sub . ?sub rdfs:subClassOf ?type . FILTER(?sub != ?type) }
            }"""
        )
        hierarchy = sparql_query(
            """SELECT ?sub ?super WHERE {
                ?sub rdfs:subClassOf ?super .
                FILTER(?super IN (:TrafficControl, :TrafficLight, :TrafficSign))
            }"""
        )
        lines = [f"[TrafficControl objects: {len(tc)}]"]
        for t in tc:
            lines.append(f"  {t['obj']} type={t['type']}")
        lines.append("[Class hierarchy]")
        for h in hierarchy:
            lines.append(f"  {h['sub']} → subClassOf → {h['super']}")
        parts.append("\n".join(lines))

    # Hierarchy for specific class
    for cls in config.get("hierarchy_for", []):
        hierarchy = sparql_query(f"SELECT ?super WHERE {{ :{cls} rdfs:subClassOf+ ?super }}")
        lines = [f"[{cls} class hierarchy]"]
        for h in hierarchy:
            lines.append(f"  {cls} → subClassOf → {h['super']}")
        parts.append("\n".join(lines))

    # Zone types
    if config.get("zone_types"):
        zones = sparql_query(
            """SELECT ?type (COUNT(?z) AS ?cnt) WHERE {
                ?z rdf:type ?type . ?type rdfs:subClassOf :Zone .
            } GROUP BY ?type"""
        )
        hierarchy = sparql_query("SELECT ?sub WHERE { ?sub rdfs:subClassOf :Zone }")
        lines = ["[Zone types]"]
        for z in zones:
            lines.append(f"  {z['type']}: {z['cnt']} instances")
        lines.append("[Zone subclasses]")
        for h in hierarchy:
            lines.append(f"  {h['sub']} → subClassOf → Zone")
        parts.append("\n".join(lines))

    # Scene overview
    if config.get("scene_overview"):
        key_objects = sparql_query(
            """SELECT ?obj ?type ?zone ?comment WHERE {
                ?obj rdfs:comment ?comment ; :inZone ?zone ; rdf:type ?type .
                FILTER(?type != owl:NamedIndividual)
                FILTER NOT EXISTS { ?obj rdf:type ?sub . ?sub rdfs:subClassOf ?type . FILTER(?sub != ?type) }
            } LIMIT 15"""
        )
        lines = [f"[Scene overview: {len(key_objects)} key objects]"]
        for o in key_objects:
            lines.append(f"  {o['obj']} type={o['type']}, zone={o['zone']}")
            lines.append(f"    desc: {o['comment'][:150]}")
        parts.append("\n".join(lines))

    return "\n\n".join(p for p in parts if p)


# (Old per-category KG retrieval functions removed — replaced by entity-driven approach above)


class _REMOVED_OLD_KG:
    """Placeholder — old functions removed."""
    pass


def _kg_spatial_context(qid: int) -> str:
    lines = []
    if qid in (1, 2, 4):
        # Nearby objects to ego
        nearby = sparql_query(
            """SELECT ?obj ?type ?zone ?x ?y ?speed ?state ?comment WHERE {
                ?obj :inZone ?zone ; :x ?x ; :y ?y ; rdf:type ?type .
                OPTIONAL { ?obj :speed ?speed }
                OPTIONAL { ?obj :state ?state }
                OPTIONAL { ?obj rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual)
            } LIMIT 25"""
        )
        for o in nearby:
            line = f"  {o['obj']} (type={o['type']}, zone={o['zone']}, pos=({o['x']},{o['y']})"
            if o.get("speed"):
                line += f", speed={o['speed']}km/h"
            if o.get("state"):
                line += f", state={o['state']}"
            line += ")"
            if o.get("comment"):
                line += f"\n    desc: {o['comment'][:200]}"
            lines.append(line)
    elif qid == 3:
        # Object count per zone
        counts = sparql_query(
            """SELECT ?zone (COUNT(?obj) AS ?cnt) WHERE {
                ?obj :inZone ?zone .
            } GROUP BY ?zone ORDER BY DESC(?cnt) LIMIT 10"""
        )
        lines.append("[Object density per zone]")
        for c in counts:
            lines.append(f"  {c['zone']}: {c['cnt']} objects")
    elif qid == 5:
        # intersection_2 connectivity + objects
        conns = sparql_query(
            """SELECT ?conn ?zone2 ?open WHERE {
                ?conn :isConnectionOf :intersection_2 ;
                      :isConnectionOf ?zone2 ;
                      :isOpen ?open .
                FILTER(?zone2 != :intersection_2)
            }"""
        )
        lines.append("[Connections to intersection_2]")
        for c in conns:
            lines.append(f"  {c['conn']} → {c['zone2']} (open={c['open']})")
        objs = sparql_query(
            """SELECT ?obj ?type ?state ?comment WHERE {
                ?obj :inZone :intersection_2 ; rdf:type ?type .
                OPTIONAL { ?obj :state ?state }
                OPTIONAL { ?obj rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual)
            }"""
        )
        lines.append("[Objects at intersection_2]")
        for o in objs:
            line = f"  {o['obj']} (type={o['type']}"
            if o.get("state"):
                line += f", state={o['state']}"
            line += ")"
            if o.get("comment"):
                line += f"\n    desc: {o['comment'][:200]}"
            lines.append(line)
    return "\n".join(lines)


def _kg_identification_context(qid: int) -> str:
    lines = []
    if qid == 6:
        vehicles = sparql_query(
            """SELECT ?v ?type ?speed ?state ?comment WHERE {
                ?v rdf:type/rdfs:subClassOf* :Vehicle ; rdf:type ?type ;
                   :speed ?speed ; :state ?state .
                OPTIONAL { ?v rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual && ?type != :Vehicle && ?type != :Object && ?type != :PhysicalObject)
            }"""
        )
        lines.append(f"[All vehicles: {len(vehicles)} found]")
        for v in vehicles:
            line = f"  {v['v']} type={v['type']}, speed={v['speed']}km/h, state={v['state']}"
            if v.get("comment"):
                line += f"\n    desc: {v['comment'][:150]}"
            lines.append(line)
    elif qid == 7:
        tl = sparql_query(
            """SELECT ?state ?zone ?comment WHERE {
                :traffic_light_1 :state ?state ; :inZone ?zone .
                OPTIONAL { :traffic_light_1 rdfs:comment ?comment }
            }"""
        )
        if tl:
            t = tl[0]
            lines.append(f"[traffic_light_1] state={t['state']}, zone={t['zone']}")
            if t.get("comment"):
                lines.append(f"  desc: {t['comment']}")
    elif qid == 8:
        stopped = sparql_query(
            """SELECT ?v ?type ?zone ?comment WHERE {
                ?v :speed ?s ; :inZone ?zone ; rdf:type ?type .
                OPTIONAL { ?v rdfs:comment ?comment }
                FILTER(?s = 0 && ?type != owl:NamedIndividual)
            }"""
        )
        lines.append(f"[Stationary vehicles: {len(stopped)} found]")
        for s in stopped:
            line = f"  {s['v']} type={s['type']}, zone={s['zone']}"
            if s.get("comment"):
                line += f"\n    desc: {s['comment'][:200]}"
            lines.append(line)
    elif qid == 9:
        peds = sparql_query(
            """SELECT ?p ?type ?zone ?speed ?state ?comment WHERE {
                ?p rdf:type/rdfs:subClassOf* :Pedestrian ; rdf:type ?type ;
                   :inZone ?zone ; :speed ?speed ; :state ?state .
                OPTIONAL { ?p rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual && ?type != :Pedestrian && ?type != :Object && ?type != :PhysicalObject)
            }"""
        )
        lines.append(f"[All pedestrians: {len(peds)} found]")
        for p in peds:
            line = f"  {p['p']} type={p['type']}, zone={p['zone']}, speed={p['speed']}km/h, state={p['state']}"
            if p.get("comment"):
                line += f"\n    desc: {p['comment'][:200]}"
            lines.append(line)
    elif qid == 10:
        # Objects in ego's forward path (urban_road_4 → intersection_1 → urban_road_1/3)
        forward = sparql_query(
            """SELECT ?obj ?type ?zone ?speed ?state ?comment WHERE {
                ?obj :inZone ?zone ; rdf:type ?type .
                OPTIONAL { ?obj :speed ?speed }
                OPTIONAL { ?obj :state ?state }
                OPTIONAL { ?obj rdfs:comment ?comment }
                FILTER(?zone IN (:urban_road_4, :intersection_1, :urban_road_1, :urban_road_3) && ?type != owl:NamedIndividual)
            }"""
        )
        lines.append(f"[Objects on ego forward path: {len(forward)} found]")
        for o in forward:
            line = f"  {o['obj']} type={o['type']}, zone={o['zone']}"
            if o.get("speed"):
                line += f", speed={o['speed']}km/h"
            if o.get("state"):
                line += f", state={o['state']}"
            if o.get("comment"):
                line += f"\n    desc: {o['comment'][:200]}"
            lines.append(line)
    return "\n".join(lines)


def _kg_semantic_context(qid: int) -> str:
    lines = []
    if qid == 11:
        # Yellow light + ego context + follower
        tl = sparql_query(
            "SELECT ?state ?comment WHERE { :traffic_light_1 :state ?state . OPTIONAL { :traffic_light_1 rdfs:comment ?comment } }"
        )
        s3 = sparql_query(
            "SELECT ?speed ?comment WHERE { :sedan_3 :speed ?speed . OPTIONAL { :sedan_3 rdfs:comment ?comment } }"
        )
        if tl:
            lines.append(f"[traffic_light_1] state={tl[0]['state']}")
            if tl[0].get("comment"):
                lines.append(f"  desc: {tl[0]['comment']}")
        if s3:
            lines.append(f"[sedan_3] speed={s3[0]['speed']}km/h")
            if s3[0].get("comment"):
                lines.append(f"  desc: {s3[0]['comment']}")
    elif qid == 12:
        truck = sparql_query(
            """SELECT ?zone ?speed ?state ?obstacle ?comment WHERE {
                :truck_4 :inZone ?zone ; :speed ?speed ; :state ?state .
                OPTIONAL { :truck_4 :isObstacle ?obstacle }
                OPTIONAL { :truck_4 rdfs:comment ?comment }
            }"""
        )
        sign = sparql_query(
            "SELECT ?comment WHERE { :no_parking_sign_6 rdfs:comment ?comment }"
        )
        if truck:
            t = truck[0]
            lines.append(f"[truck_4] zone={t['zone']}, speed={t['speed']}, state={t['state']}, obstacle={t.get('obstacle','N/A')}")
            if t.get("comment"):
                lines.append(f"  desc: {t['comment']}")
        if sign and sign[0].get("comment"):
            lines.append(f"[no_parking_sign_6] desc: {sign[0]['comment']}")
    elif qid == 13:
        sz_objs = sparql_query(
            """SELECT ?obj ?type ?speed ?state ?comment WHERE {
                ?obj :inZone :school_zone_1 ; rdf:type ?type .
                OPTIONAL { ?obj :speed ?speed }
                OPTIONAL { ?obj :state ?state }
                OPTIONAL { ?obj rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual)
            }"""
        )
        sz_info = sparql_query(
            "SELECT ?limit WHERE { :school_zone_1 :speedLimit ?limit }"
        )
        if sz_info:
            lines.append(f"[school_zone_1] speedLimit={sz_info[0]['limit']}km/h")
        lines.append(f"[Objects in school_zone_1: {len(sz_objs)}]")
        for o in sz_objs:
            line = f"  {o['obj']} type={o['type']}"
            if o.get("state"):
                line += f", state={o['state']}"
            if o.get("comment"):
                line += f"\n    desc: {o['comment'][:200]}"
            lines.append(line)
    elif qid == 14:
        s3 = sparql_query(
            "SELECT ?speed ?x ?y ?comment WHERE { :sedan_3 :speed ?speed ; :x ?x ; :y ?y . OPTIONAL { :sedan_3 rdfs:comment ?comment } }"
        )
        ego = sparql_query(
            "SELECT ?speed ?x ?y WHERE { :ego_car :speed ?speed ; :x ?x ; :y ?y }"
        )
        if s3:
            lines.append(f"[sedan_3] speed={s3[0]['speed']}km/h, pos=({s3[0]['x']},{s3[0]['y']})")
            if s3[0].get("comment"):
                lines.append(f"  desc: {s3[0]['comment']}")
        if ego:
            lines.append(f"[ego_car] speed={ego[0]['speed']}km/h, pos=({ego[0]['x']},{ego[0]['y']})")
    elif qid == 15:
        conn = sparql_query(
            """SELECT ?conn ?open WHERE {
                ?conn :isConnectionOf :intersection_3 ;
                      :isConnectionOf :side_road_1 ;
                      :isOpen ?open .
            }"""
        )
        if conn:
            lines.append(f"[Connection intersection_3↔side_road_1] conn={conn[0]['conn']}, open={conn[0]['open']}")
        # Additional context from KG
        i3_info = sparql_query(
            "SELECT ?limit WHERE { :intersection_3 :speedLimit ?limit }"
        )
        if i3_info:
            lines.append(f"[intersection_3] speedLimit={i3_info[0]['limit']}km/h")
        stop = sparql_query(
            "SELECT ?comment WHERE { :stop_sign_4 rdfs:comment ?comment }"
        )
        if stop and stop[0].get("comment"):
            lines.append(f"[stop_sign_4] desc: {stop[0]['comment']}")
    return "\n".join(lines)


def _kg_hierarchy_context(qid: int) -> str:
    lines = []
    if qid == 16:
        tc = sparql_query(
            """SELECT ?obj ?type WHERE {
                ?obj rdf:type/rdfs:subClassOf* :TrafficControl ; rdf:type ?type .
                FILTER(?type != owl:NamedIndividual && ?type != :TrafficControl && ?type != :Object && ?type != :PhysicalObject)
            }"""
        )
        hierarchy = sparql_query(
            """SELECT ?sub ?super WHERE {
                ?sub rdfs:subClassOf ?super .
                FILTER(?super IN (:TrafficControl, :TrafficLight, :TrafficSign))
            }"""
        )
        lines.append(f"[TrafficControl objects: {len(tc)} found]")
        for t in tc:
            lines.append(f"  {t['obj']} type={t['type']}")
        lines.append("[Class hierarchy]")
        for h in hierarchy:
            lines.append(f"  {h['sub']} → subClassOf → {h['super']}")
    elif qid == 17:
        hierarchy = sparql_query(
            """SELECT ?super WHERE {
                :Crosswalk rdfs:subClassOf+ ?super .
            }"""
        )
        lines.append("[Crosswalk class hierarchy]")
        for h in hierarchy:
            lines.append(f"  Crosswalk → subClassOf → {h['super']}")
        cw = sparql_query(
            """SELECT ?obj ?zone ?comment WHERE {
                ?obj rdf:type :Crosswalk ; :inZone ?zone .
                OPTIONAL { ?obj rdfs:comment ?comment }
            }"""
        )
        for c in cw:
            line = f"[{c['obj']}] zone={c['zone']}"
            if c.get("comment"):
                line += f"\n  desc: {c['comment'][:200]}"
            lines.append(line)
    elif qid == 18:
        groups = sparql_query(
            """SELECT ?parentClass (COUNT(?obj) AS ?cnt) WHERE {
                ?obj rdf:type ?type .
                ?type rdfs:subClassOf ?parentClass .
                FILTER(?parentClass IN (:Vehicle, :TrafficLight, :TrafficSign, :Pedestrian, :Obstacle, :Infrastructure))
            } GROUP BY ?parentClass ORDER BY DESC(?cnt)"""
        )
        lines.append("[Objects grouped by parent class]")
        for g in groups:
            lines.append(f"  {g['parentClass']}: {g['cnt']} objects")
    elif qid == 19:
        zones = sparql_query(
            """SELECT ?type (COUNT(?z) AS ?cnt) WHERE {
                ?z rdf:type ?type .
                ?type rdfs:subClassOf :Zone .
            } GROUP BY ?type"""
        )
        hierarchy = sparql_query(
            "SELECT ?sub WHERE { ?sub rdfs:subClassOf :Zone }"
        )
        lines.append("[Zone types in scene]")
        for z in zones:
            lines.append(f"  {z['type']}: {z['cnt']} instances")
        lines.append("[Zone subclasses of Zone]")
        for h in hierarchy:
            lines.append(f"  {h['sub']}")
    elif qid == 20:
        ego_classes = sparql_query(
            "SELECT ?super WHERE { :EgoVehicle rdfs:subClassOf ?super }"
        )
        lines.append("[EgoVehicle superclasses]")
        for e in ego_classes:
            lines.append(f"  EgoVehicle → subClassOf → {e['super']}")
        agent_info = sparql_query(
            "SELECT ?sub WHERE { ?sub rdfs:subClassOf :Agent }"
        )
        vehicle_info = sparql_query(
            "SELECT ?sub WHERE { ?sub rdfs:subClassOf :Vehicle }"
        )
        lines.append("[Agent subclasses]")
        for a in agent_info:
            lines.append(f"  {a['sub']}")
        lines.append("[Vehicle subclasses]")
        for v in vehicle_info:
            lines.append(f"  {v['sub']}")
    return "\n".join(lines)


def _kg_reasoning_context(qid: int) -> str:
    """Context for safety and dilemma queries — comprehensive."""
    lines = []

    if qid in (21, 26):
        # Yellow light dilemma — full context
        for entity, name in [
            ("traffic_light_1", "traffic_light_1"),
            ("sedan_3", "sedan_3 (following)"),
            ("emergency_vehicle_12", "emergency_vehicle_12"),
            ("ego_car", "ego_car"),
        ]:
            data = sparql_query(
                f"""SELECT ?zone ?x ?y ?speed ?state ?comment WHERE {{
                    :{entity} :inZone ?zone ; :x ?x ; :y ?y .
                    OPTIONAL {{ :{entity} :speed ?speed }}
                    OPTIONAL {{ :{entity} :state ?state }}
                    OPTIONAL {{ :{entity} rdfs:comment ?comment }}
                }}"""
            )
            if data:
                d = data[0]
                line = f"[{name}] zone={d['zone']}, pos=({d['x']},{d['y']})"
                if d.get("speed"):
                    line += f", speed={d['speed']}km/h"
                if d.get("state"):
                    line += f", state={d['state']}"
                if d.get("comment"):
                    line += f"\n  desc: {d['comment']}"
                lines.append(line)
    elif qid == 22:
        # Pedestrian risk at intersection_2
        for entity in ["bus_6", "pedestrian_1", "motorcycle_8", "crosswalk_1", "traffic_light_2"]:
            data = sparql_query(
                f"""SELECT ?zone ?speed ?state ?comment WHERE {{
                    :{entity} :inZone ?zone .
                    OPTIONAL {{ :{entity} :speed ?speed }}
                    OPTIONAL {{ :{entity} :state ?state }}
                    OPTIONAL {{ :{entity} rdfs:comment ?comment }}
                }}"""
            )
            if data:
                d = data[0]
                line = f"[{entity}] zone={d['zone']}"
                if d.get("speed"):
                    line += f", speed={d['speed']}km/h"
                if d.get("state"):
                    line += f", state={d['state']}"
                if d.get("comment"):
                    line += f"\n  desc: {d['comment']}"
                lines.append(line)
    elif qid == 23:
        # School zone danger
        sz_objs = sparql_query(
            """SELECT ?obj ?type ?speed ?state ?comment WHERE {
                ?obj :inZone :school_zone_1 ; rdf:type ?type .
                OPTIONAL { ?obj :speed ?speed }
                OPTIONAL { ?obj :state ?state }
                OPTIONAL { ?obj rdfs:comment ?comment }
                FILTER(?type != owl:NamedIndividual)
            }"""
        )
        lines.append(f"[school_zone_1 objects: {len(sz_objs)}]")
        for o in sz_objs:
            line = f"  {o['obj']} type={o['type']}"
            if o.get("state"):
                line += f", state={o['state']}"
            if o.get("comment"):
                line += f"\n    desc: {o['comment'][:250]}"
            lines.append(line)
    elif qid == 24:
        # Debris on road
        for entity in ["debris_3", "pedestrian_4"]:
            data = sparql_query(
                f"""SELECT ?zone ?speed ?state ?comment WHERE {{
                    :{entity} :inZone ?zone .
                    OPTIONAL {{ :{entity} :speed ?speed }}
                    OPTIONAL {{ :{entity} :state ?state }}
                    OPTIONAL {{ :{entity} rdfs:comment ?comment }}
                }}"""
            )
            if data:
                d = data[0]
                line = f"[{entity}] zone={d['zone']}"
                if d.get("speed"):
                    line += f", speed={d['speed']}km/h"
                if d.get("comment"):
                    line += f"\n  desc: {d['comment']}"
                lines.append(line)
        zone_info = sparql_query(
            "SELECT ?limit WHERE { :wet_road_section_1 :speedLimit ?limit }"
        )
        if zone_info:
            lines.append(f"[wet_road_section_1] speedLimit={zone_info[0]['limit']}km/h")
    elif qid == 25:
        # Overall scene risk — sample key objects
        key_objects = sparql_query(
            """SELECT ?obj ?type ?zone ?comment WHERE {
                ?obj rdfs:comment ?comment ; :inZone ?zone ; rdf:type ?type .
                FILTER(?type != owl:NamedIndividual)
            } LIMIT 15"""
        )
        lines.append(f"[Scene overview: {len(key_objects)} key objects]")
        for o in key_objects:
            lines.append(f"  {o['obj']} type={o['type']}, zone={o['zone']}")
            lines.append(f"    desc: {o['comment'][:150]}")
    elif qid == 27:
        # Speed limit dilemma
        for entity in ["speed_limit_sign_3", "sedan_10", "sedan_11", "truck_5"]:
            data = sparql_query(
                f"""SELECT ?zone ?speed ?state ?comment WHERE {{
                    :{entity} :inZone ?zone .
                    OPTIONAL {{ :{entity} :speed ?speed }}
                    OPTIONAL {{ :{entity} :state ?state }}
                    OPTIONAL {{ :{entity} rdfs:comment ?comment }}
                }}"""
            )
            if data:
                d = data[0]
                line = f"[{entity}] zone={d['zone']}"
                if d.get("speed"):
                    line += f", speed={d['speed']}km/h"
                if d.get("state"):
                    line += f", state={d['state']}"
                if d.get("comment"):
                    line += f"\n  desc: {d['comment']}"
                lines.append(line)
    elif qid == 28:
        # Truck overtake
        for entity in ["truck_4", "no_parking_sign_6"]:
            data = sparql_query(
                f"""SELECT ?zone ?speed ?state ?comment WHERE {{
                    :{entity} :inZone ?zone .
                    OPTIONAL {{ :{entity} :speed ?speed }}
                    OPTIONAL {{ :{entity} :state ?state }}
                    OPTIONAL {{ :{entity} rdfs:comment ?comment }}
                }}"""
            )
            if data:
                d = data[0]
                line = f"[{entity}] zone={d['zone']}"
                if d.get("speed"):
                    line += f", speed={d['speed']}km/h"
                if d.get("comment"):
                    line += f"\n  desc: {d['comment']}"
                lines.append(line)
    elif qid == 29:
        # Multi-agent at intersection_2
        for entity in ["bus_6", "pedestrian_1", "motorcycle_8", "crosswalk_1", "traffic_light_2", "bus_stop_1"]:
            data = sparql_query(
                f"""SELECT ?zone ?speed ?state ?comment WHERE {{
                    :{entity} :inZone ?zone .
                    OPTIONAL {{ :{entity} :speed ?speed }}
                    OPTIONAL {{ :{entity} :state ?state }}
                    OPTIONAL {{ :{entity} rdfs:comment ?comment }}
                }}"""
            )
            if data:
                d = data[0]
                line = f"[{entity}] zone={d['zone']}"
                if d.get("speed"):
                    line += f", speed={d['speed']}km/h"
                if d.get("state"):
                    line += f", state={d['state']}"
                if d.get("comment"):
                    line += f"\n  desc: {d['comment']}"
                lines.append(line)
    elif qid == 30:
        # Merge cooperation
        for entity in ["sedan_15", "yield_sign_5", "traffic_light_7", "sedan_10", "sedan_11", "guardrail_2"]:
            data = sparql_query(
                f"""SELECT ?zone ?speed ?state ?comment WHERE {{
                    :{entity} :inZone ?zone .
                    OPTIONAL {{ :{entity} :speed ?speed }}
                    OPTIONAL {{ :{entity} :state ?state }}
                    OPTIONAL {{ :{entity} rdfs:comment ?comment }}
                }}"""
            )
            if data:
                d = data[0]
                line = f"[{entity}] zone={d['zone']}"
                if d.get("speed"):
                    line += f", speed={d['speed']}km/h"
                if d.get("state"):
                    line += f", state={d['state']}"
                if d.get("comment"):
                    line += f"\n  desc: {d['comment']}"
                lines.append(line)
    return "\n".join(lines)


# ============================================================
# 3DSG Entity Map (same structure, different retrieval)
# ============================================================
DSG_ENTITY_MAP = {
    1: {"nearby_ego": True},
    2: {"nearby_ego": True},
    3: {"zones": ["intersection_2"], "zone_objects": True, "adjacent_zones": ["bus_lane_1"]},
    4: {"all_vehicles": True},
    5: {"entities": ["traffic_light_1"]},
    6: {"all_pedestrians": True},
    7: {"entities": ["traffic_light_1", "sedan_3"]},
    8: {"entities": ["truck_4", "no_parking_sign_6"]},
    9: {"zones": ["school_zone_1"], "zone_objects": True},
    10: {"entities": ["sedan_3"]},
    11: {"zones": ["intersection_3", "side_road_1"]},
    12: {"entities": ["traffic_light_6", "construction_cone_1", "construction_cone_2", "construction_sign_9"], "zones": ["construction_zone_1"]},
    13: {"entities": ["traffic_light_7", "sedan_15", "yield_sign_5", "sedan_10", "sedan_11"], "zones": ["merge_zone_1"]},
    14: {"all_traffic_control": True},
    15: {"entities": ["crosswalk_1", "crosswalk_2"], "similar_classes": ["crosswalk", "guardrail", "bus_stop", "median"]},
    16: {"zone_types": True},
    17: {"entities": ["traffic_light_1", "sedan_3", "emergency_vehicle_12"]},
    18: {"entities": ["bus_6", "pedestrian_1", "motorcycle_8", "crosswalk_1", "traffic_light_2", "bus_stop_1"]},
    19: {"zones": ["school_zone_1"], "zone_objects": True, "compare_zone": "intersection_1"},
    20: {"entities": ["debris_3", "pedestrian_4"], "zones": ["wet_road_section_1"]},
    21: {"entities": ["sedan_2", "sedan_1"]},
    22: {"entities": ["elderly_pedestrian_3", "stop_sign_4", "traffic_light_3"], "zones": ["intersection_3"]},
    23: {"scene_overview": True},
    24: {"entities": ["traffic_light_1", "sedan_3", "emergency_vehicle_12"]},
    25: {"entities": ["speed_limit_sign_3", "sedan_10", "sedan_11", "truck_5"], "zones": ["highway_section_2"]},
    26: {"entities": ["truck_4", "no_parking_sign_6"]},
    27: {"entities": ["bus_6", "pedestrian_1", "motorcycle_8", "crosswalk_1", "traffic_light_2", "bus_stop_1"]},
    28: {"entities": ["sedan_15", "yield_sign_5", "traffic_light_7", "sedan_10", "sedan_11", "guardrail_2"], "zones": ["merge_zone_1"]},
    29: {"entities": ["emergency_vehicle_12", "traffic_light_1", "sedan_3"]},
    30: {"entities": ["bus_7", "child_pedestrian_2", "child_pedestrian_5", "bicycle_9", "crosswalk_2", "speed_limit_sign_2"]},
}


# ============================================================
# Context Retrieval: 3DSG Mode (Scene Graph) — entity-driven
# ============================================================
def _dsg_format_object(obj: dict, include_edges: bool = True) -> str:
    """Format a scene graph object node."""
    a = obj["attributes"]
    line = f"  {obj['id']} (class={obj['class']}, zone={obj['in_zone']}, pos={obj['location']}"
    for key in ["speed_kmh", "state", "signal_state", "orientation", "color", "posted_value",
                "hazard_lights", "doors", "flashing_lights", "sirens", "emergency_lights",
                "lane_position", "lateral_deviation", "group_size", "walking_aid",
                "reflective_gear", "material", "size"]:
        if key in a:
            line += f", {key}={a[key]}"
    line += ")"
    if include_edges:
        edges = [e for e in SG_EDGES if e["source"] == obj["id"] or e["target"] == obj["id"]]
        for e in edges[:3]:
            edge_str = f"    edge: {e['source']} --{e['relation']}"
            if "distance_m" in e:
                edge_str += f"({e['distance_m']}m)"
            edge_str += f"--> {e['target']}"
            line += "\n" + edge_str
    return line


def retrieve_dsg_context(query_id: int, question: str) -> str:
    """Retrieve structured 3DSG context via scene graph traversal."""
    parts = []
    config = DSG_ENTITY_MAP.get(query_id, {})

    # Always include ego
    ego = SG_OBJECTS.get("ego_car")
    if ego:
        a = ego["attributes"]
        parts.append(
            f"[Ego Vehicle] zone={ego['in_zone']}, pos={ego['location']}, "
            f"speed={a['speed_kmh']}km/h, state={a['state']}, orientation={a['orientation']}, color={a['color']}"
        )

    # Specific entities
    for eid in config.get("entities", []):
        obj = SG_OBJECTS.get(eid)
        if obj:
            parts.append(_dsg_format_object(obj))

    # Nearby ego
    if config.get("nearby_ego"):
        ex, ey = ego["location"][0], ego["location"][1]
        nearby = []
        for obj in scene_graph["objects"]:
            if obj["id"] == "ego_car":
                continue
            ox, oy = obj["location"][0], obj["location"][1]
            dist = math.sqrt((ox - ex)**2 + (oy - ey)**2)
            if dist <= 60:
                nearby.append((dist, obj))
        nearby.sort(key=lambda x: x[0])
        lines = [f"[Objects near ego (within 60m): {len(nearby)}]"]
        for dist, obj in nearby[:15]:
            lines.append(f"{_dsg_format_object(obj)} dist={dist:.0f}m")
        parts.append("\n".join(lines))

    # Zone objects
    for zone_id in config.get("zones", []):
        zone = SG_ZONES.get(zone_id)
        if zone:
            lines = [f"[{zone_id}] type={zone['type']}, speed_limit={zone['speed_limit_kmh']}km/h, connected={zone['connected_zones']}"]
            if config.get("zone_objects"):
                objs = [o for o in scene_graph["objects"] if o["in_zone"] == zone_id]
                lines.append(f"  Objects: {len(objs)}")
                for obj in objs:
                    lines.append(_dsg_format_object(obj))
            parts.append("\n".join(lines))

    # Adjacent zones
    for zone_id in config.get("adjacent_zones", []):
        objs = [o for o in scene_graph["objects"] if o["in_zone"] == zone_id]
        if objs:
            lines = [f"[Adjacent {zone_id}: {len(objs)} objects]"]
            for obj in objs:
                lines.append(_dsg_format_object(obj))
            parts.append("\n".join(lines))

    # Compare zone
    if config.get("compare_zone"):
        cz = config["compare_zone"]
        zone = SG_ZONES.get(cz)
        objs = [o for o in scene_graph["objects"] if o["in_zone"] == cz]
        if zone:
            lines = [f"[{cz} for comparison: {len(objs)} objects, speed_limit={zone['speed_limit_kmh']}km/h]"]
            for obj in objs:
                lines.append(_dsg_format_object(obj, include_edges=False))
            parts.append("\n".join(lines))

    # All vehicles
    if config.get("all_vehicles"):
        vehicles = [o for o in scene_graph["objects"] if o["class"] in (
            "ego_vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
            "emergency_vehicle", "parked_vehicle"
        )]
        lines = [f"[All vehicles: {len(vehicles)}]"]
        for v in vehicles:
            lines.append(_dsg_format_object(v, include_edges=False))
        parts.append("\n".join(lines))

    # All pedestrians
    if config.get("all_pedestrians"):
        peds = [o for o in scene_graph["objects"] if "pedestrian" in o["class"]]
        lines = [f"[All pedestrians: {len(peds)}]"]
        for p in peds:
            lines.append(_dsg_format_object(p))
        parts.append("\n".join(lines))

    # All traffic control
    if config.get("all_traffic_control"):
        tc = [o for o in scene_graph["objects"] if o["class"] in (
            "traffic_light", "speed_limit_sign", "stop_sign", "yield_sign",
            "no_parking_sign", "lane_merge_sign", "school_zone_sign", "construction_sign"
        )]
        lines = [f"[Traffic control objects: {len(tc)}]"]
        by_class = {}
        for t in tc:
            by_class.setdefault(t["class"], []).append(t["id"])
        for cls, ids in by_class.items():
            lines.append(f"  {cls}: {ids}")
        parts.append("\n".join(lines))

    # Similar classes (for hierarchy questions)
    if config.get("similar_classes"):
        infra = [o for o in scene_graph["objects"] if o["class"] in config["similar_classes"]]
        lines = [f"[Fixed infrastructure objects: {len(infra)}]"]
        for i in infra:
            lines.append(f"  {i['id']} class={i['class']}, zone={i['in_zone']}")
        parts.append("\n".join(lines))

    # Zone types
    if config.get("zone_types"):
        zone_types = {}
        for z in scene_graph["layers"]["zones"]:
            zone_types.setdefault(z["type"], []).append(z["id"])
        lines = [f"[Zone types: {len(zone_types)}]"]
        for zt, ids in sorted(zone_types.items()):
            avg_limit = np.mean([SG_ZONES[z]["speed_limit_kmh"] for z in ids])
            lines.append(f"  {zt}: {len(ids)} zones, avg speed_limit={avg_limit:.0f}km/h — {ids}")
        parts.append("\n".join(lines))

    # Scene overview
    if config.get("scene_overview"):
        lines = [f"[Scene: {len(scene_graph['objects'])} objects, {len(scene_graph['layers']['zones'])} zones]"]
        class_counts = {}
        for obj in scene_graph["objects"]:
            class_counts[obj["class"]] = class_counts.get(obj["class"], 0) + 1
        lines.append("[Object classes]")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls}: {cnt}")
        lines.append(f"[Spatial relationships: {len(SG_EDGES)}]")
        for e in SG_EDGES[:10]:
            lines.append(f"  {e['source']} --{e['relation']}--> {e['target']}" +
                         (f" ({e['distance_m']}m)" if "distance_m" in e else ""))
        parts.append("\n".join(lines))

    # Add ego distance for key entities
    if config.get("entities") and ego:
        ex, ey = ego["location"][0], ego["location"][1]
        dist_lines = []
        for eid in config["entities"]:
            obj = SG_OBJECTS.get(eid)
            if obj:
                ox, oy = obj["location"][0], obj["location"][1]
                dist = math.sqrt((ox - ex)**2 + (oy - ey)**2)
                dist_lines.append(f"  ego→{eid}: {dist:.0f}m")
        if dist_lines:
            parts.append("[Distances from ego]\n" + "\n".join(dist_lines))

    return "\n\n".join(p for p in parts if p)


def _OLD_format_sg_object(obj: dict, include_edges: bool = True) -> str:
    """Format a scene graph object node."""
    a = obj["attributes"]
    line = f"  {obj['id']} (class={obj['class']}, zone={obj['in_zone']}, pos={obj['location']}"
    for key in ["speed_kmh", "state", "signal_state", "orientation", "color", "posted_value"]:
        if key in a:
            line += f", {key}={a[key]}"
    line += ")"
    if include_edges:
        edges = [e for e in SG_EDGES if e["source"] == obj["id"] or e["target"] == obj["id"]]
        for e in edges[:3]:
            other = e["target"] if e["source"] == obj["id"] else e["source"]
            edge_str = f"    edge: {e['source']} --{e['relation']}"
            if "distance_m" in e:
                edge_str += f"({e['distance_m']}m)"
            edge_str += f"--> {e['target']}"
            line += "\n" + edge_str
    return line


def _dsg_get_objects_in_zone(zone_id: str) -> list:
    return [o for o in scene_graph["objects"] if o["in_zone"] == zone_id]


def _dsg_spatial_context(qid: int) -> str:
    lines = []
    ego = SG_OBJECTS["ego_car"]
    ex, ey = ego["location"][0], ego["location"][1]

    if qid in (1, 2, 4):
        # Nearby objects with distances
        nearby = []
        for obj in scene_graph["objects"]:
            if obj["id"] == "ego_car":
                continue
            ox, oy = obj["location"][0], obj["location"][1]
            dist = math.sqrt((ox - ex)**2 + (oy - ey)**2)
            if dist <= 60:
                nearby.append((dist, obj))
        nearby.sort(key=lambda x: x[0])
        lines.append(f"[Objects near ego (within 60m): {len(nearby)}]")
        for dist, obj in nearby[:15]:
            lines.append(f"{_format_sg_object(obj)} dist={dist:.0f}m")
    elif qid == 3:
        # Object density per zone
        zone_counts = {}
        for obj in scene_graph["objects"]:
            z = obj["in_zone"]
            zone_counts[z] = zone_counts.get(z, 0) + 1
        sorted_zones = sorted(zone_counts.items(), key=lambda x: -x[1])
        lines.append("[Object density per zone]")
        for z, cnt in sorted_zones[:10]:
            lines.append(f"  {z}: {cnt} objects")
    elif qid == 5:
        # intersection_2 layout
        zone = SG_ZONES["intersection_2"]
        lines.append(f"[intersection_2] type={zone['type']}, pos={zone['location']}, speed_limit={zone['speed_limit_kmh']}km/h")
        lines.append(f"  connected_zones: {zone['connected_zones']}")
        objs = _dsg_get_objects_in_zone("intersection_2")
        lines.append(f"[Objects at intersection_2: {len(objs)}]")
        for obj in objs:
            lines.append(_format_sg_object(obj))
        # Also check bus_lane_1 (adjacent)
        bl_objs = _dsg_get_objects_in_zone("bus_lane_1")
        if bl_objs:
            lines.append(f"[Adjacent bus_lane_1: {len(bl_objs)} objects]")
            for obj in bl_objs:
                lines.append(_format_sg_object(obj))
    return "\n".join(lines)


def _dsg_identification_context(qid: int) -> str:
    lines = []
    if qid == 6:
        vehicles = [o for o in scene_graph["objects"] if o["class"] in (
            "ego_vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
            "emergency_vehicle", "parked_vehicle"
        )]
        lines.append(f"[All vehicles: {len(vehicles)}]")
        for v in vehicles:
            lines.append(_format_sg_object(v, include_edges=False))
    elif qid == 7:
        tl = SG_OBJECTS.get("traffic_light_1")
        if tl:
            lines.append(_format_sg_object(tl))
            zone = SG_ZONES.get(tl["in_zone"])
            if zone:
                lines.append(f"  zone info: {zone['type']}, speed_limit={zone['speed_limit_kmh']}km/h")
    elif qid == 8:
        stopped = [o for o in scene_graph["objects"]
                   if o["attributes"].get("speed_kmh", -1) == 0 or o["attributes"].get("state") == "stationary"]
        lines.append(f"[Stationary objects: {len(stopped)}]")
        for s in stopped:
            lines.append(_format_sg_object(s))
    elif qid == 9:
        peds = [o for o in scene_graph["objects"] if "pedestrian" in o["class"]]
        lines.append(f"[All pedestrians: {len(peds)}]")
        for p in peds:
            lines.append(_format_sg_object(p))
    elif qid == 10:
        ego = SG_OBJECTS["ego_car"]
        ex = ego["location"][0]
        forward = [o for o in scene_graph["objects"]
                   if o["location"][0] > ex - 5 and o["location"][0] < ex + 100
                   and abs(o["location"][1] - ego["location"][1]) < 20
                   and o["id"] != "ego_car"]
        forward.sort(key=lambda o: o["location"][0])
        lines.append(f"[Objects on forward path: {len(forward)}]")
        for obj in forward[:10]:
            lines.append(_format_sg_object(obj))
    return "\n".join(lines)


def _dsg_semantic_context(qid: int) -> str:
    lines = []
    if qid == 11:
        tl = SG_OBJECTS["traffic_light_1"]
        lines.append(_format_sg_object(tl))
        s3 = SG_OBJECTS["sedan_3"]
        lines.append(_format_sg_object(s3))
        ego = SG_OBJECTS["ego_car"]
        dist = math.sqrt((tl["location"][0]-ego["location"][0])**2 + (tl["location"][1]-ego["location"][1])**2)
        lines.append(f"  ego→traffic_light_1 distance: {dist:.0f}m")
    elif qid == 12:
        truck = SG_OBJECTS["truck_4"]
        lines.append(_format_sg_object(truck))
        sign = SG_OBJECTS["no_parking_sign_6"]
        lines.append(_format_sg_object(sign))
    elif qid == 13:
        zone = SG_ZONES["school_zone_1"]
        lines.append(f"[school_zone_1] type={zone['type']}, speed_limit={zone['speed_limit_kmh']}km/h")
        objs = _dsg_get_objects_in_zone("school_zone_1")
        lines.append(f"[Objects: {len(objs)}]")
        for obj in objs:
            lines.append(_format_sg_object(obj))
    elif qid == 14:
        s3 = SG_OBJECTS["sedan_3"]
        ego = SG_OBJECTS["ego_car"]
        lines.append(_format_sg_object(s3))
        lines.append(_format_sg_object(ego))
        dist = math.sqrt((s3["location"][0]-ego["location"][0])**2 + (s3["location"][1]-ego["location"][1])**2)
        lines.append(f"  sedan_3→ego distance: {dist:.1f}m")
    elif qid == 15:
        z1 = SG_ZONES.get("intersection_3")
        z2 = SG_ZONES.get("side_road_1")
        if z1:
            lines.append(f"[intersection_3] connected_zones={z1['connected_zones']}, speed_limit={z1['speed_limit_kmh']}km/h")
        if z2:
            lines.append(f"[side_road_1] connected_zones={z2['connected_zones']}, speed_limit={z2['speed_limit_kmh']}km/h")
        # Note: 3DSG has connectivity but not open/closed state
        lines.append("  Note: zone connectivity shows these zones are linked")
    return "\n".join(lines)


def _dsg_hierarchy_context(qid: int) -> str:
    lines = []
    if qid == 16:
        tc = [o for o in scene_graph["objects"] if o["class"] in (
            "traffic_light", "speed_limit_sign", "stop_sign", "yield_sign",
            "no_parking_sign", "lane_merge_sign", "school_zone_sign", "construction_sign"
        )]
        lines.append(f"[Traffic control objects: {len(tc)}]")
        by_class = {}
        for t in tc:
            by_class.setdefault(t["class"], []).append(t["id"])
        for cls, ids in by_class.items():
            lines.append(f"  {cls}: {ids}")
    elif qid == 17:
        cws = [o for o in scene_graph["objects"] if o["class"] == "crosswalk"]
        lines.append(f"[Crosswalk objects: {len(cws)}]")
        for cw in cws:
            lines.append(_format_sg_object(cw))
        infra = [o for o in scene_graph["objects"] if o["class"] in (
            "crosswalk", "guardrail", "bus_stop", "median"
        )]
        lines.append(f"[Similar fixed infrastructure objects: {len(infra)}]")
        for i in infra:
            lines.append(f"  {i['id']} class={i['class']}")
    elif qid == 18:
        groups = {}
        for obj in scene_graph["objects"]:
            groups.setdefault(obj["class"], []).append(obj["id"])
        lines.append(f"[Objects by class ({len(groups)} classes)]")
        for cls in sorted(groups.keys()):
            lines.append(f"  {cls}: {len(groups[cls])} — {groups[cls]}")
    elif qid == 19:
        zone_types = {}
        for z in scene_graph["layers"]["zones"]:
            zone_types.setdefault(z["type"], []).append(z["id"])
        lines.append(f"[Zone types: {len(zone_types)}]")
        for zt, ids in sorted(zone_types.items()):
            avg_limit = np.mean([SG_ZONES[z]["speed_limit_kmh"] for z in ids])
            lines.append(f"  {zt}: {len(ids)} zones, avg speed limit={avg_limit:.0f}km/h — {ids}")
    elif qid == 20:
        ego = SG_OBJECTS["ego_car"]
        lines.append(_format_sg_object(ego))
        lines.append(f"  class: {ego['class']}")
    return "\n".join(lines)


def _dsg_reasoning_context(qid: int) -> str:
    """Context for safety/dilemma from 3DSG — spatial + attributes, no semantic descriptions."""
    lines = []

    if qid in (21, 26):
        # Yellow light dilemma
        for eid in ["ego_car", "traffic_light_1", "sedan_3", "emergency_vehicle_12"]:
            obj = SG_OBJECTS.get(eid)
            if obj:
                lines.append(_format_sg_object(obj))
        ego = SG_OBJECTS["ego_car"]
        tl = SG_OBJECTS["traffic_light_1"]
        dist = math.sqrt((tl["location"][0]-ego["location"][0])**2 + (tl["location"][1]-ego["location"][1])**2)
        lines.append(f"  ego→traffic_light_1 distance: {dist:.0f}m")
        zone = SG_ZONES.get("intersection_1")
        if zone:
            lines.append(f"  intersection_1: speed_limit={zone['speed_limit_kmh']}km/h, connected={zone['connected_zones']}")
    elif qid == 22:
        for eid in ["bus_6", "pedestrian_1", "motorcycle_8", "crosswalk_1", "traffic_light_2"]:
            obj = SG_OBJECTS.get(eid)
            if obj:
                lines.append(_format_sg_object(obj))
    elif qid == 23:
        zone = SG_ZONES["school_zone_1"]
        lines.append(f"[school_zone_1] speed_limit={zone['speed_limit_kmh']}km/h, connected={zone['connected_zones']}")
        objs = _dsg_get_objects_in_zone("school_zone_1")
        for obj in objs:
            lines.append(_format_sg_object(obj))
        # Also show a regular intersection for comparison
        i1_objs = _dsg_get_objects_in_zone("intersection_1")
        lines.append(f"[intersection_1 for comparison: {len(i1_objs)} objects]")
        for obj in i1_objs:
            lines.append(_format_sg_object(obj, include_edges=False))
    elif qid == 24:
        for eid in ["debris_3", "pedestrian_4"]:
            obj = SG_OBJECTS.get(eid)
            if obj:
                lines.append(_format_sg_object(obj))
        zone = SG_ZONES.get("wet_road_section_1")
        if zone:
            lines.append(f"  wet_road_section_1: speed_limit={zone['speed_limit_kmh']}km/h")
    elif qid == 25:
        # Scene overview
        lines.append(f"[Scene: {len(scene_graph['objects'])} objects, {len(scene_graph['layers']['zones'])} zones]")
        zone_types = {}
        for z in scene_graph["layers"]["zones"]:
            zone_types.setdefault(z["type"], []).append(z["id"])
        for zt, ids in zone_types.items():
            lines.append(f"  {zt}: {len(ids)}")
        # Key objects by class
        class_counts = {}
        for obj in scene_graph["objects"]:
            class_counts[obj["class"]] = class_counts.get(obj["class"], 0) + 1
        lines.append("[Object classes]")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls}: {cnt}")
        # Key edges
        lines.append(f"[Spatial relationships: {len(SG_EDGES)}]")
        for e in SG_EDGES[:8]:
            lines.append(f"  {e['source']} --{e['relation']}--> {e['target']}" +
                         (f" ({e['distance_m']}m)" if "distance_m" in e else ""))
    elif qid == 27:
        for eid in ["speed_limit_sign_3", "sedan_10", "sedan_11", "truck_5"]:
            obj = SG_OBJECTS.get(eid)
            if obj:
                lines.append(_format_sg_object(obj, include_edges=False))
        zone = SG_ZONES.get("highway_section_2")
        if zone:
            lines.append(f"  highway_section_2: speed_limit={zone['speed_limit_kmh']}km/h")
    elif qid == 28:
        for eid in ["truck_4", "no_parking_sign_6"]:
            obj = SG_OBJECTS.get(eid)
            if obj:
                lines.append(_format_sg_object(obj))
    elif qid == 29:
        for eid in ["bus_6", "pedestrian_1", "motorcycle_8", "crosswalk_1", "traffic_light_2", "bus_stop_1"]:
            obj = SG_OBJECTS.get(eid)
            if obj:
                lines.append(_format_sg_object(obj))
    elif qid == 30:
        for eid in ["sedan_15", "yield_sign_5", "traffic_light_7", "sedan_10", "sedan_11", "guardrail_2"]:
            obj = SG_OBJECTS.get(eid)
            if obj:
                lines.append(_format_sg_object(obj, include_edges=True))
        zone = SG_ZONES.get("merge_zone_1")
        if zone:
            lines.append(f"  merge_zone_1: speed_limit={zone['speed_limit_kmh']}km/h, connected={zone['connected_zones']}")
    return "\n".join(lines)


# ============================================================
# LLM Answerer
# ============================================================
SYSTEM_PROMPT = """You are an autonomous driving scene understanding agent deployed in a real-time vehicle system.

CRITICAL CONSTRAINT: In a real autonomous driving system, LLM API calls introduce unacceptable latency for safety-critical decisions. Therefore, the scene representation (provided below) must contain ALL information needed for your reasoning. You must answer ONLY based on the provided scene context. Do NOT use any external knowledge, common sense, or information from your training data that is not explicitly present in the context.

If the context does not contain sufficient information to answer the question, state what information is missing and provide the best answer possible with only what is given.

Focus on:
- Identifying relevant objects and their states FROM THE CONTEXT
- Spatial relationships and distances FROM THE CONTEXT
- Safety implications based on information IN THE CONTEXT
- Providing actionable driving recommendations supported by CONTEXT EVIDENCE

Be specific: reference object IDs, speeds, distances, and signal states in your answer.
Keep your answer under 300 words."""


def get_llm_answer(context: str, question: str) -> str:
    """Get answer from GPT-4o-mini given scene context and question."""
    user_msg = f"Scene Context:\n{context}\n\nQuestion: {question}"
    try:
        resp = client.chat.completions.create(
            model=ANSWER_MODEL,
            temperature=ANSWER_TEMP,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


# ============================================================
# LLM Judge
# ============================================================
JUDGE_PROMPT = """You are an expert evaluator for autonomous driving scene understanding.

Score the given answer on a 1-5 scale based on the ground truth key points.

Scoring rubric:
5 - Excellent: All key points addressed with correct reasoning and safety awareness
4 - Good: Most key points covered, minor omissions only
3 - Adequate: Core answer correct but significant gaps in reasoning or detail
2 - Poor: Partially correct but major reasoning errors or safety blind spots
1 - Inadequate: Incorrect, irrelevant, or potentially dangerous recommendation

Evaluate ONLY the factual correctness and reasoning quality.
Do NOT penalize for writing style or brevity if the content is correct.

Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<brief justification>"}"""


def judge_answer(question: str, key_points: list, answer: str) -> dict:
    """Judge an answer against ground truth using GPT-4o."""
    user_msg = (
        f"Question: {question}\n\n"
        f"Key Points (ground truth):\n" +
        "\n".join(f"- {kp}" for kp in key_points) +
        f"\n\nAnswer to evaluate:\n{answer}"
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMP,
            max_tokens=200,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        content = resp.choices[0].message.content.strip()
        # Parse JSON from response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        return {"score": 0, "reason": f"Judge error: {e}"}


# ============================================================
# Main Experiment Runner
# ============================================================
def run_experiment():
    print("=" * 80)
    print("KSAE 2026 Experiment v2: KG vs 3DSG (LLM-in-the-loop)")
    print(f"Model: {ANSWER_MODEL} (answer) / {JUDGE_MODEL} (judge)")
    print(f"Trials: {N_TRIALS} per query per mode")
    print("=" * 80)

    all_results = []
    total_calls = 0

    for q in QUERIES:
        qid = q["id"]
        question = q["question"]
        key_points = q["key_points"]
        category = q["category"]

        print(f"\n--- Q{qid} [{category}] {question[:60]}...")

        q_result = {
            "id": qid,
            "category": category,
            "question": question,
            "kg_scores": [],
            "dsg_scores": [],
            "kg_answers": [],
            "dsg_answers": [],
            "kg_judgments": [],
            "dsg_judgments": [],
            "kg_context": "",
            "dsg_context": "",
        }

        # Retrieve contexts (once per query)
        kg_ctx = retrieve_kg_context(qid, question)
        dsg_ctx = retrieve_dsg_context(qid, question)
        q_result["kg_context"] = kg_ctx
        q_result["dsg_context"] = dsg_ctx

        # Run trials
        for trial in range(N_TRIALS):
            # KG mode
            kg_answer = get_llm_answer(kg_ctx, question)
            kg_judgment = judge_answer(question, key_points, kg_answer)
            q_result["kg_answers"].append(kg_answer)
            q_result["kg_judgments"].append(kg_judgment)
            q_result["kg_scores"].append(kg_judgment.get("score", 0))

            # 3DSG mode
            dsg_answer = get_llm_answer(dsg_ctx, question)
            dsg_judgment = judge_answer(question, key_points, dsg_answer)
            q_result["dsg_answers"].append(dsg_answer)
            q_result["dsg_judgments"].append(dsg_judgment)
            q_result["dsg_scores"].append(dsg_judgment.get("score", 0))

            total_calls += 4  # 2 answers + 2 judgments
            time.sleep(0.2)  # Rate limiting

        # Print per-query summary
        kg_mean = np.mean(q_result["kg_scores"])
        dsg_mean = np.mean(q_result["dsg_scores"])
        kg_std = np.std(q_result["kg_scores"])
        dsg_std = np.std(q_result["dsg_scores"])
        print(f"  KG:  {kg_mean:.1f} ± {kg_std:.1f}  |  3DSG: {dsg_mean:.1f} ± {dsg_std:.1f}")

        all_results.append(q_result)

    # ============================================================
    # Statistical Analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    categories = ["spatial", "identification", "semantic", "hierarchy", "safety", "dilemma"]
    cat_kg_means = {}
    cat_dsg_means = {}

    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        kg_all = [np.mean(r["kg_scores"]) for r in cat_results]
        dsg_all = [np.mean(r["dsg_scores"]) for r in cat_results]
        cat_kg_means[cat] = np.mean(kg_all)
        cat_dsg_means[cat] = np.mean(dsg_all)

    print(f"\n{'Category':<20} {'KG Mean':>8} {'3DSG Mean':>10} {'Delta':>8} {'KG Wins':>8}")
    print("-" * 60)

    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        kg_wins = sum(1 for r in cat_results if np.mean(r["kg_scores"]) > np.mean(r["dsg_scores"]))
        delta = cat_kg_means[cat] - cat_dsg_means[cat]
        print(f"{cat:<20} {cat_kg_means[cat]:>7.2f} {cat_dsg_means[cat]:>9.2f} {delta:>+7.2f} {kg_wins:>5}/{len(cat_results)}")

    # Overall
    all_kg = [np.mean(r["kg_scores"]) for r in all_results]
    all_dsg = [np.mean(r["dsg_scores"]) for r in all_results]
    overall_kg = np.mean(all_kg)
    overall_dsg = np.mean(all_dsg)
    print("-" * 60)
    print(f"{'OVERALL':<20} {overall_kg:>7.2f} {overall_dsg:>9.2f} {overall_kg-overall_dsg:>+7.2f}")

    # Statistical test (Wilcoxon signed-rank)
    print("\n--- Statistical Analysis ---")
    try:
        stat, p_value = stats.wilcoxon(all_kg, all_dsg, alternative="greater")
        print(f"Wilcoxon signed-rank test (KG > 3DSG): W={stat:.1f}, p={p_value:.6f}")
        if p_value < 0.001:
            print("  → Highly significant (p < 0.001)")
        elif p_value < 0.01:
            print("  → Significant (p < 0.01)")
        elif p_value < 0.05:
            print("  → Significant (p < 0.05)")
        else:
            print("  → Not significant (p >= 0.05)")
    except Exception as e:
        print(f"  Wilcoxon test error: {e}")

    # Effect size (rank-biserial correlation)
    try:
        n = len(all_kg)
        diffs = [k - d for k, d in zip(all_kg, all_dsg)]
        r_plus = sum(rank for rank, diff in enumerate(sorted(diffs, key=abs), 1) if diffs[sorted(range(n), key=lambda i: abs(diffs[i]))[rank-1]] > 0)
        r_minus = n * (n + 1) / 2 - r_plus
        r_rb = (r_plus - r_minus) / (n * (n + 1) / 2)
        print(f"Effect size (rank-biserial r): {r_rb:.3f}")
        if abs(r_rb) > 0.5:
            print("  → Large effect")
        elif abs(r_rb) > 0.3:
            print("  → Medium effect")
        else:
            print("  → Small effect")
    except:
        pass

    # Per-category Wilcoxon tests
    print("\n--- Per-Category Tests ---")
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        cat_kg = [np.mean(r["kg_scores"]) for r in cat_results]
        cat_dsg = [np.mean(r["dsg_scores"]) for r in cat_results]
        try:
            _, p = stats.wilcoxon(cat_kg, cat_dsg, alternative="greater")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {cat:<20} p={p:.4f} {sig}")
        except:
            print(f"  {cat:<20} insufficient data for test")

    # Detailed per-query table
    print(f"\n{'Q#':<4} {'Category':<16} {'KG':>6} {'3DSG':>6} {'Δ':>6} {'Winner':>8}")
    print("-" * 50)
    for r in all_results:
        kg_m = np.mean(r["kg_scores"])
        dsg_m = np.mean(r["dsg_scores"])
        delta = kg_m - dsg_m
        winner = "KG" if delta > 0.2 else "3DSG" if delta < -0.2 else "TIE"
        print(f"Q{r['id']:<3} {r['category']:<16} {kg_m:>5.1f} {dsg_m:>5.1f} {delta:>+5.1f} {winner:>8}")

    # Save results
    output = {
        "config": {
            "answer_model": ANSWER_MODEL,
            "judge_model": JUDGE_MODEL,
            "n_trials": N_TRIALS,
            "answer_temp": ANSWER_TEMP,
        },
        "summary": {
            "overall_kg_mean": round(overall_kg, 3),
            "overall_dsg_mean": round(overall_dsg, 3),
            "delta": round(overall_kg - overall_dsg, 3),
            "categories": {
                cat: {
                    "kg_mean": round(cat_kg_means[cat], 3),
                    "dsg_mean": round(cat_dsg_means[cat], 3),
                    "delta": round(cat_kg_means[cat] - cat_dsg_means[cat], 3),
                }
                for cat in categories
            },
        },
        "queries": [
            {
                "id": r["id"],
                "category": r["category"],
                "question": r["question"],
                "kg_scores": r["kg_scores"],
                "dsg_scores": r["dsg_scores"],
                "kg_mean": round(np.mean(r["kg_scores"]), 2),
                "dsg_mean": round(np.mean(r["dsg_scores"]), 2),
                "kg_answers": r["kg_answers"],
                "dsg_answers": r["dsg_answers"],
                "kg_judgments": r["kg_judgments"],
                "dsg_judgments": r["dsg_judgments"],
                "kg_context_preview": r["kg_context"][:500],
                "dsg_context_preview": r["dsg_context"][:500],
            }
            for r in all_results
        ],
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nTotal LLM calls: {total_calls}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_experiment()
