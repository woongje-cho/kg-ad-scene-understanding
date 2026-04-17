#!/usr/bin/env python3
"""
KSAE 2026 Experiment: KG vs 3DSG Comparison for Autonomous Driving Scene Understanding
30 queries × 2 modes = 60 evaluations
"""

import requests
import json
from dataclasses import dataclass, field
from typing import List, Optional

GRAPHDB_URL = "http://localhost:7200/repositories/DrivingKG"
DRIVING_NS = "http://www.semanticweb.org/driving-ontology/2026/3#"
PREFIX = f"PREFIX : <{DRIVING_NS}>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX owl: <http://www.w3.org/2002/07/owl#>\n"

# Load 3DSG data (temp.json)
with open('ontology_server/data/envs/Driving/temp.json', 'r') as f:
    dsg_data = json.load(f)['output']

DSG_ROOMS = dsg_data['room']
DSG_OBJECTS = dsg_data['object']


# ============================================================
# SPARQL helper
# ============================================================
def sparql_query(query: str, infer: bool = True) -> list:
    """Execute SPARQL SELECT and return list of binding dicts."""
    resp = requests.get(
        GRAPHDB_URL,
        params={'query': PREFIX + query, 'infer': 'true' if infer else 'false'},
        headers={'Accept': 'application/sparql-results+json'}
    )
    resp.raise_for_status()
    results = resp.json()['results']['bindings']
    parsed = []
    for b in results:
        row = {}
        for k, v in b.items():
            val = v['value']
            if val.startswith(DRIVING_NS):
                val = val[len(DRIVING_NS):]
            row[k] = val
        parsed.append(row)
    return parsed


# ============================================================
# 3DSG helpers
# ============================================================
def dsg_find_objects_in_room(room_id: int) -> list:
    """Find objects in a room by room_id (3DSG: spatial lookup only)."""
    return [obj for obj in DSG_OBJECTS.values() if obj.get('parent_room') == room_id]

def dsg_find_by_class(class_name: str) -> list:
    """Find objects by class_name in 3DSG."""
    return [obj for obj in DSG_OBJECTS.values() if obj.get('class_name') == class_name]

def dsg_get_object(obj_id: int) -> Optional[dict]:
    """Get object by numeric ID."""
    return DSG_OBJECTS.get(str(obj_id))

def dsg_get_room(room_id: int) -> Optional[dict]:
    """Get room by numeric ID."""
    return DSG_ROOMS.get(str(room_id))

def dsg_find_nearest_room(x: float, y: float) -> Optional[dict]:
    """Find nearest room to given coordinates."""
    best = None
    best_dist = float('inf')
    for r in DSG_ROOMS.values():
        loc = r['location']
        dist = ((loc[0]-x)**2 + (loc[1]-y)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = r
    return best


# ============================================================
# Query definitions
# ============================================================
@dataclass
class Query:
    id: int
    category: str
    question: str
    ground_truth: str
    required_info: List[str]  # what info is needed to answer correctly
    kg_sparql: str            # SPARQL for KG mode
    dsg_method: str           # description of 3DSG approach
    kg_score: float = 0.0
    dsg_score: float = 0.0
    kg_result: str = ""
    dsg_result: str = ""


QUERIES = [
    # ============================================================
    # Category 1: Spatial / Location (Q1-Q5)
    # ============================================================
    Query(1, "Spatial",
          "Where is the ego vehicle currently located?",
          "urban_road_4 in urban_district, coordinates (-85, 10)",
          ["zone name", "coordinates"],
          "SELECT ?zone ?x ?y WHERE { :ego_car :inZone ?zone . :ego_car :x ?x . :ego_car :y ?y }",
          "lookup object id=1 -> parent_room -> room info"),

    Query(2, "Spatial",
          "Which zone contains the most vehicles?",
          "urban_road_4 (3: ego_car, sedan_3, emergency_vehicle_12) or highway_section_2 (3: truck_5, sedan_10, sedan_11)",
          ["zone name", "vehicle count per zone"],
          """SELECT ?zone (COUNT(?v) as ?cnt) WHERE {
             ?v :inZone ?zone . ?v a :Vehicle .
          } GROUP BY ?zone ORDER BY DESC(?cnt) LIMIT 5""",
          "count objects per room where class_name contains vehicle types"),

    Query(3, "Spatial",
          "How many objects are in the school zone?",
          "5 objects: bus_7, bicycle_9, child_pedestrian_2, child_pedestrian_5, crosswalk_2 + signs",
          ["object count", "object list"],
          "SELECT ?obj ?type WHERE { ?obj :inZone :school_zone_1 . ?obj a ?type . FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?obj a ?sub . FILTER(?sub != ?type) } }",
          "find room with scene_category=school_zone -> count objects with that parent_room"),

    Query(4, "Spatial",
          "What zones are connected to intersection_2?",
          "urban_road_1, urban_road_2, urban_road_3 (via conn_2, conn_3, conn_5)",
          ["connected zone names", "connection details"],
          """SELECT ?conn ?zone WHERE {
             ?conn :isConnectionOf :intersection_2 .
             ?conn :isConnectionOf ?zone .
             FILTER(?zone != :intersection_2)
          }""",
          "no connectivity data in temp.json (rooms have no connection info)"),

    Query(5, "Spatial",
          "List all zones in the highway district.",
          "highway_section_1/2/3, merge_zone_1/2, highway_road_1/2",
          ["zone names in highway district"],
          "SELECT ?zone ?type WHERE { ?zone :isZoneOf :highway_district . ?zone a ?type . FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?zone a ?sub . FILTER(?sub != ?type) } }",
          "filter rooms where floor_number=highway"),

    # ============================================================
    # Category 2: Object Identification (Q6-Q10)
    # ============================================================
    Query(6, "Identification",
          "How many traffic lights are in the scene?",
          "8 traffic lights",
          ["exact count"],
          "SELECT (COUNT(?tl) as ?cnt) WHERE { ?tl a :TrafficLight }",
          "count objects where class_name=traffic_light"),

    Query(7, "Identification",
          "Find all pedestrians and their locations.",
          "5 pedestrians: pedestrian_1 (intersection_2), child_pedestrian_2 (school_zone_1), elderly_pedestrian_3 (intersection_3), pedestrian_4 (residential_road_1), child_pedestrian_5 (school_zone_1)",
          ["pedestrian names", "locations", "pedestrian types"],
          "SELECT ?ped ?type ?zone WHERE { ?ped a :Pedestrian . ?ped :inZone ?zone . ?ped a ?type . FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?ped a ?sub . FILTER(?sub != ?type) } }",
          "filter objects where class_name=pedestrian"),

    Query(8, "Identification",
          "List all parked vehicles.",
          "parked_car_13 (parking_area_1), parked_car_14 (parking_area_2), truck_4 (urban_road_3, illegally parked)",
          ["parked vehicle IDs", "locations", "legal/illegal status"],
          """SELECT ?v ?zone ?comment WHERE {
             { ?v a :ParkedVehicle } UNION { ?v :speed 0 . ?v a :Vehicle }
             ?v :inZone ?zone . ?v rdfs:comment ?comment
          }""",
          "filter objects where class_name=parked_vehicle"),

    Query(9, "Identification",
          "How many obstacles are on the highway?",
          "1: debris_4 on highway_section_2",
          ["count", "obstacle types", "locations"],
          """SELECT ?obs ?type ?zone WHERE {
             ?obs a :Obstacle . ?obs :inZone ?zone .
             ?zone :isZoneOf :highway_district .
             ?obs a ?type .
             FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?obs a ?sub . FILTER(?sub != ?type) }
          }""",
          "filter objects where class_name in (debris, construction_cone) and parent_room in highway rooms"),

    Query(10, "Identification",
          "Find all emergency vehicles and their status.",
          "emergency_vehicle_12: ambulance with sirens active, 90 km/h, approaching from behind on urban_road_4",
          ["vehicle ID", "status", "speed", "behavior description"],
          "SELECT ?v ?zone ?speed ?state ?comment WHERE { ?v a :EmergencyVehicle . ?v :inZone ?zone . ?v :speed ?speed . ?v :state ?state . ?v rdfs:comment ?comment }",
          "filter objects where class_name=emergency_vehicle"),

    # ============================================================
    # Category 3: Semantic Properties (Q11-Q15)
    # ============================================================
    Query(11, "Semantic",
          "What is the speed limit at school_zone_1?",
          "30 km/h",
          ["speed limit value"],
          "SELECT ?limit WHERE { :school_zone_1 :speedLimit ?limit }",
          "no speed limit data in temp.json"),

    Query(12, "Semantic",
          "What state is traffic_light_1 currently in?",
          "yellow — creating a dilemma zone situation",
          ["signal state", "semantic context of what yellow means here"],
          "SELECT ?state ?comment WHERE { :traffic_light_1 :state ?state . :traffic_light_1 rdfs:comment ?comment }",
          "no state data in temp.json"),

    Query(13, "Semantic",
          "Which vehicles are currently stationary?",
          "truck_4 (illegally parked), bus_6 (at bus stop), parked_car_13, parked_car_14",
          ["vehicle IDs", "reason for being stationary"],
          """SELECT ?v ?type ?zone ?comment WHERE {
             ?v a :Vehicle . ?v :speed 0 .
             ?v :inZone ?zone . ?v rdfs:comment ?comment .
             ?v a ?type .
             FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?v a ?sub . FILTER(?sub != ?type) }
          }""",
          "no speed data in temp.json"),

    Query(14, "Semantic",
          "What is sedan_3's speed and why does it matter?",
          "58 km/h, following ego at 15m — critical for dilemma zone (rear-end collision risk if ego brakes)",
          ["speed value", "following distance", "safety implication"],
          "SELECT ?speed ?comment WHERE { :sedan_3 :speed ?speed . :sedan_3 rdfs:comment ?comment }",
          "no speed data in temp.json, only coordinates"),

    Query(15, "Semantic",
          "Is the connection between intersection_3 and side_road_1 open?",
          "No, conn_8 isOpen=false (closed)",
          ["open/closed status"],
          "SELECT ?conn ?open WHERE { ?conn :isConnectionOf :intersection_3 . ?conn :isConnectionOf :side_road_1 . ?conn :isOpen ?open }",
          "no connection or open/closed data in temp.json"),

    # ============================================================
    # Category 4: Class Hierarchy Reasoning (Q16-Q20)
    # ============================================================
    Query(16, "Hierarchy",
          "Find all TrafficControl objects (lights + signs).",
          "18 objects: 8 TrafficLights + 10 TrafficSigns",
          ["all traffic control objects with subtypes"],
          "SELECT ?obj ?type WHERE { ?obj a :TrafficControl . ?obj a ?type . FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?obj a ?sub . FILTER(?sub != ?type) } } ORDER BY ?type",
          "filter by class_name in (traffic_light, speed_limit_sign, stop_sign, yield_sign, ...) — must enumerate all subtypes manually"),

    Query(17, "Hierarchy",
          "What types of vehicles are present in the scene?",
          "Car(6), Truck(2), Bus(2), Motorcycle(1), Bicycle(1), EmergencyVehicle(1), ParkedVehicle(2), EgoVehicle(1)",
          ["vehicle subtype counts"],
          "SELECT ?type (COUNT(?v) as ?cnt) WHERE { ?v a :Vehicle . ?v a ?type . FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?v a ?sub . FILTER(?sub != ?type) } } GROUP BY ?type ORDER BY DESC(?cnt)",
          "count by class_name for vehicle-related names"),

    Query(18, "Hierarchy",
          "What subtypes of Zone are present?",
          "Intersection(4), LaneSegment(13), HighwaySection(3), ParkingArea(2), SchoolZone(1), MergeZone(2)",
          ["zone subtype names and counts"],
          "SELECT ?type (COUNT(?z) as ?cnt) WHERE { ?z a :Zone . ?z a ?type . FILTER(?type != :Zone && ?type != :Environment && ?type != :TopologicalObject) FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?z a ?sub . FILTER(?sub != ?type) } } GROUP BY ?type",
          "count rooms by scene_category"),

    Query(19, "Hierarchy",
          "Find all Infrastructure objects.",
          "7: guardrail_1, guardrail_2, crosswalk_1, crosswalk_2, bus_stop_1, bus_stop_2, median_1",
          ["infrastructure objects with subtypes"],
          "SELECT ?obj ?type ?zone WHERE { ?obj a :Infrastructure . ?obj :inZone ?zone . ?obj a ?type . FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?obj a ?sub . FILTER(?sub != ?type) } }",
          "filter by class_name in (guardrail, crosswalk, bus_stop, median) — must know all subtypes"),

    Query(20, "Hierarchy",
          "Is a Crosswalk classified as Infrastructure? What is its full class hierarchy?",
          "Yes: Crosswalk → Infrastructure → Object → PhysicalObject",
          ["class hierarchy chain"],
          "SELECT ?parent WHERE { :Crosswalk rdfs:subClassOf+ ?parent } ORDER BY ?parent",
          "no class hierarchy in temp.json"),

    # ============================================================
    # Category 5: Semantic/Common-Sense Reasoning (Q21-Q25)
    # ============================================================
    Query(21, "Reasoning",
          "Why is truck_4 a hazard and how should vehicles respond?",
          "Illegally parked in right lane of urban_road_3, blocks visibility. Vehicles must cross center line to pass when oncoming lane is clear.",
          ["parking status", "visibility impact", "recommended action"],
          "SELECT ?comment ?zone ?obstacle WHERE { :truck_4 rdfs:comment ?comment . :truck_4 :inZone ?zone . :truck_4 :isObstacle ?obstacle }",
          "only know: truck at (-18, 9) in room 6 — no description of why it's there or how to respond"),

    Query(22, "Reasoning",
          "What makes the school zone dangerous right now?",
          "Children crossing (child_pedestrian_2 at crosswalk), child group on sidewalk edge (child_pedestrian_5), school bus approaching (bus_7), cyclist (bicycle_9). Children are unpredictable.",
          ["active pedestrians", "child behavior patterns", "bus status", "combined risk assessment"],
          """SELECT ?obj ?type ?comment WHERE {
             ?obj :inZone :school_zone_1 . ?obj rdfs:comment ?comment .
             ?obj a ?type .
             FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?obj a ?sub . FILTER(?sub != ?type) }
          }""",
          "know objects in room 9 by class_name and coordinates — but no behavior descriptions or risk context"),

    Query(23, "Reasoning",
          "Should the ego vehicle brake hard at intersection_1 given the yellow light?",
          "No — sedan_3 following at 15m/58km/h makes emergency braking dangerous (rear-end collision risk). Proceeding through is safer.",
          ["yellow light state", "following vehicle distance/speed", "braking deceleration required", "rear-end risk"],
          """SELECT ?obj ?comment WHERE {
             { :traffic_light_1 rdfs:comment ?comment . BIND(:traffic_light_1 as ?obj) }
             UNION
             { :sedan_3 rdfs:comment ?comment . BIND(:sedan_3 as ?obj) }
             UNION
             { :ego_car rdfs:comment ?comment . BIND(:ego_car as ?obj) }
          }""",
          "know traffic_light at (-80,12) in room 1, car at (-88,9.8) in room 7 — no signal state, no speeds, no reasoning context"),

    Query(24, "Reasoning",
          "What pedestrian risk exists near bus_6 at intersection_2?",
          "Bus obscures crosswalk_1 behind it. pedestrian_1 waiting impatiently may jaywalk. Hidden crosswalk is one of most dangerous pedestrian scenarios.",
          ["bus position", "crosswalk visibility", "pedestrian behavior", "hidden hazard"],
          """SELECT ?obj ?comment WHERE {
             { ?obj :inZone :intersection_2 . ?obj rdfs:comment ?comment }
             UNION
             { ?obj :inZone :bus_lane_1 . ?obj rdfs:comment ?comment }
          }""",
          "know bus at (-42,14) in room 25, pedestrian at (-41,8) in room 2 — no visibility or behavior info"),

    Query(25, "Reasoning",
          "What's the danger of debris_3 and how does road condition affect it?",
          "Dark debris on wet road after rain, blends with surface. Risk of tire damage or loss of control on slippery surface. Hard to see at night.",
          ["debris description", "road condition", "visibility", "vehicle control risk"],
          "SELECT ?comment WHERE { :debris_3 rdfs:comment ?comment }",
          "know: debris at (72,76) in room 23 — no material, road condition, or visibility info"),

    # ============================================================
    # Category 6: Dilemma Scenario Reasoning (Q26-Q30)
    # ============================================================
    Query(26, "Dilemma",
          "Describe all factors in the yellow light dilemma at intersection_1.",
          "Ego at 55km/h ~30m from intersection, yellow light, sedan_3 following at 15m/58km/h. Hard braking needs >4m/s² deceleration. Rear-end collision risk vs running late yellow.",
          ["ego speed/distance", "light state", "following vehicle", "deceleration physics", "risk trade-off"],
          """SELECT ?obj ?type ?comment WHERE {
             VALUES ?obj { :ego_car :traffic_light_1 :sedan_3 :intersection_1 }
             ?obj rdfs:comment ?comment .
             ?obj a ?type .
             FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?obj a ?sub . FILTER(?sub != ?type) }
          }""",
          "know positions of 3 objects — no speeds, no light state, no following distance, no physics"),

    Query(27, "Dilemma",
          "Should the ego vehicle exceed the 90 km/h speed limit on the highway to match 95 km/h traffic flow?",
          "Yes — matching flow is safer. Strict compliance creates dangerous speed differential. sedan_10 (95), sedan_11 (93) demonstrate consistent flow above limit.",
          ["speed limit", "actual traffic flow speeds", "speed differential risk", "common sense vs legal"],
          """SELECT ?obj ?comment WHERE {
             VALUES ?obj { :speed_limit_sign_3 :sedan_10 :sedan_11 :truck_5 }
             ?obj rdfs:comment ?comment
          }""",
          "know objects in highway rooms by position — no speed data, no limit info, no flow context"),

    Query(28, "Dilemma",
          "How to safely pass the illegally parked truck_4 on urban_road_3?",
          "Cross center line when oncoming lane clear. Truck blocks visibility of oncoming traffic (tall vehicle). No-parking sign confirms illegal status. Standard safe response despite technical lane violation.",
          ["truck position/size", "visibility impact", "oncoming traffic check", "legal vs practical action"],
          """SELECT ?obj ?comment WHERE {
             { :truck_4 rdfs:comment ?comment . BIND(:truck_4 as ?obj) }
             UNION
             { :no_parking_sign_6 rdfs:comment ?comment . BIND(:no_parking_sign_6 as ?obj) }
             UNION
             { ?obj :inZone :urban_road_3 . ?obj rdfs:comment ?comment }
          }""",
          "know truck at (-18,9) in room 6 — no size, no visibility, no legal context"),

    Query(29, "Dilemma",
          "Assess the multi-factor risk at intersection_2 (bus, pedestrian, motorcycle, crosswalk).",
          "Bus_6 blocks crosswalk visibility, pedestrian_1 may jaywalk, motorcycle_8 lane-splitting (hard to detect), crosswalk_1 has LED warnings. Compound risk requires slow approach.",
          ["all objects at intersection", "each object's risk contribution", "compound risk assessment"],
          """SELECT ?obj ?type ?comment WHERE {
             { ?obj :inZone :intersection_2 . ?obj rdfs:comment ?comment . ?obj a ?type }
             UNION
             { ?obj :inZone :bus_lane_1 . ?obj rdfs:comment ?comment . ?obj a ?type }
             FILTER NOT EXISTS { ?sub rdfs:subClassOf ?type . ?obj a ?sub . FILTER(?sub != ?type) }
          }""",
          "know objects in rooms 2 and 25 by class and position — no behavior, risk, or visibility info"),

    Query(30, "Dilemma",
          "What makes the merge at merge_zone_1 dangerous and how should vehicles cooperate?",
          "sedan_15 entering at 55km/h vs highway flow at 90+ km/h — 35km/h speed differential. Ramp metering signal (red). yield_sign_5 present. Requires cooperative lane change from highway traffic.",
          ["merging vehicle speed", "highway flow speed", "speed differential", "traffic control", "cooperative behavior needed"],
          """SELECT ?obj ?comment WHERE {
             { ?obj :inZone :merge_zone_1 . ?obj rdfs:comment ?comment }
             UNION
             { :yield_sign_5 rdfs:comment ?comment . BIND(:yield_sign_5 as ?obj) }
          }""",
          "know objects in room 18 — no speeds, no signal state, no cooperation context"),
]


# ============================================================
# Run experiment
# ============================================================
def run_kg_query(q: Query) -> str:
    """Run KG mode: SPARQL query against GraphDB."""
    try:
        results = sparql_query(q.kg_sparql)
        if not results:
            return "[NO RESULTS]"

        lines = []
        for row in results:
            parts = []
            for k, v in row.items():
                if k == 'comment':
                    parts.append(f"comment=\"{v[:200]}\"")
                else:
                    parts.append(f"{k}={v}")
            lines.append("; ".join(parts))
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR: {e}]"


def run_dsg_query(q: Query) -> str:
    """Run 3DSG mode: JSON lookup."""
    try:
        # Map zone names to room IDs
        zone_room_map = {
            'intersection_1': 1, 'intersection_2': 2, 'intersection_3': 3,
            'urban_road_1': 4, 'urban_road_2': 5, 'urban_road_3': 6,
            'urban_road_4': 7, 'parking_area_1': 8, 'school_zone_1': 9,
            'residential_road_1': 10, 'residential_road_2': 11, 'residential_road_3': 12,
            'intersection_4': 13, 'parking_area_2': 14, 'highway_section_1': 15,
            'highway_section_2': 16, 'highway_section_3': 17, 'merge_zone_1': 18,
            'merge_zone_2': 19, 'highway_road_1': 20, 'highway_road_2': 21,
            'side_road_1': 22, 'wet_road_section_1': 23, 'construction_zone_1': 24,
            'bus_lane_1': 25
        }

        # General approach: extract what 3DSG CAN provide
        info_parts = []

        # For each query, find relevant objects from JSON
        # Simple heuristic: look for keywords in question to determine what to look up
        question_lower = q.question.lower()

        # Try to find objects by class or location
        if 'ego' in question_lower:
            obj = dsg_get_object(1)
            if obj:
                room = dsg_get_room(obj['parent_room'])
                info_parts.append(f"ego: class={obj['class_name']}, loc={obj['location']}, room={room['scene_category'] if room else '?'}")

        if 'school zone' in question_lower or 'school_zone' in question_lower:
            objs = dsg_find_objects_in_room(9)
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'intersection_1' in question_lower:
            objs = dsg_find_objects_in_room(1)
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'intersection_2' in question_lower:
            objs = dsg_find_objects_in_room(2)
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")
            objs25 = dsg_find_objects_in_room(25)
            for obj in objs25:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'merge_zone_1' in question_lower or 'merge' in question_lower:
            objs = dsg_find_objects_in_room(18)
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'highway' in question_lower:
            for rid in [15, 16, 17, 18, 19, 20, 21]:
                objs = dsg_find_objects_in_room(rid)
                for obj in objs:
                    info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}, room={rid}")

        if 'traffic light' in question_lower:
            objs = dsg_find_by_class('traffic_light')
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'pedestrian' in question_lower:
            objs = dsg_find_by_class('pedestrian')
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'parked' in question_lower:
            objs = dsg_find_by_class('parked_vehicle')
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'truck_4' in question_lower or 'truck' in question_lower:
            objs = dsg_find_by_class('truck')
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'emergency' in question_lower:
            objs = dsg_find_by_class('emergency_vehicle')
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'debris' in question_lower:
            objs = dsg_find_by_class('debris')
            for obj in objs:
                info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'obstacle' in question_lower:
            for cls in ['debris', 'construction_cone']:
                objs = dsg_find_by_class(cls)
                for obj in objs:
                    info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'infrastructure' in question_lower or 'crosswalk' in question_lower:
            for cls in ['guardrail', 'crosswalk', 'bus_stop', 'median']:
                objs = dsg_find_by_class(cls)
                for obj in objs:
                    info_parts.append(f"obj_{obj['id']}: class={obj['class_name']}, loc={obj['location']}")

        if 'vehicle' in question_lower and 'type' in question_lower:
            all_vehicle_classes = set()
            for obj in DSG_OBJECTS.values():
                cn = obj['class_name']
                if cn in ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'emergency_vehicle', 'parked_vehicle', 'ego_vehicle'):
                    all_vehicle_classes.add(cn)
            info_parts.append(f"vehicle classes found: {sorted(all_vehicle_classes)}")

        if 'zone' in question_lower and ('subtype' in question_lower or 'type' in question_lower):
            cats = set()
            for r in DSG_ROOMS.values():
                cats.add(r['scene_category'])
            info_parts.append(f"room categories: {sorted(cats)}")

        if 'connected' in question_lower or 'connection' in question_lower:
            info_parts.append("[NO CONNECTIVITY DATA IN 3DSG]")

        if 'speed' in question_lower and 'limit' in question_lower:
            info_parts.append("[NO SPEED LIMIT DATA IN 3DSG]")

        if 'state' in question_lower or 'yellow' in question_lower or 'signal' in question_lower:
            info_parts.append("[NO SIGNAL STATE DATA IN 3DSG]")

        if 'hierarchy' in question_lower or 'classified' in question_lower:
            info_parts.append("[NO CLASS HIERARCHY IN 3DSG]")

        if not info_parts:
            return "[NO RELEVANT DATA FOUND]"

        return "\n".join(info_parts[:15])  # limit output

    except Exception as e:
        return f"[ERROR: {e}]"


def evaluate_score(q: Query, result: str, required_info: list) -> float:
    """Score how many required info items are present in the result."""
    if "[NO RESULTS]" in result or "[ERROR" in result:
        return 0.0

    result_lower = result.lower()
    found = 0

    for info in required_info:
        info_lower = info.lower()

        # Check if the type of info is present
        if 'count' in info_lower and any(c.isdigit() for c in result):
            found += 1
        elif 'name' in info_lower and len(result) > 20:
            found += 1
        elif 'coordinate' in info_lower and ('loc=' in result_lower or 'x=' in result_lower):
            found += 1
        elif 'speed' in info_lower and ('speed' in result_lower or 'km/h' in result_lower):
            found += 1
        elif 'comment' in info_lower or 'description' in info_lower or 'context' in info_lower:
            if 'comment=' in result_lower or len(result) > 100:
                found += 1
        elif 'state' in info_lower and ('state=' in result_lower or 'yellow' in result_lower or 'green' in result_lower or 'red' in result_lower):
            found += 1
        elif 'hierarchy' in info_lower and ('subclassof' in result_lower or 'parent' in result_lower or '->' in result_lower):
            found += 1
        elif 'connect' in info_lower and ('conn' in result_lower or 'connection' in result_lower):
            found += 1
        elif 'risk' in info_lower or 'danger' in info_lower or 'hazard' in info_lower:
            if 'comment=' in result_lower and len(result) > 100:
                found += 1
        elif 'behavior' in info_lower or 'pattern' in info_lower:
            if 'comment=' in result_lower and len(result) > 100:
                found += 1
        elif 'action' in info_lower or 'response' in info_lower or 'recommend' in info_lower:
            if 'comment=' in result_lower and len(result) > 100:
                found += 1
        elif 'visibility' in info_lower or 'impact' in info_lower:
            if 'comment=' in result_lower and len(result) > 80:
                found += 1
        elif 'trade-off' in info_lower or 'dilemma' in info_lower or 'physics' in info_lower:
            if 'comment=' in result_lower and 'dilemma' in result_lower:
                found += 1
        elif 'legal' in info_lower or 'practical' in info_lower:
            if 'comment=' in result_lower and ('illegal' in result_lower or 'legal' in result_lower or 'law' in result_lower):
                found += 1
        elif 'open' in info_lower or 'closed' in info_lower:
            if 'open=' in result_lower or 'true' in result_lower or 'false' in result_lower:
                found += 1
        elif 'cooperative' in info_lower:
            if 'comment=' in result_lower and 'cooperat' in result_lower:
                found += 1
        elif any(keyword in result_lower for keyword in info_lower.split()):
            found += 1

    return round(found / len(required_info), 2) if required_info else 0.0


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("KSAE 2026 Experiment: KG vs 3DSG for AD Scene Understanding")
    print("=" * 80)

    for q in QUERIES:
        print(f"\n--- Q{q.id} [{q.category}] {q.question}")

        # KG mode
        q.kg_result = run_kg_query(q)
        q.kg_score = evaluate_score(q, q.kg_result, q.required_info)

        # 3DSG mode
        q.dsg_result = run_dsg_query(q)
        q.dsg_score = evaluate_score(q, q.dsg_result, q.required_info)

        print(f"  KG:   score={q.kg_score:.2f}  |  3DSG: score={q.dsg_score:.2f}")

    # ============================================================
    # Results summary
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    categories = ['Spatial', 'Identification', 'Semantic', 'Hierarchy', 'Reasoning', 'Dilemma']

    print(f"\n{'Category':<18} {'KG Avg':>8} {'3DSG Avg':>10} {'Delta':>8} {'KG Wins':>9}")
    print("-" * 58)

    total_kg = 0
    total_dsg = 0
    total_queries = 0
    kg_wins = 0

    for cat in categories:
        cat_queries = [q for q in QUERIES if q.category == cat]
        kg_avg = sum(q.kg_score for q in cat_queries) / len(cat_queries)
        dsg_avg = sum(q.dsg_score for q in cat_queries) / len(cat_queries)
        delta = kg_avg - dsg_avg
        wins = sum(1 for q in cat_queries if q.kg_score > q.dsg_score)

        print(f"{cat:<18} {kg_avg:>7.1%} {dsg_avg:>9.1%} {delta:>+7.1%} {wins:>5}/{len(cat_queries)}")

        total_kg += sum(q.kg_score for q in cat_queries)
        total_dsg += sum(q.dsg_score for q in cat_queries)
        total_queries += len(cat_queries)
        kg_wins += wins

    print("-" * 58)
    overall_kg = total_kg / total_queries
    overall_dsg = total_dsg / total_queries
    print(f"{'OVERALL':<18} {overall_kg:>7.1%} {overall_dsg:>9.1%} {overall_kg - overall_dsg:>+7.1%} {kg_wins:>5}/{total_queries}")

    # Detailed per-query table
    print(f"\n{'Q#':<4} {'Category':<15} {'KG':>6} {'3DSG':>6} {'Winner':>8}  Question")
    print("-" * 90)
    for q in QUERIES:
        winner = "KG" if q.kg_score > q.dsg_score else ("TIE" if q.kg_score == q.dsg_score else "3DSG")
        print(f"Q{q.id:<3} {q.category:<15} {q.kg_score:>5.0%} {q.dsg_score:>5.0%} {winner:>8}  {q.question[:50]}")

    # Save detailed results to JSON
    results = {
        "experiment": "KSAE 2026 KG vs 3DSG",
        "total_queries": total_queries,
        "overall_kg_accuracy": round(overall_kg * 100, 1),
        "overall_dsg_accuracy": round(overall_dsg * 100, 1),
        "kg_wins": kg_wins,
        "dsg_wins": sum(1 for q in QUERIES if q.dsg_score > q.kg_score),
        "ties": sum(1 for q in QUERIES if q.kg_score == q.dsg_score),
        "by_category": {},
        "queries": []
    }

    for cat in categories:
        cat_queries = [q for q in QUERIES if q.category == cat]
        results["by_category"][cat] = {
            "kg_avg": round(sum(q.kg_score for q in cat_queries) / len(cat_queries) * 100, 1),
            "dsg_avg": round(sum(q.dsg_score for q in cat_queries) / len(cat_queries) * 100, 1),
        }

    for q in QUERIES:
        results["queries"].append({
            "id": q.id,
            "category": q.category,
            "question": q.question,
            "ground_truth": q.ground_truth,
            "kg_score": q.kg_score,
            "dsg_score": q.dsg_score,
            "kg_result_preview": q.kg_result[:300],
            "dsg_result_preview": q.dsg_result[:300],
        })

    with open('experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to experiment_results.json")
