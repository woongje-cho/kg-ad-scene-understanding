#!/usr/bin/env python3
"""
FastAPI Server for Ontology Management
Real-time REST API for ontology operations
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from .ontology import OntologyManager
from .env import EnvManager
from .embedding import EmbeddingManager
from .models import IndividualData, IndividualUpdate, StatusResponse, OperationResponse, BatchIndividualsData, AffordanceQuery, SceneQuery, SemanticQuery, ClassQuery, AffordanceQuery, RoomQuery
from typing import List, Dict, Any, Optional, cast
import os
import json # <-- 추가
from pathlib import Path # <-- 추가
from .config import ConfigLoader

# Global manager instances
manager: OntologyManager = None
env_manager: EnvManager = None
current_env_id: Optional[str] = None

config_loader = ConfigLoader()

# 3DSG JSON cache (loaded per active environment)
SCENE_3DSG_DATA = {}
try:
    data_cfg = config_loader.get_data_config()
    envs_dir = data_cfg.get("envs_dir", "data/envs")
    env_id_for_3dsg = os.getenv("ONTOLOGY_ENV_ID") or config_loader.get_active_env() or "Darden"
    json_path = Path(envs_dir) / env_id_for_3dsg / "temp.json"

    with open(json_path, "r", encoding="utf-8") as f:
        SCENE_3DSG_DATA = json.load(f)
    print(f"✓ Successfully loaded 3DSG data from {json_path}")

except Exception as e:
    print(f"⚠️ WARNING: Failed to load 3DSG data (temp.json). 3DSG tools will fail. Error: {e}")


def get_lifespan(env_id: Optional[str] = None):
    """Create lifespan context manager with space parameter."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup/shutdown."""
        global manager, env_manager, current_env_id

        # Startup
        print("🚀 Starting Ontology Manager Server...")

        # Initialize space manager
        env_manager = EnvManager()
        current_env_id = env_id

        # Get ontology path
        ontology_path = str(env_manager.get_ontology_path())

        # Initialize ontology manager with space
        manager = OntologyManager(owl_path=ontology_path, env_id=env_id)

        # Ensure connectedTo graph (doors/openings/stairs) is built at startup
        try:
            _rebuild_connections_from_doors()
        except Exception as exc:
            print(f"[WARN] Failed to rebuild connectedTo on startup: {exc}")

        if env_id:
            space_config = env_manager.get_env_config(env_id)
            if space_config:
                env_label = space_config.get("env_name") or space_config.get("name") or env_id
                print(f"📍 Active environment: {env_label} ({env_id})")

        print("✅ Server ready!\n")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🔗 Neo4j Browser: http://localhost:7474\n")

        yield

        # Shutdown
        print("\n🛑 Shutting down Ontology Manager Server...")
        if manager:
            manager.close()
        print("✅ Server stopped")

    return lifespan


# Get space from environment variable (for server startup)
ENV_ID = os.getenv("ONTOLOGY_ENV_ID", None)

# Create FastAPI app
app = FastAPI(
    title="Ontology Manager API",
    description="Real-time ontology management with owlready2 + HermiT reasoner + Neo4j (space-aware)",
    version="2.0.0",
    lifespan=get_lifespan(ENV_ID)
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Ontology Manager API",
        "docs": "/docs",
        "status": "/status"
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current ontology status."""
    if not manager:
        raise HTTPException(status_code=503, detail="Manager not initialized")

    status = manager.get_status()
    # Add space information
    if current_env_id and env_manager:
        space_config = env_manager.get_env_config(current_env_id)
        if space_config:
            status["env_id"] = current_env_id
            status["env_name"] = space_config.get("env_name") or space_config.get("name")
    return status


@app.get("/spaces")
async def list_spaces():
    """List all available spaces."""
    if not env_manager:
        raise HTTPException(status_code=503, detail="Space manager not initialized")

    return {"spaces": env_manager.list_spaces()}


@app.get("/spaces/summary")
async def get_spaces_summary():
    """Get summary of all spaces."""
    if not env_manager:
        raise HTTPException(status_code=503, detail="Space manager not initialized")

    return env_manager.get_summary()


@app.get("/spaces/{env_id}")
async def get_space_info(env_id: str):
    """Get information about a specific space."""
    if not env_manager:
        raise HTTPException(status_code=503, detail="Space manager not initialized")

    config = env_manager.get_env_config(env_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Space '{env_id}' not found")

    # Get file paths
    static_path = env_manager.get_static_file_path(env_id)
    dynamic_path = env_manager.get_dynamic_file_path(env_id)

    return {
        "config": config,
        "static_file": str(static_path) if static_path else None,
        "dynamic_file": str(dynamic_path) if dynamic_path else None,
        "is_active": env_id == current_env_id
    }


@app.post("/individuals", response_model=OperationResponse)
async def add_individual(data: IndividualData):
    """
    Add a new individual to the ontology.

    Automatically runs reasoner and syncs to Neo4j.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Manager not initialized")

    # Convert Pydantic model to dict
    individual_dict = {
        "id": data.id,
        "class": data.class_name,
        "data_properties": data.data_properties or {},
        "object_properties": data.object_properties or {}
    }

    result = manager.add_individual(individual_dict)

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return result


@app.post("/individuals/batch", response_model=OperationResponse)
async def add_individuals_batch(data: BatchIndividualsData):
    """
    Add multiple individuals at once (batch operation).

    Runs reasoner only once after all individuals are added.
    Much faster than adding individuals one by one.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Manager not initialized")

    # Convert Pydantic models to dicts
    individuals_dicts = []
    for individual in data.individuals:
        individual_dict = {
            "id": individual.id,
            "class": individual.class_name,
            "data_properties": individual.data_properties or {},
            "object_properties": individual.object_properties or {}
        }
        individuals_dicts.append(individual_dict)

    result = manager.add_individuals_batch(individuals_dicts)

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return result


@app.put("/individuals/{individual_id}", response_model=OperationResponse)
async def update_individual(individual_id: str, data: IndividualUpdate):
    """
    Update an existing individual.

    Automatically runs reasoner and syncs to Neo4j.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Manager not initialized")

    # Convert Pydantic model to dict
    update_dict = {}
    if data.data_properties is not None:
        update_dict["data_properties"] = data.data_properties
    if data.object_properties is not None:
        update_dict["object_properties"] = data.object_properties

    result = manager.update_individual(individual_id, update_dict)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])

    return result


@app.delete("/individuals/{individual_id}", response_model=OperationResponse)
async def delete_individual(individual_id: str):
    """
    Delete an individual from the ontology.

    Automatically runs reasoner and syncs to Neo4j.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Manager not initialized")

    result = manager.delete_individual(individual_id)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])

    return result


@app.post("/load_ttl", response_model=OperationResponse)
async def load_ttl(file_path: dict):
    """
    Load individuals from a TTL file.

    Request body:
    {
        "file_path": "data/envs/engineering_building/static.ttl"
    }

    Automatically runs reasoner and syncs to Neo4j after loading.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Manager not initialized")

    ttl_path = file_path.get("file_path")
    if not ttl_path:
        raise HTTPException(status_code=400, detail="file_path is required")

    result = manager.load_instances_from_ttl(ttl_path)

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    # After load, rebuild connectedTo relationships with status from isOpen (Door/Opening)
    try:
        _rebuild_connections_from_doors()
    except Exception as e:
        print(f"[WARN] Failed to rebuild connectedTo from doors: {e}")

    return result


@app.post("/sync", response_model=OperationResponse)
async def sync_ontology():
    """
    Manually trigger reasoner and Neo4j sync.

    Note: Sync is automatically triggered after add/update/delete operations.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Manager not initialized")

    result = manager.sync_to_neo4j()

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return result


@app.post("/sparql")
async def execute_sparql(query: dict):
    """
    Execute SPARQL query on the loaded ontology.

    Request body:
    {
        "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
    }
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Manager not initialized")

    sparql_query = query.get("query")
    if not sparql_query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # Execute SPARQL query using owlready2
        results = list(manager.world.sparql(sparql_query))

        # Convert results to JSON-serializable format
        json_results = []
        for row in results:
            json_row = []
            for item in row:
                if hasattr(item, 'name'):
                    json_row.append({"type": "individual", "value": item.name})
                elif hasattr(item, 'iri'):
                    json_row.append({"type": "iri", "value": str(item.iri)})
                else:
                    json_row.append({"type": "literal", "value": str(item)})
            json_results.append(json_row)

        return {
            "status": "success",
            "count": len(json_results),
            "results": json_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@app.post("/semantic_search")
async def semantic_search(query: str, top_k: int = 5):
    """
    Semantic search using natural language query.

    Args:
        query: Natural language query string
        top_k: Number of top results to return (default: 5)

    Returns:
        List of similar individuals with their descriptions and similarity scores
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Ontology manager not initialized")

    try:
        # Generate embedding for the query
        from .config import get_config
        config = get_config()
        embedding_config = config.get_embedding_config()

        embedding_manager = EmbeddingManager(
            model=embedding_config.get('model', 'text-embedding-3-small'),
            dimensions=embedding_config.get('dimensions', 512)
        )
        query_embedding = embedding_manager.generate_embedding(query)

        # Perform vector similarity search in Neo4j
        with manager.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'individualEmbeddingIndex',
                    $top_k,
                    $query_embedding
                )
                YIELD node, score
                RETURN node.id AS id,
                       labels(node) AS types,
                       node.description AS description,
                       score
                ORDER BY score DESC
            """, top_k=top_k, query_embedding=query_embedding)

            results = []
            for record in result:
                # Filter out 'Individual' from types list
                types = [t for t in record["types"] if t != "Individual"]
                results.append({
                    "id": record["id"],
                    "types": types,
                    "description": record["description"],
                    "score": record["score"]
                })

            return {
                "status": "success",
                "query": query,
                "count": len(results),
                "results": results
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


def _rebuild_connections_from_doors():
    """Create connectedTo edges with status=open/closed based on Door/Opening isOpen property."""
    if not manager or not getattr(manager, "driver", None):
        return
    queries = [
        """
        // Doors: connect rooms via isDoorOf
        MATCH (d:Door)-[:isDoorOf]->(a)
        MATCH (d)-[:isDoorOf]->(b)
        WHERE id(a) < id(b)
        WITH DISTINCT d, a, b, coalesce(d.isOpen, true) AS open
        MERGE (a)-[r:connectedTo]->(b)
        MERGE (b)-[r2:connectedTo]->(a)
        SET r.status  = CASE WHEN open THEN 'open' ELSE 'closed' END,
            r2.status = r.status;
        """,
        """
        // Openings: connect rooms via isOpeningOf, always open
        MATCH (o:Opening)-[:isOpeningOf]->(x)
        MATCH (o)-[:isOpeningOf]->(y)
        WHERE id(x) < id(y)
        WITH DISTINCT o, x, y
        MERGE (x)-[r3:connectedTo]->(y)
        MERGE (y)-[r4:connectedTo]->(x)
        SET r3.status = 'open',
            r4.status = 'open';
        """,
        """
        // Stairs: connect rooms via isStairsOf, always open
        MATCH (s:Stairs)-[:isStairsOf]->(m)
        MATCH (s)-[:isStairsOf]->(n)
        WHERE id(m) < id(n)
        WITH DISTINCT m, n
        MERGE (m)-[r5:connectedTo]->(n)
        MERGE (n)-[r6:connectedTo]->(m)
        SET r5.status = 'open',
            r6.status = 'open';
        """
    ]
    with manager.driver.session() as session:
        for q in queries:
            session.run(q)
    
@app.get("/kg/search_by_name")
async def kg_search_by_name(name: str):
    """
    이름(혹은 id)로 KG(Neo4j)에서 개체를 찾는 엔드포인트.

    Args:
        name: 개체 id 또는 이름 (예: "bottle_2", "lab_201")

    Returns:
        {
          "status": "success",
          "query": "bottle_2",
          "count": 1,
          "results": [
            {
              "id": "bottle_2",
              "types": ["Bottle", "Object"],
              "description": "...",
              "room": "lab_201"
            }
          ]
        }
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Ontology manager not initialized")

    try:
        with manager.driver.session() as session:
            result = session.run(
                """
                MATCH (n {id: $name})
                RETURN n.id AS id,
                       labels(n) AS types,
                       n.description AS description,
                       n.room_id AS room
                """,
                name=name,
            )

            rows: List[Dict[str, Any]] = list(result)

        if not rows:
            # 없을 땐 그냥 빈 리스트 반환
            return {
                "status": "success",
                "query": name,
                "count": 0,
                "results": [],
            }

        results: List[Dict[str, Any]] = []
        for r in rows:
            # 'Individual' 같은 메타 라벨은 제거
            types = [t for t in r["types"] if t != "Individual"]
            results.append(
                {
                    "id": r["id"],
                    "types": types,
                    "description": r.get("description"),
                    "room": r.get("room"),
                }
            )

        return {
            "status": "success",
            "query": name,
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search_by_name failed: {e}")
    


@app.get("/kg/list_in_category")
async def kg_list_in_category(
    category: str = Query(..., description="KG 카테고리/클래스 이름 (예: 'Chair', 'Bottle')"),
    room_id: Optional[str] = Query(None, description="특정 room_id로 제한 (예: 'kitchen_20')")
):
    """
    [KG] 특정 카테고리에 속한 개체들을 나열하는 엔드포인트.

    - category는 Neo4j 노드의 label 중 하나여야 한다. (예: 'Chair', 'Bottle')
    - room_id가 주어지면 해당 방에 있는 개체만 필터링한다.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Ontology manager not initialized")

    try:
        with manager.driver.session() as session:
            if room_id:
                result = session.run(
                    """
                    MATCH (n:Individual {room_id: $room_id})
                    WHERE $category IN labels(n)
                    RETURN n.id AS id,
                           labels(n) AS types,
                           n.description AS description,
                           n.room_id AS room
                    """,
                    room_id=room_id,
                    category=category,
                )
            else:
                result = session.run(
                    """
                    MATCH (n:Individual)
                    WHERE $category IN labels(n)
                    RETURN n.id AS id,
                           labels(n) AS types,
                           n.description AS description,
                           n.room_id AS room
                    """,
                    category=category,
                )

            rows: List[Dict[str, Any]] = list(result)

        objects: List[Dict[str, Any]] = []
        for r in rows:
            types = [t for t in r["types"] if t != "Individual"]
            objects.append(
                {
                    "id": r["id"],
                    "types": types,
                    "description": r.get("description"),
                    "room": r.get("room"),
                }
            )

        return {
            "status": "success",
            "category": category,
            "room_id": room_id,
            "count": len(objects),
            "results": objects,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list_in_category failed: {e}")
    




@app.post("/hybrid/locate_semantic")
async def hybrid_locate_semantic(payload: Dict[str, Any] = Body(...)):
    """
    [Hybrid] 특정 방 + 시맨틱 질의로 개체를 찾는 엔드포인트.

    요청 예:
        {
          "query": "마실 것",
          "room_id": "kitchen_20",
          "top_k": 5
        }
    """
    query = payload.get("query")
    room_id = payload.get("room_id")
    top_k = payload.get("top_k", 5)

    if not query:
        raise HTTPException(status_code=400, detail="'query' 필드는 필수입니다.")

    # 간단 버전: room_id가 있으면 질의에 살짝 녹여서 semantic_search 재사용
    if room_id:
        combined_query = f"{room_id} 안에 있는 {query}"
    else:
        combined_query = query

    # 같은 파일 안의 semantic_search 엔드포인트 재사용
    result = await semantic_search(query=combined_query, top_k=top_k)

    # 나중에 필요하면 여기서 room_id 기반 필터링/3DSG 위치 붙이기 등 추가
    result["original_query"] = query
    result["room_id"] = room_id
    result["combined_query"] = combined_query
    return result



# api.py 안, 적당한 Tool 섹션에 추가

@app.post("/hybrid/find_affordance_target")
async def find_affordance_target(
    request: AffordanceQuery = Body(...)
):
    """
    [Hybrid] 특정 affordance를 만족하는 개체를 찾습니다.
    - 3DSG JSON (SCENE_3DSG_DATA["output"]["object"])의 'action_affordance'를 직접 검색합니다.
    - 예: affordance="sit_on" 또는 "sit on" → "sit on"이 있는 의자/소파/벤치만 반환
    """
    if not SCENE_3DSG_DATA:
        raise HTTPException(status_code=503, detail="3DSG data not loaded")

    # temp.json 구조: SCENE_3DSG_DATA["output"]["object"][object_id]
    obj_dict = SCENE_3DSG_DATA.get("output", {}).get("object", {})
    if not obj_dict:
        raise HTTPException(status_code=500, detail="3DSG object data not found in JSON")

    affordance_raw = request.affordance or ""
    # "sit_on" / "sit-on" / "sit on" → "sit on" 으로 정규화
    affordance_norm = (
        affordance_raw.lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    print(f"--- API: /hybrid/find_affordance_target (affordance={affordance_raw} -> {affordance_norm}, room_id={request.room_id}) ---")

    results = []

    for obj_id, obj_data in obj_dict.items():
        # room 필터 (있으면)
        if request.room_id is not None:
            parent_room = obj_data.get("parent_room")
            # 요청이 "6" 이든 "room_6" 이든 그냥 문자열로 비교
            if str(parent_room) != str(request.room_id).replace("room_", ""):
                continue

        afford_list = obj_data.get("action_affordance", []) or []
        # affordance 리스트를 전부 소문자/공백 기준으로 봄
        afford_list_norm = [a.lower().strip() for a in afford_list]

        # 완전 일치 또는 부분 일치 허용
        matched = any(
            (affordance_norm == a) or (affordance_norm in a)
            for a in afford_list_norm
        )
        if not matched:
            continue

        class_name = obj_data.get("class_", "object")
        materials = obj_data.get("material") or []
        if isinstance(materials, list):
            materials = [str(m) for m in materials if m]
            material_str = ", ".join(materials)
        else:
            material_str = str(materials) if materials else ""

        # 3DSG object id → KG 스타일 id로 맞춰줌 (예: chair + 23 → "chair_23")
        node_id = f"{class_name}_{obj_id}"

        desc = f"A {class_name}"
        if material_str:
            desc = f"A {material_str} {class_name}"
        desc = desc + f" that affords '{affordance_norm}'."

        results.append(
            {
                "id": node_id,
                "types": ["PhysicalObject", class_name.capitalize()],
                "description": desc,
                "score": 1.0,          # affordance 직접 매칭이므로 confidence=1.0
                "source": "3DSG_JSON",
            }
        )

    # score 기준 정렬 + top_k 자르기 (나중에 score 다르게 줄 거면 여기에 반영)
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    results = results[: request.top_k]

    return {
        "status": "success",
        "affordance": affordance_raw,
        "normalized_affordance": affordance_norm,
        "count": len(results),
        "results": results,
    }





    
@app.get("/kg/get_room")
async def api_get_room(
    object_id: str = Query(..., description="개체 id (예: 'bottle_2')")
):
    """
    [KG] 개체가 속한 room/space 정보를 반환합니다.
    예: bottle_2 -> kitchen_20
    """
    if manager is None:
        raise HTTPException(status_code=503, detail="Ontology manager not initialized")

    try:
        with manager.driver.session() as session:
            record = session.run(
                """
                MATCH (o:Individual {id: $id})-[:inSpace]->(r)
                RETURN
                  r.id          AS room_id,
                  labels(r)     AS room_labels,
                  r.description AS room_description
                """,
                id=object_id,
            ).single()
    except Exception as e:
        # 여기서 예외를 로깅하고 500으로 넘김
        print(f"[ERROR] /kg/get_room Neo4j query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j error: {e}")

    # inSpace 관계가 없을 때 → room 정보를 None으로 돌려줌 (500 내지 않음)
    if record is None or record["room_id"] is None:
        return {
            "status": "success",
            "object_id": object_id,
            "room": None,
        }

    return {
        "status": "success",
        "object_id": object_id,
        "room": {
            "id": record["room_id"],
            "types": record.get("room_labels", []),
            "description": record.get("room_description"),
        },
    }

@app.get("/scene/get_position")
async def api_get_position(
    object_id: str = Query(..., description="3DSG object id (예: '29' 또는 'chair_29')")
):
    """
    [3DSG] 3DSG 데이터에서 특정 개체의 3D 위치를 반환합니다.

    - 기본적으로 temp.json (SCENE_3DSG_DATA["output"]["object"])를 사용.
    - object_id는 다음 두 가지 형식을 모두 지원:
      - "29"              : 3DSG 원본 키
      - "chair_29" 등     : '<class>_<id>' 형태 (벡터/KG에서 쓰는 형식)
    - 나중에 Neo4j 3DSG로 옮겨도 응답 포맷만 유지하면 scene_tools 코드는 안 바꿔도 됨.
    """
    print(f"--- API: /scene/get_position called (object_id={object_id}) ---")

    if not SCENE_3DSG_DATA:
        raise HTTPException(status_code=503, detail="3DSG data not loaded")

    obj_dict = SCENE_3DSG_DATA.get("output", {}).get("object", {})
    if not obj_dict:
        raise HTTPException(status_code=503, detail="3DSG object data not loaded")

    # 1) 우선 전달된 object_id 그대로 시도
    candidate_ids = [object_id]

    # 2) 만약 키가 없고, 'chair_29' 같은 형식이라면 뒤의 숫자만 떼어서 재시도
    if object_id not in obj_dict and "_" in object_id:
        suffix = object_id.split("_")[-1]  # "chair_29" -> "29"
        if suffix not in candidate_ids:
            candidate_ids.append(suffix)

    resolved_id = None
    obj_data = None

    for oid in candidate_ids:
        data = obj_dict.get(oid)
        if data is None:
            continue
        loc = data.get("location")
        if not loc or len(loc) != 3:
            # 위치 정보가 이상하면 패스하고 다른 candidate를 시도
            continue
        resolved_id = oid
        obj_data = data
        break

    if obj_data is None or resolved_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Object '{object_id}' not found in 3DSG data."
        )

    x, y, z = obj_data["location"]

    return {
        "status": "success",
        # 클라이언트가 원래 요청한 id (chair_29 같은 것)를 그대로 돌려줌
        "id": object_id,
        # 실제 3DSG 내부에서 사용된 key도 같이 전달
        "resolved_id": resolved_id,
        "position": {
            "x": x,
            "y": y,
            "z": z,
        },
        "parent_room": obj_data.get("parent_room"),
        "class_": obj_data.get("class_"),
    }

@app.get("/graph/get_node_info")
async def api_get_node_info(
    node_id: str = Query(..., description="KG node id (예: 'chair_24', 'bottle_2')")
):
    """
    [Unified] KG + 3DSG에서 node_id에 해당하는 정보를 한 번에 가져오는 엔드포인트.

    - KG: Neo4j에서 id=node_id인 노드 + inSpace room 정보
    - 3DSG: SCENE_3DSG_DATA에서 위치/크기/재질 등 (suffix 매칭: 'chair_24' -> '24')
    """
    # --- 1) KG 쪽 정보 조회 ---
    if manager is None:
        raise HTTPException(status_code=503, detail="Ontology manager not initialized")

    kg_info = None
    room_info = None

    try:
        with manager.driver.session() as session:
            record = session.run(
                """
                MATCH (n {id: $id})
                OPTIONAL MATCH (n)-[:inSpace]->(r)
                RETURN
                  n.id          AS id,
                  labels(n)     AS labels,
                  n.description AS description,
                  n.room_id     AS room_id,
                  r.id          AS in_space_room_id,
                  labels(r)     AS room_labels,
                  r.description AS room_description
                """,
                id=node_id,
            ).single()
    except Exception as e:
        print(f"[ERROR] /graph/get_node_info Neo4j query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j error: {e}")

    if record:
        kg_info = {
            "id": record["id"],
            "types": [l for l in record["labels"] if l != "Individual"],
            "description": record.get("description"),
        }

        # room_id는 inSpace로 연결된 게 있으면 그걸 우선 사용
        room_id = record.get("in_space_room_id") or record.get("room_id")
        if room_id:
            room_info = {
                "id": room_id,
                "types": record.get("room_labels", []),
                "description": record.get("room_description"),
            }

    # --- 2) 3DSG 쪽 정보 조회 (temp.json) ---
    scene_info = None

    if SCENE_3DSG_DATA:
        obj_dict = SCENE_3DSG_DATA.get("output", {}).get("object", {}) or {}

        # 우선 node_id 그대로 시도, 안 되면 suffix 숫자(e.g., "chair_24" -> "24")로 시도
        candidate_ids = [node_id]
        if node_id not in obj_dict and "_" in node_id:
            suffix = node_id.split("_")[-1]
            if suffix not in candidate_ids:
                candidate_ids.append(suffix)

        resolved_id = None
        obj_data = None

        for oid in candidate_ids:
            data = obj_dict.get(oid)
            if data is None:
                continue
            loc = data.get("location")
            if not loc or len(loc) != 3:
                continue
            resolved_id = oid
            obj_data = data
            break

        if obj_data:
            x, y, z = obj_data["location"]
            scene_info = {
                "id": resolved_id,
                "position": {"x": x, "y": y, "z": z},
                "class_": obj_data.get("class_"),
                "parent_room": obj_data.get("parent_room"),
                "size": obj_data.get("size"),
                "material": obj_data.get("material"),
                "floor_area": obj_data.get("floor_area"),
                "surface_coverage": obj_data.get("surface_coverage"),
                "volume": obj_data.get("volume"),
            }

    # --- 3) 둘 다 없으면 404 ---
    if not kg_info and not scene_info:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{node_id}' not found in KG nor 3DSG.",
        )

    return {
        "status": "success",
        "id": node_id,
        "kg": {
            **kg_info,
            "room": room_info,
        } if kg_info else None,
        "scene": scene_info,
    }



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "manager_ready": manager is not None}



# --- Added 3DSG convenience endpoints (fuzzy position, room, listing) ---
@app.get('/scene/get_position_fuzzy')
async def api_get_position_fuzzy(object_id: str = Query(..., description="3DSG object id ('29','chair_29') or class name ('chair')")):
    if not SCENE_3DSG_DATA:
        raise HTTPException(status_code=503, detail="3DSG data not loaded")
    obj_dict = SCENE_3DSG_DATA.get('output', {}).get('object', {})
    if not obj_dict:
        raise HTTPException(status_code=503, detail="3DSG object data not loaded")
    candidates = [object_id]
    if object_id not in obj_dict and '_' in object_id:
        suffix = object_id.split('_')[-1]
        candidates.append(suffix)
    for oid in candidates:
        data = obj_dict.get(oid)
        loc = data.get('location') if data else None
        if loc and len(loc)==3:
            return {
                'status':'success',
                'matched_by':'id',
                'query':object_id,
                'id':oid,
                'position':{'x':loc[0],'y':loc[1],'z':loc[2]},
                'parent_room':data.get('parent_room'),
                'class_':data.get('class_'),
            }
    matches=[]
    for oid,data in obj_dict.items():
        if data.get('class_','').lower()==object_id.lower():
            loc=data.get('location')
            if loc and len(loc)==3:
                matches.append({'id':oid,'class_':data.get('class_'),'parent_room':data.get('parent_room'),'position':{'x':loc[0],'y':loc[1],'z':loc[2]}})
    if matches:
        return {'status':'success','matched_by':'class','query':object_id,'count':len(matches),'results':matches}
    raise HTTPException(status_code=404, detail=f"Object '{object_id}' not found in 3DSG data.")

@app.get('/scene/get_room')
async def api_scene_get_room(object_id: str = Query(..., description="3DSG object id (e.g., '29' or 'chair_29')")):
    if not SCENE_3DSG_DATA:
        raise HTTPException(status_code=503, detail="3DSG data not loaded")
    obj_dict = SCENE_3DSG_DATA.get('output', {}).get('object', {})
    if not obj_dict:
        raise HTTPException(status_code=503, detail="3DSG object data not loaded")
    candidates=[object_id]
    if object_id not in obj_dict and '_' in object_id:
        candidates.append(object_id.split('_')[-1])
    for oid in candidates:
        data=obj_dict.get(oid)
        if data is not None:
            return {'status':'success','id':object_id,'resolved_id':oid,'room_id':data.get('parent_room'),'class_':data.get('class_')}
    raise HTTPException(status_code=404, detail=f"Object '{object_id}' not found in 3DSG data.")

@app.get('/scene/list_in_category')
async def api_scene_list_in_category(category: str = Query(..., description="3DSG class name, e.g., 'chair'"), top_k: Optional[int] = Query(None, description="Optional cap on results")):
    if not SCENE_3DSG_DATA:
        raise HTTPException(status_code=503, detail="3DSG data not loaded")
    obj_dict = SCENE_3DSG_DATA.get('output', {}).get('object', {})
    if not obj_dict:
        raise HTTPException(status_code=503, detail="3DSG object data not loaded")
    results=[]
    for oid,data in obj_dict.items():
        if data.get('class_','').lower()!=category.lower():
            continue
        loc=data.get('location')
        position=None
        if loc and len(loc)==3:
            position={'x':loc[0],'y':loc[1],'z':loc[2]}
        results.append({'id':oid,'class_':data.get('class_'),'parent_room':data.get('parent_room'),'position':position})
        if top_k and len(results)>=top_k:
            break
    return {'status':'success','category':category,'count':len(results),'results':results}

@app.get('/scene/search')
async def api_scene_search(query: str = Query(..., description="Free-text class search, e.g., 'chair'"), top_k: int = Query(5, description="Number of results to return")):
    if not SCENE_3DSG_DATA:
        raise HTTPException(status_code=503, detail="3DSG data not loaded")
    obj_dict = SCENE_3DSG_DATA.get('output', {}).get('object', {})
    if not obj_dict:
        raise HTTPException(status_code=503, detail="3DSG object data not loaded")
    q=query.lower()
    results=[]
    for oid,data in obj_dict.items():
        cls=data.get('class_','')
        if q in cls.lower():
            loc=data.get('location')
            position=None
            if loc and len(loc)==3:
                position={'x':loc[0],'y':loc[1],'z':loc[2]}
            results.append({'id':oid,'class_':cls,'parent_room':data.get('parent_room'),'position':position})
        if len(results)>=top_k:
            break
    return {'status':'success','query':query,'count':len(results),'results':results}

@app.get('/scene/shortest_path')
async def api_scene_shortest_path(
    start_room_id: str = Query(..., description="Start room id (e.g., '20' or 20)"),
    goal_room_id: str = Query(..., description="Goal room id (e.g., '9' or 9)"),
):
    """Compute a simple shortest path between rooms using 3DSG connections."""
    if not SCENE_3DSG_DATA:
        raise HTTPException(status_code=503, detail="3DSG data not loaded")
    connections = SCENE_3DSG_DATA.get("output", {}).get("connections", {})
    if not connections:
        raise HTTPException(status_code=503, detail="3DSG connections not loaded")

    adj: Dict[str, set] = {}
    for conn in connections.values():
        rooms = conn.get("connected_rooms") or []
        if len(rooms) != 2:
            continue
        a, b = str(rooms[0]), str(rooms[1])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    start = str(start_room_id).replace("room_", "")
    goal = str(goal_room_id).replace("room_", "")

    if start not in adj or goal not in adj:
        raise HTTPException(status_code=404, detail="Start or goal room not found in connections.")

    from collections import deque
    queue = deque([[start]])
    seen = {start}
    found_path = None
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            found_path = path
            break
        for nei in adj.get(node, []):
            if nei not in seen:
                seen.add(nei)
                queue.append(path + [nei])

    if not found_path:
        return {"status": "success", "reachable": False, "start": start, "goal": goal, "path": []}

    return {
        "status": "success",
        "reachable": True,
        "start": start,
        "goal": goal,
        "path": found_path,
        "hops": len(found_path) - 1,
    }


# KG shortest path endpoint (room/space graph in Neo4j)
@app.get('/kg/shortest_path')
async def kg_shortest_path(
    start_id: str = Query(..., description="Start room/space id (e.g., 'kitchen_20' or '20')"),
    goal_id: str = Query(..., description="Goal room/space id (e.g., 'bedroom_9' or '9')"),
    max_hops: int = Query(10, description="Max hops for search (default 10)"),
    allow_closed: bool = Query(True, description="If false, skip relationships with status='closed'")
):
    if manager is None:
        raise HTTPException(status_code=503, detail="Ontology manager not initialized")

    max_hops = max(1, min(max_hops, 50))  # clamp for safety

    # Normalize ids: allow bare numbers ("9") or prefixed "room_9"
    def normalize(rid: str) -> str:
        s = str(rid)
        if s.startswith('room_'):
            return s
        if s.isdigit():
            return f"room_{s}"
        return s

    start_norm = normalize(start_id)
    goal_norm = normalize(goal_id)

    query = f"""
    MATCH (start {{id: $start}}), (goal {{id: $goal}})
    MATCH p=shortestPath((start)-[:connectedTo*..{max_hops}]->(goal))
    {'' if allow_closed else 'WHERE all(r IN relationships(p) WHERE coalesce(r.status,"open") <> "closed")'}
    RETURN p
    """

    try:
        with manager.driver.session() as session:
            record = session.run(query, start=start_norm, goal=goal_norm).single()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j error: {e}")

    if not record:
        return {
            "status": "success",
            "reachable": False,
            "start": start_norm,
            "goal": goal_norm,
            "path": [],
        }

    path = record[0]
    node_ids = [n.get("id") for n in path.nodes]
    rels = []
    for rel in path.relationships:
        rels.append({"type": rel.type, "status": rel.get("status")})

    return {
        "status": "success",
        "reachable": True,
        "start": start_norm,
        "goal": goal_norm,
        "path": node_ids,
        "hops": len(node_ids) - 1,
        "relationships": rels,
    }

