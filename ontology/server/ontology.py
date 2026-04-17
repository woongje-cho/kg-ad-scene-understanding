#!/usr/bin/env python3
"""
OntologyManager: Core ontology management with owlready2 + reasoner + Neo4j
"""

import os
import owlready2 as owl
from neo4j import GraphDatabase
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback
from dotenv import load_dotenv
from .config import get_config

# Load environment variables from .env file
load_dotenv()


class OntologyManager:
    """Manage OWL ontology with real-time Neo4j synchronization."""

    def __init__(self, owl_path: str = "data/robot.owl",
                 env_id: Optional[str] = None,
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None):
        """Initialize ontology manager.

        Args:
            owl_path: Path to shared ontology schema file
            env_id: ID of the space to load (optional, for context)
            neo4j_uri: Neo4j connection URI (defaults to config.yaml or NEO4J_URI env)
            neo4j_user: Neo4j username (defaults to config.yaml or NEO4J_USER env)
            neo4j_password: Neo4j password (defaults to config.yaml or NEO4J_PASSWORD env)
        """
        self.owl_path = owl_path
        self.env_id = env_id

        # Resolve Neo4j connection info in priority: explicit arg -> env var -> config.yaml defaults
        config = get_config()
        neo4j_cfg = config.get_neo4j_config()
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI") or neo4j_cfg.get("uri")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER") or neo4j_cfg.get("user")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD") or neo4j_cfg.get("password")

        missing = [name for name, value in [("NEO4J_URI", self.neo4j_uri),
                                            ("NEO4J_USER", self.neo4j_user),
                                            ("NEO4J_PASSWORD", self.neo4j_password)] if not value]
        if missing:
            raise ValueError(f"Missing Neo4j configuration values: {', '.join(missing)}")

        self.world = None
        self.ontology = None
        self.driver = None

        # Load OWL schema
        self._load_ontology()

        # Connect to Neo4j
        self._connect_neo4j()

        # Initialize Neo4j with schema
        self._initialize_neo4j_schema()

        space_info = f" (space: {env_id})" if env_id else ""
        print(f"✅ OntologyManager initialized successfully{space_info}")

    def _load_ontology(self):
        """Load OWL ontology schema."""
        try:
            self.world = owl.World()
            self.world.get_ontology(f"file://{Path(self.owl_path).absolute()}").load()

            # Get ontology by IRI
            ontology_iri = "http://www.semanticweb.org/namh_woo/ontologies/2025/9/untitled-ontology-5"
            self.ontology = self.world.get_ontology(ontology_iri)

            if not self.ontology:
                self.ontology = list(self.world.ontologies())[0]

            classes = list(self.ontology.classes())
            obj_props = list(self.ontology.object_properties())
            data_props = list(self.ontology.data_properties())

            print(f"✓ Loaded OWL ontology: {len(classes)} classes, "
                  f"{len(obj_props)} object properties, {len(data_props)} data properties")

        except Exception as e:
            print(f"✗ Failed to load ontology: {e}")
            raise

    def _connect_neo4j(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✓ Connected to Neo4j")
        except Exception as e:
            print(f"✗ Neo4j connection failed: {e}")
            raise

    def _initialize_neo4j_schema(self):
        """Initialize Neo4j with OWL schema (classes, properties, hierarchy)."""
        try:
            with self.driver.session() as session:
                # Clear all data
                session.run("MATCH (n) DETACH DELETE n")

                # Sync classes
                for cls in self.ontology.classes():
                    session.run("""
                        MERGE (c:Class {id: $class_id})
                        SET c.name = $class_name,
                            c.uri = $class_uri
                    """, class_id=cls.name, class_name=cls.name, class_uri=str(cls.iri))

                # Sync class hierarchy
                for cls in self.ontology.classes():
                    for parent in cls.is_a:
                        if hasattr(parent, 'name') and parent != owl.Thing:
                            session.run("""
                                MATCH (child:Class {id: $child_id})
                                MATCH (parent:Class {id: $parent_id})
                                MERGE (child)-[:SUBCLASS_OF]->(parent)
                            """, child_id=cls.name, parent_id=parent.name)

                # Sync properties
                for prop in self.ontology.object_properties():
                    session.run("""
                        MERGE (p:Property {id: $prop_id})
                        SET p.name = $prop_name,
                            p.uri = $prop_uri,
                            p.type = 'ObjectProperty'
                    """, prop_id=prop.name, prop_name=prop.name, prop_uri=str(prop.iri))

                for prop in self.ontology.data_properties():
                    session.run("""
                        MERGE (p:Property {id: $prop_id})
                        SET p.name = $prop_name,
                            p.uri = $prop_uri,
                            p.type = 'DataProperty'
                    """, prop_id=prop.name, prop_name=prop.name, prop_uri=str(prop.iri))

                # Sync property domains and ranges
                for prop in self.ontology.object_properties():
                    for domain in prop.domain:
                        if hasattr(domain, 'name'):
                            session.run("""
                                MATCH (p:Property {id: $prop_id})
                                MATCH (c:Class {id: $class_id})
                                MERGE (p)-[:HAS_DOMAIN]->(c)
                            """, prop_id=prop.name, class_id=domain.name)

                    for range_cls in prop.range:
                        if hasattr(range_cls, 'name'):
                            session.run("""
                                MATCH (p:Property {id: $prop_id})
                                MATCH (c:Class {id: $class_id})
                                MERGE (p)-[:HAS_RANGE]->(c)
                            """, prop_id=prop.name, class_id=range_cls.name)

                # Sync property hierarchy
                for prop in self.ontology.object_properties():
                    for parent in prop.is_a:
                        if hasattr(parent, 'name') and parent != owl.ObjectProperty:
                            if parent.name not in ['SymmetricProperty', 'TransitiveProperty', 'topObjectProperty']:
                                session.run("""
                                    MATCH (child:Property {id: $child_id})
                                    MATCH (parent:Property {id: $parent_id})
                                    MERGE (child)-[:SUBPROPERTY_OF]->(parent)
                                """, child_id=prop.name, parent_id=parent.name)

                # Setup vector index for semantic search
                self._setup_vector_index(session)

            print("✓ Initialized Neo4j with OWL schema")

        except Exception as e:
            print(f"✗ Failed to initialize Neo4j schema: {e}")
            raise

    def _setup_vector_index(self, session):
        """Setup vector index for semantic search based on config.yaml."""
        try:
            from .config import get_config

            # Load embedding configuration
            config = get_config()
            embedding_config = config.get_embedding_config()
            required_dimensions = embedding_config.get('dimensions', 512)

            # Check if index exists and get its configuration
            result = session.run("SHOW INDEXES")
            existing_index = None
            for record in result:
                if record.get("name") == "individualEmbeddingIndex":
                    existing_index = record
                    break

            # Determine if we need to recreate the index
            recreate_index = False
            if existing_index is None:
                print(f"ℹ️  Vector index not found, creating with {required_dimensions} dimensions...")
                recreate_index = True
            else:
                # Check if dimensions match
                # Note: Neo4j doesn't expose dimension info directly in SHOW INDEXES
                # We'll drop and recreate if config changed
                print(f"ℹ️  Existing vector index found, verifying dimensions...")
                recreate_index = True  # Always recreate to ensure correct dimensions

            if recreate_index:
                # Drop existing index if present
                session.run("DROP INDEX individualEmbeddingIndex IF EXISTS")
                print(f"✓ Dropped old vector index")

                # Create new index with correct dimensions
                session.run(f"""
                    CREATE VECTOR INDEX individualEmbeddingIndex
                    FOR (n:Individual)
                    ON n.embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {required_dimensions},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                print(f"✓ Created vector index with {required_dimensions} dimensions")

        except Exception as e:
            print(f"⚠️  Vector index setup failed: {e}")
            print("   (This is optional - semantic search will not work without it)")

    def add_individual(self, data: Dict[str, Any], auto_sync: bool = True) -> Dict[str, Any]:
        """
        Add a new individual to the ontology.

        Args:
            data: {
                "id": "room_101",
                "class": "Room",
                "data_properties": {"roomNumber": "101"},
                "object_properties": {"isSpaceOf": "floor_1"}
            }
            auto_sync: Whether to automatically sync to Neo4j (default: True)

        Returns:
            Status dictionary
        """
        try:
            individual_id = data["id"]
            class_name = data["class"]

            # Get class from ontology
            cls = getattr(self.ontology, class_name, None)
            if not cls:
                return {"status": "error", "message": f"Class {class_name} not found"}

            # Check if individual already exists
            existing = self.ontology.search_one(iri=f"*{individual_id}")
            if existing:
                return {"status": "error", "message": f"Individual {individual_id} already exists"}

            # Create individual
            individual = cls(individual_id)

            # Set data properties
            if "data_properties" in data:
                for prop_name, value in data["data_properties"].items():
                    setattr(individual, prop_name, value)

            # Set object properties
            if "object_properties" in data:
                for prop_name, target_ids in data["object_properties"].items():
                    if not isinstance(target_ids, list):
                        target_ids = [target_ids]

                    targets = []
                    for target_id in target_ids:
                        target = self.ontology.search_one(iri=f"*{target_id}")
                        if target:
                            targets.append(target)

                    if targets:
                        setattr(individual, prop_name, targets)

            print(f"✓ Added individual: {individual_id}")

            # Auto sync (optional)
            if auto_sync:
                self.sync_to_neo4j()

            return {"status": "success", "id": individual_id}

        except Exception as e:
            print(f"✗ Failed to add individual: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def load_instances_from_ttl(self, ttl_path: str) -> Dict[str, Any]:
        """
        Load individuals from TTL file by parsing and using owlready2's add_individual API.

        Args:
            ttl_path: Path to TTL file containing individual instances

        Returns:
            Status dictionary with count of loaded individuals
        """
        try:
            import rdflib
            from rdflib.namespace import RDF

            ttl_file = Path(ttl_path).absolute()
            if not ttl_file.exists():
                return {"status": "error", "message": f"TTL file not found: {ttl_path}"}

            print(f"📖 Loading individuals from TTL: {ttl_path}")

            # Parse TTL using rdflib
            g = rdflib.Graph()
            g.parse(str(ttl_file), format="turtle")
            print(f"  Parsed {len(g)} triples from TTL")

            # Extract individuals from TTL
            individuals_data = []
            subjects = set(g.subjects(RDF.type, None))

            for subject in subjects:
                subject_id = str(subject).split('#')[-1]

                # Skip ontology declaration
                if 'Ontology' in str(subject) or subject_id == '':
                    continue

                # Get rdf:type (class)
                types = [obj for obj in g.objects(subject, RDF.type)
                        if 'Ontology' not in str(obj)]
                if not types:
                    continue

                class_name = str(types[0]).split('#')[-1]

                # Get data and object properties
                data_properties = {}
                object_properties = {}

                for pred, obj in g.predicate_objects(subject):
                    if pred == RDF.type:
                        continue

                    pred_local = str(pred).split('#')[-1]

                    if isinstance(obj, rdflib.Literal):
                        data_properties[pred_local] = obj.toPython()
                    elif isinstance(obj, rdflib.URIRef):
                        obj_local = str(obj).split('#')[-1]
                        if pred_local not in object_properties:
                            object_properties[pred_local] = []
                        object_properties[pred_local].append(obj_local)

                individuals_data.append({
                    "id": subject_id,
                    "class": class_name,
                    "data_properties": data_properties,
                    "object_properties": object_properties
                })

            print(f"  Extracted {len(individuals_data)} individuals")

            # Use batch add method
            result = self.add_individuals_batch(individuals_data)
            return result

        except Exception as e:
            print(f"✗ Failed to load TTL file: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def add_individuals_batch(self, individuals_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple individuals at once and run reasoning only once at the end.
        Uses 2-pass approach to handle dependencies.

        Args:
            individuals_data: List of individual data dictionaries

        Returns:
            Status dictionary with count of added individuals
        """
        try:
            added_count = 0
            failed_count = 0

            print(f"Adding {len(individuals_data)} individuals in batch...")

            # Pass 1: Create all individuals without properties (to avoid order dependencies)
            for data in individuals_data:
                individual_id = data["id"]
                class_name = data["class"]

                # Check if already exists
                existing = self.ontology.search_one(iri=f"*{individual_id}")
                if existing:
                    failed_count += 1
                    print(f"✗ Failed to add individual: {individual_id} - Individual {individual_id} already exists")
                    continue

                # Get class from ontology
                cls = getattr(self.ontology, class_name, None)
                if not cls:
                    failed_count += 1
                    print(f"✗ Failed to add individual: {individual_id} - Class {class_name} not found")
                    continue

                # Create individual without properties
                try:
                    cls(individual_id)
                    added_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"✗ Failed to add individual: {individual_id} - {e}")

            # Pass 2: Set properties for all individuals
            for data in individuals_data:
                individual_id = data["id"]
                individual = self.ontology.search_one(iri=f"*{individual_id}")

                if not individual:
                    continue

                # Set data properties
                if "data_properties" in data:
                    for prop_name, value in data["data_properties"].items():
                        setattr(individual, prop_name, value)

                # Set object properties
                if "object_properties" in data:
                    for prop_name, target_ids in data["object_properties"].items():
                        if not isinstance(target_ids, list):
                            target_ids = [target_ids]

                        targets = []
                        for target_id in target_ids:
                            target = self.ontology.search_one(iri=f"*{target_id}")
                            if target:
                                targets.append(target)

                        if targets:
                            setattr(individual, prop_name, targets)

            print(f"✓ Added {added_count} individuals (failed: {failed_count})")

            # Run reasoning once for all individuals
            print("🧠 Running reasoning for all individuals...")
            self.sync_to_neo4j()

            return {
                "status": "success",
                "added": added_count,
                "failed": failed_count
            }

        except Exception as e:
            print(f"✗ Failed to add individuals batch: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def update_individual(self, individual_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing individual."""
        try:
            # Find individual
            individual = self.ontology.search_one(iri=f"*{individual_id}")
            if not individual:
                return {"status": "error", "message": f"Individual {individual_id} not found"}

            # Update data properties
            if "data_properties" in data:
                for prop_name, value in data["data_properties"].items():
                    setattr(individual, prop_name, value)

            # Update object properties
            if "object_properties" in data:
                for prop_name, target_ids in data["object_properties"].items():
                    if not isinstance(target_ids, list):
                        target_ids = [target_ids]

                    targets = []
                    for target_id in target_ids:
                        target = self.ontology.search_one(iri=f"*{target_id}")
                        if target:
                            targets.append(target)

                    if targets:
                        setattr(individual, prop_name, targets)

            print(f"✓ Updated individual: {individual_id}")

            # Auto sync
            self.sync_to_neo4j()

            return {"status": "success", "id": individual_id}

        except Exception as e:
            print(f"✗ Failed to update individual: {e}")
            return {"status": "error", "message": str(e)}

    def delete_individual(self, individual_id: str) -> Dict[str, Any]:
        """Delete an individual from the ontology."""
        try:
            individual = self.ontology.search_one(iri=f"*{individual_id}")
            if not individual:
                return {"status": "error", "message": f"Individual {individual_id} not found"}

            owl.destroy_entity(individual)

            print(f"✓ Deleted individual: {individual_id}")

            # Auto sync
            self.sync_to_neo4j()

            return {"status": "success", "id": individual_id}

        except Exception as e:
            print(f"✗ Failed to delete individual: {e}")
            return {"status": "error", "message": str(e)}

    def sync_to_neo4j(self) -> Dict[str, Any]:
        """Run reasoner and sync all individuals to Neo4j."""
        try:
            print("🧠 Running HermiT reasoner...")

            # Run reasoner
            with self.ontology:
                owl.sync_reasoner_hermit(self.world, infer_property_values=True)

            print("✓ Reasoner completed")

            # Sync to Neo4j
            with self.driver.session() as session:
                # Clear existing individuals
                session.run("MATCH (i:Individual) DETACH DELETE i")

                individuals_count = 0
                relationships_count = 0

                # Sync all individuals
                for individual in self.ontology.individuals():
                    # Collect all class labels (including superclasses via INDIRECT_is_a)
                    # This is fully dynamic - reads from OWL hierarchy automatically
                    class_labels = []
                    for cls in individual.INDIRECT_is_a:
                        if hasattr(cls, 'name') and cls.name and cls != owl.Thing:
                            class_labels.append(cls.name)

                    # Build label string for Neo4j multi-labeling
                    # Example: :Individual:Room:Space for a Room instance
                    labels = "Individual" + "".join(f":`{label}`" for label in class_labels)

                    # Create individual node with multiple labels
                    # Using backticks to handle special characters in class names
                    session.run(f"""
                        MERGE (i:{labels} {{id: $individual_id}})
                        SET i.name = $individual_name,
                            i.uri = $individual_uri
                    """, individual_id=individual.name,
                        individual_name=individual.name,
                        individual_uri=str(individual.iri))

                    individuals_count += 1

                    # Link to classes (using INDIRECT_is_a to include superclasses)
                    for cls in individual.INDIRECT_is_a:
                        if hasattr(cls, 'name') and cls.name and cls != owl.Thing:
                            session.run("""
                                MATCH (i:Individual {id: $individual_id})
                                MATCH (c:Class {id: $class_id})
                                MERGE (i)-[:INSTANCE_OF]->(c)
                            """, individual_id=individual.name, class_id=cls.name)

                    # Sync object properties (using INDIRECT_ to include subproperties)
                    for prop in self.ontology.object_properties():
                        # Use INDIRECT_ prefix to get all values including inferred ones
                        indirect_attr = f"INDIRECT_{prop.name}"
                        prop_values = getattr(individual, indirect_attr, [])
                        if not isinstance(prop_values, list):
                            prop_values = [prop_values] if prop_values else []

                        for value in prop_values:
                            if hasattr(value, 'name'):
                                session.run(f"""
                                    MATCH (subj:Individual {{id: $subj_id}})
                                    MATCH (obj:Individual {{id: $obj_id}})
                                    MERGE (subj)-[:{prop.name}]->(obj)
                                """, subj_id=individual.name, obj_id=value.name)
                                relationships_count += 1

                    # Sync data properties (as node properties)
                    data_props = {}
                    for prop in self.ontology.data_properties():
                        prop_values = getattr(individual, prop.name, [])
                        if not isinstance(prop_values, list):
                            prop_values = [prop_values] if prop_values is not None else []

                        # Check if list is not empty (don't use 'if prop_values' as it fails for [False])
                        if len(prop_values) > 0:
                            data_props[prop.name] = prop_values[0] if len(prop_values) == 1 else prop_values

                    if data_props:
                        for key, value in data_props.items():
                            session.run("""
                                MATCH (i:Individual {id: $individual_id})
                                SET i[$prop_name] = $prop_value
                            """, individual_id=individual.name, prop_name=key, prop_value=value)

                # Generate embeddings for individuals with descriptions
                print("🔄 Generating embeddings...")
                from .embedding import EmbeddingManager
                from .config import get_config

                config = get_config()
                embedding_config = config.get_embedding_config()

                embedding_manager = EmbeddingManager(
                    model=embedding_config.get('model', 'text-embedding-3-small'),
                    dimensions=embedding_config.get('dimensions', 512)
                )

                embeddings_count = 0
                failed_embeddings = 0

                for individual in self.ontology.individuals():
                    try:
                        if embedding_manager.embed_individual(individual, session):
                            embeddings_count += 1
                    except Exception as e:
                        failed_embeddings += 1
                        print(f"✗ Failed to embed {individual.name}: {e}")

                print(f"✓ Generated {embeddings_count} embeddings (failed: {failed_embeddings})")


            return {
                "status": "success",
                "individuals": individuals_count,
                "relationships": relationships_count
            }

        except Exception as e:
            print(f"✗ Sync failed: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current ontology status."""
        try:
            individuals = list(self.ontology.individuals())
            classes = list(self.ontology.classes())

            return {
                "status": "running",
                "ontology": str(self.ontology.base_iri),
                "individuals_count": len(individuals),
                "classes_count": len(classes),
                "individuals": [ind.name for ind in individuals]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self, cleanup_neo4j: bool = True):
        """Close connections and optionally cleanup Neo4j data.

        Args:
            cleanup_neo4j: If True, delete all data from Neo4j before closing
        """
        if self.driver:
            if cleanup_neo4j:
                try:
                    with self.driver.session() as session:
                        session.run("MATCH (n) DETACH DELETE n")
                    print("✓ Cleaned up Neo4j data")
                except Exception as e:
                    print(f"⚠ Failed to cleanup Neo4j: {e}")

            self.driver.close()
            print("✓ Closed Neo4j connection")
