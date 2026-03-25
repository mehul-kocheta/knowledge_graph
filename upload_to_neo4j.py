import json
import re
from neo4j import GraphDatabase

# --------------------------------
# NEO4J CONFIGURATION
# --------------------------------

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

JSON_FILE = "kg_extraction.json"

# --------------------------------
# EXTRACT JSON
# --------------------------------

def extract_json(text):
    if not text:
        return None

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except:
        return None


def merge_kg(data):
    entities = {}
    relationships = set()

    for page in data:
        parsed = page.get("data", {})
        
        # If the LLM returned markdown wrapped json, it might be sitting in raw_output
        if "raw_output" in parsed:
            raw_parsed = extract_json(parsed["raw_output"])
            if raw_parsed:
                parsed = raw_parsed

        if not parsed:
            continue

        # Extract explicitly defined entities
        for e in parsed.get("entities", []):
            name = e.get("name", "").strip()
            typ = e.get("type", "Unknown").strip()
            if name:
                entities[name] = typ

        # Extract relationships and implicitly define missing nodes
        for r in parsed.get("relationships", []):
            s = r.get("source", "").strip()
            rel = r.get("relation", "").strip()
            t = r.get("target", "").strip()

            if s and rel and t:
                relationships.add((s, rel, t))
                
                if s not in entities:
                    entities[s] = "Unknown"
                if t not in entities:
                    entities[t] = "Unknown"

    return entities, relationships


# --------------------------------
# NEO4J UPLOAD FUNCTIONS
# --------------------------------

def clean_label(label):
    """
    Neo4j labels should be alphanumeric and ideally PascalCase. 
    This is a simple cleaner to remove invalid characters.
    """
    cleaned = ''.join(e for e in label if e.isalnum())
    return cleaned if cleaned else "Concept"


def create_nodes(tx, entities):
    """
    Creates multiple nodes with their specific labels in a batch manner where possible.
    Since nodes have different labels, we group them by label and then insert.
    """
    print(f"Uploading {len(entities)} nodes to Neo4j...")
    
    # Group entities by label
    grouped = {}
    for name, typ in entities.items():
        label = clean_label(typ)
        if label not in grouped:
            grouped[label] = []
        grouped[label].append({"name": name})

    for label, nodes in grouped.items():
        query = (
            f"UNWIND $nodes AS node "
            f"MERGE (n:`{label}` {{name: node.name}})"
        )
        tx.run(query, nodes=nodes)


def create_edges(tx, relationships):
    """
    Creates relationships between nodes.
    Neo4j relationship types must be alphanumeric and usually uppercase.
    """
    print(f"Uploading {len(relationships)} relationships to Neo4j...")
    
    # Group edges by relation type
    grouped = {}
    for s, rel, t in relationships:
        rel_type = clean_label(rel).upper()
        if rel_type not in grouped:
            grouped[rel_type] = []
        grouped[rel_type].append({"source": s, "target": t})

    for rel_type, edges in grouped.items():
        query = (
            f"UNWIND $edges AS edge "
            f"MATCH (s {{name: edge.source}}) "
            f"MATCH (t {{name: edge.target}}) "
            f"MERGE (s)-[:`{rel_type}`]->(t)"
        )
        tx.run(query, edges=edges)


# --------------------------------
# MAIN
# --------------------------------

if __name__ == "__main__":
    
    # Read JSON
    try:
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {JSON_FILE} not found. Please run the extraction script first.")
        exit(1)

    entities, relationships = merge_kg(data)

    print(f"Found {len(entities)} unique entities and {len(relationships)} relationship edges.")

    # Upload to Neo4j
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        # We use driver.execute_query to automatically manage transactions
        with driver.session() as session:
            session.execute_write(create_nodes, entities)
            session.execute_write(create_edges, relationships)
            
    print("\nUpload to Neo4j complete! Open http://localhost:7474 to browse your graph.")
