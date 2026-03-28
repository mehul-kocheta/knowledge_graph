import fitz  # PyMuPDF
from google import genai
from PIL import Image
import json
import io
import os
import asyncio
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ---------------------------
# CONFIG
# ---------------------------

client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
# Load a small local embedding model (384 dimensions)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# HELPERS
# ---------------------------

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

async def generate_embedding(text):
    """Generates embedding for a single text string using local model."""
    # SentenceTransformer.encode is synchronous, but we can wrap it or just call it 
    # since it's relatively fast for single strings.
    # For better async performance with many strings, we could use a thread pool.
    emb = embedding_model.encode(text)
    return emb.tolist()

def clean_label(label):
    cleaned = ''.join(e for e in label if e.isalnum())
    return cleaned if cleaned else "Concept"

# ---------------------------
# PROMPT
# ---------------------------

PROMPT = """
You are an information extraction system.

From the page image extract:

1. Entities
2. Relationships between entities

Previously extracted entities from this resource (use these exactly as written for consistency if referring to the same concept):
{previous_entities}

Ensure entities are not isolated but are connected to each other through relationships.

Make sure each entity is independant of the source and other entities for eg: A Complete formula, A table, A name, A concept, A place (not just reference of table/formula/equation)
Each enitity should contain a meaningful info on it's own

Ignore the references / bibliography / table of content / cover page parts, only extract things that are meaningfull, otherwise result empty json

Return output in JSON format:

{{
  "entities": [
    {{"name": "", "type": ""}}
  ],
  "relationships": [
    {{"source": "", "relation": "", "target": ""}}
  ]
}}
"""

# ---------------------------
# PDF → Images
# ---------------------------

def pdf_to_images(pdf_content):
    """pdf_content can be a path or bytes"""
    if isinstance(pdf_content, bytes):
        doc = fitz.open(stream=pdf_content, filetype="pdf")
    else:
        doc = fitz.open(pdf_content)

    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images

    return images


# ---------------------------
# VECTOR INDEX SETUP
# ---------------------------

def setup_vector_index(tx):
    """Creates a vector index on the 'Concept' label for the 'embedding' property."""
    # Check if the index already exists to avoid errors
    result = tx.run("SHOW INDEXES YIELD name WHERE name = 'entity_embeddings' RETURN count(*) > 0 AS exists").single()
    if result and result["exists"]:
        return

    # Use the procedural call for compatibility with Neo4j 5.11+ (including 5.14.0)
    tx.run("CALL db.index.vector.createNodeIndex('entity_embeddings', 'Concept', 'embedding', 384, 'cosine')")


# ---------------------------
# IMAGE → LLM
# ---------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def extract_from_image(image, page_number, previous_entities):
    # Check API key
    if not os.environ.get('GEMINI_API_KEY'):
        print("CRITICAL: GEMINI_API_KEY environment variable is not set!")
    
    print(f"Processing page {page_number}... Image size: {image.size}, Mode: {image.mode}")
    
    formatted_prompt = PROMPT.replace(
        "{previous_entities}",
        json.dumps(previous_entities, indent=2) if previous_entities else "None"
    )

    try:
        # Use the genai client
        response = await client.aio.models.generate_content(
            model='gemini-3-flash-preview', 
            contents=[formatted_prompt, image]
        )
    except Exception as e:
        print(f"GenAI Client Error for page {page_number}: {type(e).__name__}: {str(e)}")
        # Check if it has internal details
        if hasattr(e, 'response'):
             print(f"Error Response: {getattr(e, 'response')}")
        raise e

    text = response.text
    
    try:
        data = extract_json(text)
        if not data:
             data = {"entities": [], "relationships": []}
    except:
        data = {"entities": [], "relationships": []}

    print(f"Finished page {page_number}")
    return {
        "page": page_number,
        "data": data
    }

# ---------------------------
# NEO4J UPLOAD FUNCTIONS
# ---------------------------

def create_nodes(tx, entities_metadata):
    """
    entities_metadata: dict mapping name -> {"type": typ, "page_numbers": set, "pdf_url": url, "embedding": list}
    """
    grouped = {}
    for name, meta in entities_metadata.items():
        label = clean_label(meta["type"])
        if label not in grouped:
            grouped[label] = []
        
        grouped[label].append({
            "name": name,
            "page_numbers": list(meta["page_numbers"]),
            "pdf_url": meta["pdf_url"],
            "embedding": meta.get("embedding")
        })

    for label, nodes in grouped.items():
        query = (
            f"UNWIND $nodes AS node "
            f"MERGE (n:`{label}` {{name: node.name}}) "
            f"SET n:Concept "
            f"SET n.page_numbers = node.page_numbers, "
            f"    n.pdf_url = node.pdf_url, "
            f"    n.embedding = node.embedding"
        )
        tx.run(query, nodes=nodes)

def create_edges(tx, relationships_with_meta):
    """
    relationships_with_meta: list of {"source": s, "relation": rel, "target": t, "page_number": p, "pdf_url": url}
    """
    grouped = {}
    for r in relationships_with_meta:
        rel_type = clean_label(r["relation"]).upper()
        if rel_type not in grouped:
            grouped[rel_type] = []
        grouped[rel_type].append(r)

    for rel_type, edges in grouped.items():
        query = (
            f"UNWIND $edges AS edge "
            f"MATCH (s {{name: edge.source}}) "
            f"MATCH (t {{name: edge.target}}) "
            f"MERGE (s)-[r:`{rel_type}`]->(t) "
            f"SET r.page_number = edge.page_number, "
            f"    r.pdf_url = edge.pdf_url"
        )
        tx.run(query, edges=edges)

# ---------------------------
# MAIN PIPELINE
# ---------------------------

async def process_pdf_to_neo4j(pdf_content, pdf_url, neo4j_config):
    """
    pdf_content: bytes or path
    pdf_url: string for traceability
    neo4j_config: {"uri": ..., "user": ..., "password": ...}
    """
    images = pdf_to_images(pdf_content)
    
    accumulated_entities = []
    entities_metadata = {} # name -> {"type": typ, "page_numbers": set, "pdf_url": url}
    relationships_with_meta = [] # list of dicts

    for i, img in enumerate(images):
        page_num = i + 1
        result = await extract_from_image(img, page_num, accumulated_entities)
        data = result["data"]

        # Track entities for LLM context and metadata
        for e in data.get("entities", []):
            name = e.get("name", "").strip()
            typ = e.get("type", "Unknown").strip()
            if name:
                if name not in entities_metadata:
                    entities_metadata[name] = {"type": typ, "page_numbers": set(), "pdf_url": pdf_url}
                    accumulated_entities.append(e)
                entities_metadata[name]["page_numbers"].add(page_num)

        # Track relationships with metadata
        for r in data.get("relationships", []):
            s = r.get("source", "").strip()
            rel = r.get("relation", "").strip()
            t = r.get("target", "").strip()
            if s and rel and t:
                relationships_with_meta.append({
                    "source": s,
                    "relation": rel,
                    "target": t,
                    "page_number": page_num,
                    "pdf_url": pdf_url
                })
                # Ensure source and target nodes exist in metadata if not explicitly mentioned
                if s not in entities_metadata:
                    entities_metadata[s] = {"type": "Unknown", "page_numbers": {page_num}, "pdf_url": pdf_url}
                    accumulated_entities.append({"name": s, "type": "Unknown"})
                else:
                    entities_metadata[s]["page_numbers"].add(page_num)
                
                if t not in entities_metadata:
                    entities_metadata[t] = {"type": "Unknown", "page_numbers": {page_num}, "pdf_url": pdf_url}
                    accumulated_entities.append({"name": t, "type": "Unknown"})
                else:
                    entities_metadata[t]["page_numbers"].add(page_num)

    # Generate embeddings for all unique entities
    print(f"Generating embeddings for {len(entities_metadata)} entities...")
    for name in entities_metadata:
        entities_metadata[name]["embedding"] = await generate_embedding(name)

    # Upload to Neo4j
    with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"])) as driver:
        with driver.session() as session:
            session.execute_write(setup_vector_index)
            session.execute_write(create_nodes, entities_metadata)
            session.execute_write(create_edges, relationships_with_meta)

    return {
        "status": "success",
        "entities_count": len(entities_metadata),
        "relationships_count": len(relationships_with_meta)
    }

# ---------------------------
# QUERY LOGIC
# ---------------------------

async def query_kg_by_vector(query_text, neo4j_config):
    """
    1. Extract entities from query
    2. Embed entities
    3. Search Neo4j for similar entities
    4. Fetch relationships
    """
    # 1. Extract entities from query using a simple LLM call
    extract_prompt = f"Extract important entities from this search query: '{query_text}'. Return them as a JSON list of strings: ['entity1', 'entity2']"
    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=[extract_prompt]
    )
    extracted_entities = extract_json(response.text)
    if not extracted_entities or not isinstance(extracted_entities, list):
         # Fallback to the whole query if extraction fails
         extracted_entities = [query_text]

    all_results = {"entities": [], "relationships": []}
    seen_nodes = set()
    seen_rels = set()

    with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"])) as driver:
        with driver.session() as session:
            for entity in extracted_entities:
                # 2. Embed
                emb = await generate_embedding(entity)
                
                # 3. Vector Search + Fetch Relationships
                query = """
                CALL db.index.vector.queryNodes('entity_embeddings', 5, $embedding)
                YIELD node, score
                MATCH (node)-[r]-(neighbor)
                RETURN node, r, neighbor, score
                LIMIT 20
                """
                results = session.run(query, embedding=emb)
                
                for record in results:
                    n = record["node"]
                    r = record["r"]
                    m = record["neighbor"]
                    
                    # Add node info
                    if n.element_id not in seen_nodes:
                        all_results["entities"].append({
                            "name": n["name"],
                            "type": list(n.labels)[0] if n.labels else "Unknown",
                            "score": record["score"]
                        })
                        seen_nodes.add(n.element_id)
                    
                    if m.element_id not in seen_nodes:
                        all_results["entities"].append({
                            "name": m["name"],
                            "type": list(m.labels)[0] if m.labels else "Unknown"
                        })
                        seen_nodes.add(m.element_id)

                    # Add relationship info
                    rel_id = r.element_id
                    if rel_id not in seen_rels:
                        all_results["relationships"].append({
                            "source": n["name"] if r.start_node == n else m["name"],
                            "relation": r.type,
                            "target": m["name"] if r.end_node == m else n["name"],
                            "page_number": r.get("page_number"),
                            "pdf_url": r.get("pdf_url")
                        })
                        seen_rels.add(rel_id)

    return all_results
