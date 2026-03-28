import io
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from llm import get_client
from helpers.df_helpers import documents2Dataframe, df2Graph, graph2Df
from helpers.kg_pipeline import process_pdf_to_neo4j, query_kg_by_vector
import os

app = FastAPI(title="Knowledge Graph Extractor")


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    client: str = Query(default="ollama"),
    model: str = Query(default="mistral-openorca:latest"),
):
    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    text = content.decode("utf-8", errors="replace")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.create_documents([text])

    # Build dataframe from chunks
    df = documents2Dataframe(chunks)

    # Select LLM client (raises ValueError → 422 for unknown names)
    try:
        llm_client = get_client(client)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Run extraction pipeline
    nodes_list = df2Graph(df, model=model, llm_client=llm_client)
    graph_df = graph2Df(nodes_list)

    # Ensure required columns exist even when output is empty
    for col in ("node_1", "node_2", "edge", "chunk_id"):
        if col not in graph_df.columns:
            graph_df[col] = None

    csv_bytes = graph_df[["node_1", "node_2", "edge", "chunk_id"]].to_csv(index=False).encode("utf-8")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=graph.csv"},
    )


async def background_process_pdf(pdf_url: str, neo4j_config: dict):
    """
    Background task to download and process a PDF.
    """
    import httpx
    import tempfile
    import os
    async with httpx.AsyncClient() as client:
        try:
            print(f"Background: Downloading PDF from {pdf_url}...")
            response = await client.get(pdf_url, timeout=60.0)
            response.raise_for_status()
            pdf_content = response.content
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name

            try:
                print(f"Background: Processing PDF from {tmp_path}...")
                await process_pdf_to_neo4j(tmp_path, pdf_url, neo4j_config)
                print(f"Background: Successfully ingested {pdf_url} into Neo4j.")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as e:
            print(f"Background Error processing {pdf_url}: {str(e)}")


@app.post("/extract-pdf-to-neo4j")
async def extract_pdf_to_neo4j(
    background_tasks: BackgroundTasks,
    pdf_url: str = Query(..., description="URL to the PDF file"),
    neo4j_uri: str = Query(default=None, description="Neo4j URI (defaults to NEO4J_URI env)"),
    neo4j_user: str = Query(default=None, description="Neo4j User (defaults to NEO4J_USER env)"),
    neo4j_password: str = Query(default=None, description="Neo4j Password (defaults to NEO4J_PASSWORD env)"),
):
    """
    Triggers the PDF ingestion pipeline in the background.
    Downloads a PDF from a URL, extracts entities and relationships using Gemini,
    and uploads them to Neo4j.
    """
    # Use environment variables if not provided
    uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    pwd = neo4j_password or os.environ.get("NEO4J_PASSWORD", "password")

    neo4j_config = {
        "uri": uri,
        "user": user,
        "password": pwd
    }

    # Add to background tasks
    background_tasks.add_task(background_process_pdf, pdf_url, neo4j_config)

    return {
        "status": "ingestion_started",
        "message": "The PDF ingestion pipeline has been triggered in the background.",
        "pdf_url": pdf_url
    }


@app.post("/query-kg")
async def query_kg(
    query: str = Query(..., description="The natural language query"),
    neo4j_uri: str = Query(default=None),
    neo4j_user: str = Query(default=None),
    neo4j_password: str = Query(default=None),
):
    """
    Search the Knowledge Graph using vector similarity.
    Extracts entities from the query, finds similar entities in the graph,
    and returns their relationships.
    """
    uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    pwd = neo4j_password or os.environ.get("NEO4J_PASSWORD", "password")

    try:
        neo4j_config = {"uri": uri, "user": user, "password": pwd}
        results = await query_kg_by_vector(query, neo4j_config)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")
