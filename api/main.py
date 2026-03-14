import io
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from llm import get_client
from helpers.df_helpers import documents2Dataframe, df2Graph, graph2Df

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
