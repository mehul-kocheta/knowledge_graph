import streamlit as st
import json
import re
import pandas as pd
import networkx as nx
from pyvis.network import Network


# --------------------------------
# Extract JSON from code blocks
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


# --------------------------------
# Merge KG
# --------------------------------

def merge_kg(data):

    entities = {}
    relationships = set()

    for page in data:

        raw = page["data"].get("raw_output")

        parsed = extract_json(raw)

        if not parsed:
            continue

        # entities
        for e in parsed.get("entities", []):

            name = e["name"].strip()
            typ = e["type"].strip()

            entities[name] = typ

        # relationships
        for r in parsed.get("relationships", []):

            s = r["source"].strip()
            rel = r["relation"].strip()
            t = r["target"].strip()

            relationships.add((s, rel, t))

    return entities, relationships


# --------------------------------
# Graph Visualization
# --------------------------------

def render_graph(entities, relationships):

    G = nx.DiGraph()

    # Add explicit entities first
    for name, typ in entities.items():
        G.add_node(name, title=typ)

    # Add edges, and add nodes if they don't exist yet with a default title
    for s, r, t in relationships:
        if not G.has_node(s):
            G.add_node(s, title="Unknown")
        if not G.has_node(t):
            G.add_node(t, title="Unknown")
            
        G.add_edge(s, t, label=r)

    net = Network(height="700px", width="100%", directed=True)

    for node, data in G.nodes(data=True):
        # We use .get("title", "Unknown") just to be extra safe
        net.add_node(node, label=node, title=data.get("title", "Unknown"))

    for s, t, data in G.edges(data=True):
        net.add_edge(s, t, label=data["label"])

    net.save_graph("graph.html")

    with open("graph.html", "r") as f:
        html = f.read()

    st.components.v1.html(html, height=700)


# --------------------------------
# Streamlit UI
# --------------------------------

st.set_page_config(page_title="Research Paper Knowledge Graph", layout="wide")

st.title("📚 Research Paper Knowledge Graph Viewer")

uploaded = st.file_uploader("Upload KG JSON", type=["json"])


if uploaded:

    data = json.load(uploaded)

    entities, relationships = merge_kg(data)

    st.success("Knowledge graph successfully parsed")

    col1, col2 = st.columns(2)

    col1.metric("Entities", len(entities))
    col2.metric("Relationships", len(relationships))

    st.divider()

    # Graph
    st.subheader("Interactive Knowledge Graph")

    render_graph(entities, relationships)

    st.divider()

    # Entity table
    st.subheader("Entities")

    entity_df = pd.DataFrame(
        [{"Entity": k, "Type": v} for k, v in entities.items()]
    )

    st.dataframe(entity_df, use_container_width=True)

    st.divider()

    # Relationship table
    st.subheader("Relationships")

    rel_df = pd.DataFrame(
        relationships,
        columns=["Source", "Relation", "Target"]
    )

    st.dataframe(rel_df, use_container_width=True)

    st.download_button(
        "Download Clean KG JSON",
        data=json.dumps({
            "entities": entity_df.to_dict("records"),
            "relationships": rel_df.to_dict("records")
        }, indent=2),
        file_name="clean_kg.json"
    )