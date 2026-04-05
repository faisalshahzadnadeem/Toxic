import cohere
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import COHERE_API_KEY

co = cohere.ClientV2(COHERE_API_KEY)


# -----------------------------
# DOCUMENT MODE
# -----------------------------
def ingest_documents():

    pdf_path = "data/toxicology_study.pdf"

    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    if len(chunks) == 0:
        raise ValueError("No text extracted from PDF")

    # ✅ Cohere v5 embedding fix
    embed_response = co.embed(
        model="embed-english-v3.0",
        texts=chunks,
        input_type="search_document"
    )

    embeddings = embed_response.embeddings.float
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


# -----------------------------
# CONNECTORS MODE
# -----------------------------
def web_connector(query):

    prompt = f"""
You are a medical research assistant.

Provide the latest publicly known toxicology guidance related to:
{query}

If no recent updates exist, say:
'No recent public updates found.'
"""

    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.message.content[0].text


# -----------------------------
# HYBRID RETRIEVAL
# -----------------------------
def hybrid_retrieve(query, index, chunks, top_k=3):

    # ✅ Query embedding (Cohere v5 correct format)
    query_embed_response = co.embed(
        model="embed-english-v3.0",
        texts=[query],
        input_type="search_query"
    )

    query_embedding = query_embed_response.embeddings.float
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, top_k)

    doc_results = [chunks[i] for i in indices[0]]
    live_results = web_connector(query)

    return {
        "documents": doc_results,
        "live": live_results
    }


# -----------------------------
# GENERATION
# -----------------------------
def generate_explainable_response(query, retrieved, safety):

    doc_context = "\n\n".join(retrieved["documents"])
    live_context = retrieved["live"]

    prompt = f"""
You are an explainable toxicology AI assistant.

Patient Case:
{query}

Evidence from toxicology_study.pdf:
{doc_context}

Live Evidence:
{live_context}

Safety Layer Assessment:
Risk Level: {safety['risk_level']}
Reason: {safety['reason']}

Provide:
1. Step-by-step reasoning
2. Risk classification
3. Clear triage recommendation
4. Explicit source justification
5. Short medical disclaimer
"""

    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.message.content[0].text