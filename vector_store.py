# vector_store.py
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector
from test_db import get_db_connection  # keep your existing connection helper

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

INPUT_JSON = os.getenv("INPUT_JSON", "wockhardt_products.json")  # your file from the last message

def build_text_for_embedding(doc: dict) -> str:
    """
    Construct a single text blob from the Wockhardt product record.
    Uses available fields safely (some keys may not exist).
    """
    fields_in_order = [
        "product_name", "brand_name", "therapeutic_class", "strength",
        "dosage_form", "pack_size", "composition", "indication_summary",
        "extracted_text"
    ]
    parts = []
    for k in fields_in_order:
        v = doc.get(k)
        if v:
            parts.append(f"{k.replace('_',' ').title()}: {v}")
    # Fallback if nothing present
    if not parts:
        parts.append(doc.get("id", ""))
    text_blob = "\n".join(parts).strip()
    # keep it within a reasonable size for embeddings
    return text_blob[:6000]

def create_embedding(text: str):
    """Generate an embedding vector for a given text."""
    if not text:
        text = " "  # avoid empty input to embeddings API
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def store_embeddings():
    # connect DB
    conn = get_db_connection()
    register_vector(conn)  # ensure pgvector adapter is registered
    cur = conn.cursor()

    # load JSON
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)

    for i, doc in enumerate(docs):
        uid = doc.get("id", f"vec_{i}")
        title = doc.get("product_name") or doc.get("brand_name") or uid
        source_id = doc.get("source_url")
        text = build_text_for_embedding(doc)
        embedding = create_embedding(text)

        cur.execute("""
            INSERT INTO public.medical_vectors (id, title, source_id, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (uid, title, source_id, 0, text, embedding))

    conn.commit()
    cur.close()
    conn.close()
    print("✅ All embeddings stored successfully!")

if __name__ == "__main__":
    store_embeddings()
