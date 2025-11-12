import os
import openai
import psycopg2
import numpy as np
from test_db import get_db_connection
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """Generate embedding for query."""
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def retrieve_similar_chunks(query, top_k=3):
    """Retrieve top-k similar chunks for the given query."""
    query_embedding = get_embedding(query)

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT text, 1 - (embedding <=> %s::vector) AS similarity
        FROM medical_vectors
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding, query_embedding, top_k)
    )
    results = cur.fetchall()

    cur.close()
    conn.close()

    return [{"text": r[0], "similarity": float(r[1])} for r in results]
