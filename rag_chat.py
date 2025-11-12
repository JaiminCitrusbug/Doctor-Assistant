import os
import openai
from retriever import retrieve_similar_chunks

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(user_query, history):
    """Generate context-aware answer using chat history and RAG."""
    retrieved_chunks = retrieve_similar_chunks(user_query, top_k=3)
    context_text = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    # Convert Streamlit history to OpenAI chat format
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history]

    system_prompt = {
    "role": "system",
    "content": (
        "You are a reliable medical assistant designed to support doctors in emergencies. "
        "Use both the full conversation history and the provided context to answer accurately, concisely, and safely. "
        "Never guess or invent information—if uncertain, clearly say so.\n\n"

        "When the user’s query partially matches or closely resembles something in the available context, "
        "respond with a short clarification such as: "
        "\"Did you mean [closest match]?\" and wait for confirmation before proceeding. "
        "Once the user confirms, continue the response accordingly using verified context.\n\n"

        "Clarification logic must remain dynamic—detect partial matches for any relevant term, "
        "entity, condition, drug, or medical procedure from the provided context. "
        "If no match exists, reply with: "
        "\"I couldn’t find that in my current data. Can you please rephrase or specify what you meant?\"\n\n"

        "Always maintain a professional, empathetic tone suitable for medical professionals. "
        "Be brief but precise, focusing on actionable, evidence-based information.\n\n"

        f"Context:\n{context_text}"
    ),
}

    messages = [system_prompt] + chat_history + [{"role": "user", "content": user_query}]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
