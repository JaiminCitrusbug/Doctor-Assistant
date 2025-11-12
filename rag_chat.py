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

        "If the user's query partially matches, closely resembles, or appears to be a misspelling of any term in the context "
        "(such as a drug name, disease, procedure, or medical term), dynamically suggest the most relevant match by replying: "
        "\"Did you mean [closest match]?\" and wait for confirmation before proceeding. "
        "Once the user confirms, continue the response accordingly using verified information from the context.\n\n"

        "The clarification logic must handle fuzzy or approximate matches—detect small spelling variations, typos, "
        "or partial similarities across all relevant entities or terms in the context.\n\n"

        "If no sufficiently similar match exists, reply with: "
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
