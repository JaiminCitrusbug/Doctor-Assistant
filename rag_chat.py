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

        "When the user’s query partially matches, closely resembles, or appears to be a misspelling of any term in the context "
        "(such as a drug, disease, condition, or procedure), respond with a clarification like: "
        "\"Did you mean [closest match]?\" and pause for confirmation.\n\n"

        "If the user confirms (e.g., says 'Yes', 'Exactly', 'That’s right', or similar), "
        "immediately proceed to retrieve and respond with the information related to the confirmed term from the context, "
        "without asking again or revalidating. The assistant must remember the last suggested term and use it if the user confirms it.\n\n"

        "The clarification logic must handle fuzzy or approximate matches—detect small spelling errors, phonetic variations, "
        "or partial similarities in any relevant term found in the provided context. "
        "When generating the clarification, prefer the single best fuzzy match to avoid multiple-choice confusion.\n\n"

        "If no meaningful match exists, reply with: "
        "\"I couldn’t find that in my current data. Can you please rephrase or specify what you meant?\"\n\n"

        "Ensure that the assistant uses conversation history to maintain context: "
        "- If the user confirms a suggestion, remember it. "
        "- If the user asks a follow-up like 'What did I ask?' or 'Continue', recall the last clarified or confirmed term. "
        "- If the user refers to 'it', 'that', or similar pronouns, resolve them to the most recent confirmed entity.\n\n"

        "Maintain a professional and empathetic tone suitable for medical professionals. "
        "Be concise but precise, focusing on actionable, evidence-based information.\n\n"

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
