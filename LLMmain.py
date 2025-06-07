import os
import time
from typing import List

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Pinecone setup
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
spec = ServerlessSpec(cloud="aws", region="us-west-2")

# Ensure index exists
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")
existing = [idx["name"] for idx in pc.list_indexes()]
if pinecone_index_name not in existing:
    pc.create_index(
        name=pinecone_index_name,
        dimension=768,
        metric="cosine",
        spec=spec,
        deletion_protection=False
    )
    while not pc.describe_index(pinecone_index_name).status.get("ready", False):
        time.sleep(1)
index = pc.Index(pinecone_index_name)
time.sleep(1)

# Encoder for queries
from semantic_router.encoders import HuggingFaceEncoder
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

# Q&A history store
qa_history: List[dict] = []

# Groq client for LLM
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from pydantic import BaseModel

def get_docs(query: str, top_k: int = 5) -> List[dict]:
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    return [m["metadata"] for m in matches] if matches else []


def generate_answer(query: str, docs: List[dict], user_intro: str) -> str:
    # Build document context
    context = "\n---\n".join(doc.get("text", "") for doc in docs)
    # Build chat history
    history_texts = [f"Q: {h['question']}\nA: {h['answer']}" for h in qa_history]
    history = "\n\n".join(history_texts)
    
    system_message = (
        "You are a compassionate and helpful medical chatbot designed for mothers."
        "do not mention you about user EPDS Scale  or any other specific medical terms unless asked."
        "Please recall that some past information may be sensitive, so avoid repeating such content unless explicitly asked.\n\n"
        f"USER CONTEXT:\n{user_intro}\n\n"
        f"CHAT HISTORY:\n{history}\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    try:
        resp = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    # Record Q&A
    qa_history.append({"question": query, "answer": answer})
    return answer + "\n\n"

# Optional CLI chatbot loop
if __name__ == "__main__":
    def chatbot():
        print("Welcome to the MommyCare Medical Chatbot! Type 'bye' to exit.")
        while True:
            q = input("You: ").strip()
            if q.lower() in ["bye", "exit"]:
                print("Chatbot: Take care!")
                break
            docs = get_docs(q, top_k=5)
            ans = generate_answer(q, docs, "")
            print(f"Chatbot: {ans}")

    chatbot()