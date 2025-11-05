import os
import requests
from dotenv import load_dotenv
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores.faiss import FAISS
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Embedding & Vector DBs ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = "vector_dbs/db_semantic"
FAISS_PATH = "vector_dbs/db_precise"

chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
faiss_db = FAISS.load_local(FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

# --- Local LLM for fallback ---
local_llm = Ollama(model="phi3:mini", num_predict=1000, temperature=0.7)

# --- GROQ API call ---
def query_groq(prompt: str, model="llama-3.1-8b-instant"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"GROQ API error {resp.status_code}: {resp.text}")

# --- Context gating ---
def needs_context(query: str) -> bool:
    keywords = ["project", "code", "architecture", "vector", "database", "ProxyMind", "retriever"]
    return any(k.lower() in query.lower() for k in keywords)

# --- DB selection ---
def choose_db(query: str):
    precise_keywords = ["exact", "file", "error", "stacktrace", "implementation"]
    if any(k in query.lower() for k in precise_keywords):
        print("🔍 Using FAISS (precise)")
        return faiss_db
    else:
        print("💡 Using Chroma (semantic)")
        return chroma_db

# --- RAG orchestration ---
def run_rag(query: str):
    if needs_context(query):
        db = choose_db(query)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10}) if db == chroma_db else db.as_retriever(search_kwargs={"k": 10})
        # Build LangChain prompt template
        qa_chain = create_stuff_documents_chain(
            local_llm, 
            ChatPromptTemplate.from_messages([
                ("system", "Use the given context to answer. If unknown, say you don't know. Context: {context}"),
                ("human", "{input}")
            ])
        )
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        # Try GROQ first with context
        try:
            docs = db.similarity_search(query, k=10)  # <-- fixed retrieval
            context = "\n".join(d.page_content for d in docs)
            full_prompt = f"Context: {context}\n\nQuestion: {query}"
            return query_groq(full_prompt)
        except Exception as e:
            print(f"⚠️ GROQ failed: {e}. Falling back to local LLM with retrieval.")
            result = rag_chain.invoke({"input": query})
            return result["answer"]
    else:
        # No context needed
        try:
            return query_groq(query)
        except Exception as e:
            print(f"⚠️ GROQ failed: {e}. Falling back to local LLM.")
            return local_llm.invoke(query)

# --- Interactive loop ---
if __name__ == "__main__":
    while True:
        query = input("Query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        try:
            answer = run_rag(query)
            print("\n📌 Answer:\n", answer)
        except Exception as e:
            print("❌ Fatal error:", e)