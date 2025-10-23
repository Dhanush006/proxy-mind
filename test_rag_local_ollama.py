import os
from dotenv import load_dotenv

# Import the necessary LangChain classes
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# Import the Ollama class for local models
from langchain_community.llms import Ollama

# Import the components for your original retriever setup
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load API token from .env (This is no longer needed for the LLM, but kept for clarity)
load_dotenv()
# The Hugging Face token is not required for the Ollama model.
hf_token = os.getenv("HF_TOKEN")

# --- Define Embedding ---
# The embedding model remains the same from your original script.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load Vector DBs ---
# These paths are from your original script.
CHROMA_PATH = "vector_dbs/db_semantic"
FAISS_PATH = "vector_dbs/db_precise"

# Load the vector stores.
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
faiss_db = FAISS.load_local(FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

# Choose DB type as in your original script
db_type = input("Choose DB [semantic | precise]: ").strip().lower()
retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 10}) if db_type == "semantic" else faiss_db.as_retriever(search_kwargs={"k": 10})

# --- Ollama Model Setup ---
# Use the Ollama class to connect to your local model.
# The 'model' parameter should match the model you downloaded (e.g., "phi3", "llama3").
llm = Ollama(model="phi3", num_predict=1000, temperature=0.7)

# --- Define the new LangChain Prompt Template ---
# This is the prompt template you provided in our previous conversation.
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --- Create the new LangChain RAG Chains ---
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Run Inference ---
while True:
    query = input("\nEnter your query (or 'exit' to quit): ").strip()
    if query.lower() == 'exit':
        break
    
    # We now invoke the 'rag_chain' and access the 'answer' key.
    try:
        result = rag_chain.invoke({"input": query})
        print(f"\n📌 Answer:\n{result['answer']}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check if the Ollama server is running and the model is loaded.")
