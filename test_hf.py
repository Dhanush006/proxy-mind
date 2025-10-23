import os
from dotenv import load_dotenv

# Import the necessary LangChain classes
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint

# Load API token from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# --- HuggingFace Model Setup ---
# Using 'gpt2' which is a stable and well-supported text-generation model
# on the free Hugging Face inference API. This should resolve the recurring errors.
llm = None
try:
    if not hf_token:
        raise ValueError("Hugging Face API token not found. Please set HF_TOKEN in your .env file.")

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.7,
        max_new_tokens=250,
        huggingfacehub_api_token=hf_token,
        task="conversatonal"  # Explicitly setting the task to conversational to fix the error.
    )
    
except Exception as e:
    print(f"Failed to initialize HuggingFaceEndpoint. Error: {e}")
    print("Please ensure your Hugging Face API token is correct and has the necessary permissions.")
    print("You can get a token from https://huggingface.co/settings/tokens.")

# --- Define Prompt Template ---
# A simple prompt to test the model's ability to respond to a query.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions in a concise manner."),
        ("human", "{input}"),
    ]
)

# --- Create a simple Chain ---
# This chain just takes the user input, applies the prompt, and sends it to the LLM.
# We'll check if llm is valid before creating the chain.
if llm:
    chain = prompt | llm
else:
    chain = None

# --- Run Inference ---
while True:
    if chain is None:
        print("\nChain could not be initialized. Please fix the LLM endpoint issue and restart.")
        break
        
    query = input("\nEnter your query (or 'exit' to quit): ").strip()
    if query.lower() == 'exit':
        break
    
    # Invoke the chain with the user's query
    try:
        result = chain.invoke({"input": query})
        print(f"\n📌 Answer:\n{result}")
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        print("This may be due to an issue with the Hugging Face endpoint or an invalid query.")
