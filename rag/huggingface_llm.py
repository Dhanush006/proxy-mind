from huggingface_hub import InferenceClient
from rag.base_llm import BaseLLM
import os
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        token = os.getenv("HF_TOKEN")
        assert token, "HF_TOKEN missing in environment"
        self.client = InferenceClient(model=model_id, token=token)

    def invoke(self, prompt: str) -> str:
        response = self.client.text_generation(prompt, max_new_tokens=300)
        return response
