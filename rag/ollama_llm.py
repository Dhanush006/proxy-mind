from rag.base_llm import BaseLLM
import requests

class OllamaLLM(BaseLLM):
    def __init__(self, model="mistral"):
        self.model = model

    def invoke(self, prompt: str) -> str:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt}
        )
        return res.json()["response"]
