from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List

class QueryEngine:
    def __init__(self, retriever: Runnable, llm):
        self.retriever = retriever
        self.llm = llm

    def format_prompt(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"""Answer the question based on the context below.

Context:
{context}

Question: {query}"""

    def run(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        prompt = self.format_prompt(query, docs)
        return self.llm.invoke(prompt)
