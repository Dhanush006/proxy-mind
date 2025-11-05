#!/usr/bin/env python3
"""
grok_rag_with_mcp.py

Refactored as a Python module / class for reuse.

Features:
- MCP session initialization
- Query gating to decide whether to call retrieval tool
- process_query method with integrated LLM gating and MCP tool call
- CLI test for standalone execution
"""

import asyncio
import json
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import requests
import os

# -----------------------------
# Groq API configuration
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
HTTP_TIMEOUT = 10

# -----------------------------
# GrokRAGWithMCP class
# -----------------------------
class GroqRAGWithMCP:
    """Encapsulates MCP session, gating, and query processing for Grok-RAG."""

    def __init__(self, mcp_server_path="/Users/dhanushdshekar/Documents/Academia/Projects/proxy-mind/api/retrieval_mcpserver.py"):
        self.mcp_server_path = mcp_server_path
        self.session = None
        self.stack = None

    # -------------------------
    # MCP session management
    # -------------------------
    async def init_mcp_session(self):
        """Initialize MCP session asynchronously."""
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        server_params = StdioServerParameters(
            command="python",
            args=[self.mcp_server_path]
        )
        stdio_transport = await self.stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()
        print("✅ MCP session initialized.")

    async def close(self):
        """Close MCP session and exit stack."""
        if self.stack:
            await self.stack.__aexit__(None, None, None)
            print("✅ MCP session closed.")

    # -------------------------
    # Gating: decide which tool to use
    # -------------------------
    def _ask_llm_which_tool(self, query: str, tools: list) -> dict:
        """Ask Groq LLM which tool to use (or none). Returns decision dict."""
        tools_text = "\n".join([f"- {t.name}: {t.description}" for t in tools])
        system = "You are a meta-reasoner that decides whether to use external tools to answer a user question. Respond with JSON only."
        prompt = f"""
You have access to the following tools and their brief descriptions:
{tools_text}

User query:
\"\"\"{query}\"\"\"

Rules:
1) If the query asks about the developer's personal projects, code, architecture, or documentation, prefer using the retrieval tool.
2) If a tool can help, return JSON:
   {{ "use_tool": true, "tool_name": "<tool>", "mode": "<semantic|precise|auto>" }}
3) If no tool is needed, return: {{ "use_tool": false }}

Return only valid JSON.
"""
        try:
            raw = self._query_groq(system, prompt)
            return json.loads(raw.strip())
        except Exception as e:
            # fallback heuristic
            if any(k in query.lower() for k in ["project","proxymind","code","architecture","stacktrace","error",".py"]):
                return {"use_tool": True, "tool_name": "retrieve_context", "mode": "auto"}
            return {"use_tool": False}

    # -------------------------
    # Groq API call
    # -------------------------
    def _query_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Call Groq API with system + user prompt and return assistant text."""
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 768,
            "temperature": 0.0
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"GROQ API error {resp.status_code}: {resp.text}")

    # -------------------------
    # Query processing
    # -------------------------
    async def process_query(self, query: str) -> str:
        """Process a user query, using gating + MCP tool if needed."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")

        # Step 1: discover available tools
        tool_list_response = await self.session.list_tools()
        tools = getattr(tool_list_response, "tools", [])

        # Step 2: ask LLM which tool to use
        decision = self._ask_llm_which_tool(query, tools)

        # Step 3: call tool if needed
        context = ""
        if decision.get("use_tool") and "tool_name" in decision:
            tool_name = decision["tool_name"]
            mode = decision.get("mode", "auto")
            result = await self.session.call_tool(tool_name, {"query": query, "retrieval_mode": mode, "top_k": 5})

            # extract text from CallToolResult
            if hasattr(result, "structuredContent") and result.structuredContent:
                try:
                    data = json.loads(result.structuredContent.get("result", "{}"))
                    context = "\n\n".join(data.get("results", []))
                except Exception:
                    context = str(result.structuredContent)
            elif hasattr(result, "content"):
                context = "\n\n".join(getattr(r, "text", str(r)) for r in result.content)
            else:
                context = str(result)

        # Step 4: final prompt to Groq
        final_prompt = f"Context:\n{context}\n\nQuestion: {query}" if context else f"Question: {query}"
        try:
            return self._query_groq("You are a helpful assistant.", final_prompt)
        except Exception as e:
            return f"Groq failed: {e}"

# -----------------------------
# CLI test for standalone run
# -----------------------------
async def main_cli():
    grok = GroqRAGWithMCP()
    await grok.init_mcp_session()
    try:
        print("\n=== Grok-RAG CLI Test ===")
        while True:
            q = input("Query: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            ans = await grok.process_query(q)
            print("\n=== Answer ===")
            print(ans)
            print("==============\n")
    finally:
        await grok.close()

if __name__ == "__main__":
    asyncio.run(main_cli())