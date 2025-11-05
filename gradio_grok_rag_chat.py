#!/usr/bin/env python3
import asyncio
import sys
from contextlib import AsyncExitStack
import gradio as gr

from api.groq_rag_with_mcp import GroqRAGWithMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# -----------------------------
# Global session holder
# -----------------------------
session_holder = {}
chat_history = []

async def init_mcp_session():
    """Initialize MCP session and store in session_holder"""
    stack = AsyncExitStack()
    await stack.__aenter__()

    # Use the same Python executable as the running process so subprocess
    # creation works under uvloop/asyncio (avoids FileNotFoundError).
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["api/retrieval_mcpserver.py"],
    )

    stdio_transport = await stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport

    session = await stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()

    session_holder['session'] = session
    session_holder['stack'] = stack
    # Create backend and attach the initialized MCP session/stack so the
    # GroqRAGWithMCP instance can use the already-created ClientSession.
    backend = GroqRAGWithMCP()
    backend.session = session
    backend.stack = stack
    session_holder['backend'] = backend

    print("✅ MCP session initialized.")

# -----------------------------
# Async chat function
# -----------------------------
async def async_chat(user_message):
    # Lazily initialize the MCP session in the same async event loop that
    # Gradio uses. This prevents creating the async stdio client in a
    # different loop (which later causes "Attempted to exit cancel scope in a
    # different task than it was entered in").
    backend = session_holder.get('backend')
    if not backend:
        await init_mcp_session()
        backend = session_holder.get('backend')
        if not backend:
            return chat_history + [{"role": "system", "content": "[Error: MCP session failed to initialize]"}]

    answer = await backend.process_query(user_message)
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

# -----------------------------
# Launch Gradio
# -----------------------------
def launch_chat_interface():
    with gr.Blocks(title="Grok-RAG Chat") as demo:
        gr.Markdown("## 🧠 Grok-RAG Chat Interface")
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="Type your query here...")
        clear = gr.Button("Clear Chat")

        msg.submit(async_chat, inputs=msg, outputs=chatbot)
        clear.click(lambda: [], None, chatbot)

    demo.launch()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Do NOT initialize the MCP session with asyncio.run() here: that
    # would create the async stdio client in a different event loop than
    # the one Gradio runs, leading to cancel-scope / asyncgen shutdown
    # errors. Instead, we lazily initialize inside `async_chat` (above),
    # which runs on Gradio's event loop.
    launch_chat_interface()