# test_mcp_client.py
import subprocess
import json
import time
import sys

# Test print to confirm script start
print("Starting test_mcp_client.py")

# Start the MCP server as a subprocess
print("Launching MCP server subprocess...")
try:
    server_process = subprocess.Popen(
        ["uv", "run", "api/main.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
except Exception as e:
    print(f"Error starting subprocess: {e}")
    sys.exit(1)

# Test print to confirm subprocess started
print("Subprocess started with PID:", server_process.pid)

# Wait for the server to initialize
print("Waiting 2 seconds for server to initialize...")
time.sleep(2)

# Check if the server is still running
if server_process.poll() is not None:
    print("Error: Server process terminated early. Exit code:", server_process.poll())
    stderr_output = server_process.stderr.read()
    print("STDERR:", stderr_output)
    sys.exit(1)
else:
    print("Server process is running.")

# Test the retrieval tool
retrieve_message = {
    "method": "tool_call",
    "tool": "retrieve_relevant_personal_projects_and_data_tool",
    "params": {
        "query": "AI projects",
        "db_type": "semantic",
        "top_k": 3
    }
}

print("Sending retrieval tool request:", json.dumps(retrieve_message))
server_process.stdin.write(json.dumps(retrieve_message) + "\n")
server_process.stdin.flush()

# Try to read the response with a timeout
try:
    response = server_process.stdout.readline().strip()
    print("Retrieval tool response:", response)
except Exception as e:
    print(f"Error reading response: {e}")

# Capture and print any STDERR output (e.g., tool prints or errors)
stderr_output = server_process.stderr.read().strip()
if stderr_output:
    print("Retrieval tool STDERR:", stderr_output)

# Terminate the server
print("Terminating MCP server subprocess...")
server_process.terminate()

# Wait for the process to exit and capture remaining output
try:
    stdout, stderr = server_process.communicate(timeout=5)
    if stdout:
        print("Remaining STDOUT:", stdout.strip())
    if stderr:
        print("Remaining STDERR:", stderr.strip())
except subprocess.TimeoutExpired:
    print("Warning: Subprocess did not terminate gracefully.")
    server_process.kill()

print("Test script completed.")