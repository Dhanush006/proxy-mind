#!/bin/bash
# Quick MCP Server Status Check

echo "🔍 MCP Server Status Check"
echo "=========================="

# Check if Python is available
if command -v python &> /dev/null; then
    echo "✅ Python is available: $(python --version)"
else
    echo "❌ Python not found"
    exit 1
fi

# Check if api module exists
if [ -d "api" ]; then
    echo "✅ api directory found"
    if [ -f "api/main.py" ]; then
        echo "✅ api/main.py exists"
    else
        echo "❌ api/main.py not found"
    fi
    if [ -f "api/__init__.py" ]; then
        echo "✅ api/__init__.py exists"
    else
        echo "⚠️  api/__init__.py missing (may cause import issues)"
    fi
else
    echo "❌ api directory not found"
fi

# Test module import
echo -e "\n🧪 Testing module import..."
python -c "
try:
    import api.main
    print('✅ api.main module imports successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
except Exception as e:
    print(f'❌ Other error: {e}')
"

# Check for running processes
echo -e "\n🔄 Checking for running server processes..."
if pgrep -f "api.main" > /dev/null; then
    echo "✅ Found running server process(es):"
    pgrep -f "api.main" | xargs ps -p
else
    echo "❌ No server processes found"
fi

echo -e "\n✨ Status check complete!"