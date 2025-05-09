#!/bin/bash
# filepath: /home/quanghung20gg/Documents/new/neu_solution/evaluating_cluster/install.sh

set -e  # Exit on error

echo "=== Installing system dependencies ==="
sudo apt-get update && sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential

# Set Python aliases if they don't exist
if [ ! -f /usr/bin/python ]; then
    sudo ln -sf /usr/bin/python3 /usr/bin/python
fi
if [ ! -f /usr/bin/pip ]; then
    sudo ln -sf /usr/bin/pip3 /usr/bin/pip
fi

echo "=== Setting up Python virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing requirements ==="
if [ -f "requirements.txt" ]; then
    pip install --no-cache-dir --ignore-installed -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

echo "=== Installing FastAPI and related dependencies ==="
pip install fastapi uvicorn requests pydantic python-dotenv

echo "=== Cloning LLM repository ==="
if [ ! -d "llm" ]; then
    git clone https://github.com/hung20gg/llm.git
else
    echo "LLM repository already exists, updating..."
    cd llm
    git pull
    cd ..
fi

echo "=== Installing LLM requirements ==="
if [ -f "llm/quick_requirements.txt" ]; then
    pip install -r llm/quick_requirements.txt
else
    echo "Error: LLM requirements not found"
    exit 1
fi

echo "=== Setting up environment variables ==="
cat > .env << EOF
PYTHONPATH=$(pwd)
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_VISIBLE_DEVICES=0
EOF

# Create a run script for convenience
cat > run.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
source .env
cd api && python evaluation_api.py
EOF
chmod +x run.sh

echo "=== Installation complete ==="
echo "To run the evaluation API:"
echo "  ./run.sh"
echo "The API will be available on port 23477"