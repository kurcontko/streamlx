# MLX Chat

A simple chat interface for MLX served LLMs built with Streamlit.

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/kurcontko/mlx-chat.git
cd mlx-chat
```

2. Run:
```bash
./start.sh
```

Or manually:
```bash
# Terminal 1: Start MLX server
python3 -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt

# Pick the most suitable model based on your VRAM:
# - mlx-community/Qwen2.5-Coder-32B-Instruct-4bit 
# - mlx-community/Mistral-7B-Instruct-v0.3-8bit

mlx_lm.server --model MODEL_NAME --port 8080

# Terminal 2: Start Streamlit app
streamlit run src/main.py
```

## Features

- Simple UI
- No external dependencies
- Mlx engine

## Configuration

Create `.env` file (optional):
```env
MLX_MODEL=mlx-community/Qwen2.5-Coder-32B-Instruct-4bit
MLX_PORT=8080
STREAMLIT_PORT=7860
```

## Requirements

- Apple M chip
- Python 3.11+
- MLX
- Streamlit
- See `requirements.txt`