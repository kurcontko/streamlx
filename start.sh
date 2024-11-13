#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Load environment variables from .env file
load_env() {
    if [ -f .env ]; then
        echo "Loading configuration from .env file..."
        # Read each line from .env, ignore comments and empty lines
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Skip comments and empty lines
            [[ $line =~ ^[[:space:]]*# ]] && continue
            [[ -z "$line" ]] && continue
            
            # Export the variable
            export "${line?}"
        done < .env
    else
        echo "No .env file found, using default values"
    fi
}

# Load environment variables
load_env

# Default values (used if not set in .env)
DEFAULT_MODEL=${MLX_MODEL:-"mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"}
PORT_MLX=${MLX_PORT:-8080}
PORT_STREAMLIT=${STREAMLIT_PORT:-7860}
VENV_PATH=${VENV_PATH:-.venv}
LOG_FILE=${LOG_FILE:-server.log}
LOG_LEVEL=${LOG_LEVEL:-DEBUG}

# Get the absolute path to the project root directory
PROJECT_ROOT=$(pwd)

# Export PYTHONPATH to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Function to display usage
usage() {
    echo "Usage: $0 [-m MODEL]"
    echo ""
    echo "Options:"
    echo " -m, --model    Specify the model to use (default: $DEFAULT_MODEL)"
    echo " -h, --help     Display this help message"
    echo ""
    echo "Environment Variables (can be set in .env file):"
    echo " MLX_MODEL         Model to use (default: $DEFAULT_MODEL)"
    echo " MLX_PORT         Port for MLX server (default: $PORT_MLX)"
    echo " STREAMLIT_PORT   Port for Streamlit app (default: $PORT_STREAMLIT)"
    echo " VENV_PATH       Path to virtual environment (default: $VENV_PATH)"
    echo " LOG_FILE        Log file path (default: $LOG_FILE)"
    echo " LOG_LEVEL       Logging level (default: $LOG_LEVEL)"
    exit 1
}

# Parse command-line arguments
MODEL="$DEFAULT_MODEL"

# Use getopt for better parsing (supports long options)
PARSED_ARGS=$(getopt -o m:h --long model:,help -n "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    usage
fi

eval set -- "$PARSED_ARGS"
while true; do
    case "$1" in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unexpected option: $1"
            usage
            ;;
    esac
done

# Function to handle cleanup on exit
cleanup() {
    echo "Terminating the mlx_lm.server..."
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null
        echo "mlx_lm.server with PID $SERVER_PID terminated."
    fi
    echo "Cleanup complete."
}

# Trap EXIT and INT signals to run the cleanup function
trap cleanup EXIT INT

# Create Python virtual environment
echo "Creating Python virtual environment \"$VENV_PATH\" in root directory..."
python3 -m venv "$VENV_PATH"

# Activate the virtual environment
echo "Activating the virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from "requirements.txt" into the virtual environment
echo "Installing dependencies from \"requirements.txt\" into virtual environment..."
pip install -r requirements.txt

# Start the mlx_lm server in the background with the specified model
echo "Starting the mlx_lm.server with model '$MODEL'..."
mlx_lm.server --model "$MODEL" --log-level "$LOG_LEVEL" --port "$PORT_MLX" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "mlx_lm.server started with PID $SERVER_PID."

# Optional: Wait for the server to be ready
echo "Waiting for the mlx_lm.server to be ready on port $PORT_MLX..."
while ! nc -z localhost "$PORT_MLX"; do
    sleep 1
done
echo "mlx_lm.server is ready."

# Run the Streamlit application
echo "Running the Streamlit application..."
PYTHONPATH="${PROJECT_ROOT}" streamlit run src/main.py --server.port "$PORT_STREAMLIT"