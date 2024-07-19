#!/bin/bash

# Create Python virtual environment
echo 'Creating Python virtual environment ".venv" in agents directory'
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies from "requirements.txt" into the virtual environment
echo 'Installing dependencies from "requirements.txt" into virtual environment'
pip install -r requirements.txt

# Run the Streamlit application
echo 'Running the Streamlit application'
streamlit run app.py --server.port 7860