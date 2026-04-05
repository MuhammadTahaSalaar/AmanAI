#!/bin/bash
# AmanAI Local Setup Script
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Downloading Spacy model..."
python -m spacy download en_core_web_lg
echo "Setup complete! Run 'python -m src.data_processing.etl_pipeline' to start."