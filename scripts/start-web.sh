#!/bin/bash
echo "🌐 Starting Doorbell Security Web Interface..."
source venv/bin/activate
export DEVELOPMENT_MODE=true
python app.py
