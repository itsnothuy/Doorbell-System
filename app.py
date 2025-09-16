#!/usr/bin/env python3
"""
Cloud deployment entry point for Doorbell Security System
This file serves as the main entry point for cloud platforms like Vercel, Render, etc.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set environment variables for cloud deployment
os.environ['DEVELOPMENT_MODE'] = 'true'
os.environ['PORT'] = os.environ.get('PORT', '8000')

from src.web_interface import create_web_app
from src.doorbell_security import DoorbellSecuritySystem
from config.logging_config import setup_logging # Import centralized logging

# Setup logging early for app.py and define logger immediately
setup_logging(level=logging.DEBUG) # Configure logging for app.py, no file output here
logger = logging.getLogger(__name__) # Define logger at module level after setup_logging

# Global variable to store initialization error message
system_init_error_message = None # Initialize as None

# Initialize the doorbell system
try:
    logger.info("Initializing Doorbell Security System for cloud deployment...")
    
    doorbell_system = DoorbellSecuritySystem()
    doorbell_system.start() # Explicitly start the system
    
    # Create Flask app
    app = create_web_app(doorbell_system)
    
    logger.info("✅ Cloud deployment ready")
    
except Exception as init_exception:
    system_init_error_message = str(init_exception)
    logger.error(f"Failed to initialize system: {system_init_error_message}")
    # Create a minimal app for error display
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return jsonify({
            'status': 'error',
            'message': f'System initialization failed: {system_init_error_message}',
            'note': 'This is a development/testing deployment. Some features may not work without proper hardware.'
        })

# For cloud platforms that expect 'application'
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting web application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
