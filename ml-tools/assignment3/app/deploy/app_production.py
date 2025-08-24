#!/usr/bin/env python3
"""
AI Text Detection Flask Application - Production Version
Optimized for Oracle server deployment
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/ai-text-detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'production_ai_text_detection_2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Force CPU usage in production
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Your existing model configurations and classes go here
# (Copy from the main app.py file)

if __name__ == '__main__':
    # Production server should not use debug mode
    app.run(host='127.0.0.1', port=5005, debug=False)
else:
    # When run with gunicorn
    logger.info("AI Text Detection app starting in production mode")
