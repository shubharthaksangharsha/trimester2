#!/bin/bash

# AI Text Detection Deployment Script for Oracle Server
# Run this script as ubuntu user

set -e

echo "🚀 AI Text Detection Deployment Script"
echo "======================================"

# Variables
APP_DIR="/home/ubuntu/ai-text-detection"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="ai-text-detection"

# Create application directory
echo "📁 Creating application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Create Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install Flask==2.3.3 Werkzeug==2.3.7 Jinja2==3.1.2 numpy==1.24.3 gunicorn==21.2.0
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Create systemd service file
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << 'EOF'
[Unit]
Description=AI Text Detection Flask Application
After=network.target

[Service]
Type=exec
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/ai-text-detection
Environment=PATH=/home/ubuntu/ai-text-detection/venv/bin
Environment=FLASK_ENV=production
Environment=PYTHONPATH=/home/ubuntu/ai-text-detection
ExecStart=/home/ubuntu/ai-text-detection/venv/bin/gunicorn --bind 127.0.0.1:5005 --workers 2 --timeout 300 app:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Update Caddyfile
echo "🌐 Updating Caddyfile..."
sudo tee -a /etc/caddy/Caddyfile > /dev/null << 'EOF'

mlt-a3.devshubh.me {
    reverse_proxy localhost:5005

    # Enable gzip compression
    encode gzip

    # Security headers
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        X-XSS-Protection "1; mode=block"
        Referrer-Policy "strict-origin-when-cross-origin"
    }

    # CORS for API endpoints
    @api path /api/*
    handle @api {
        header Access-Control-Allow-Origin *
        header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
        header Access-Control-Allow-Headers "Content-Type, Authorization"
        reverse_proxy localhost:5005
    }

    # Handle preflight requests
    @options method OPTIONS
    handle @options {
        header Access-Control-Allow-Origin *
        header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
        header Access-Control-Allow-Headers "Content-Type, Authorization"
        respond 200
    }
}
EOF

# Reload systemd and start service
echo "🔄 Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl reload caddy

echo "✅ Deployment setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Upload your app files to $APP_DIR"
echo "2. Upload your models to $APP_DIR/models-ml/"
echo "3. Start the service: sudo systemctl start $SERVICE_NAME"
echo "4. Check status: sudo systemctl status $SERVICE_NAME"
echo "5. View logs: sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "🌐 Your app will be available at: https://mlt-a3.devshubh.me/"
