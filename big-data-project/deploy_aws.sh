#!/bin/bash

# Adelaide Traffic Intelligence - AWS Deployment Script
# Author: Shubharthak Sangharsha
# URL: https://ati-bigdata.devshubh.me

echo "🚀 Deploying Adelaide Traffic Intelligence Dashboard to AWS..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and pip if not present
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3 python3-pip python3-venv

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p /home/ubuntu/trimester2/big-data-project
sudo chown ubuntu:ubuntu /home/ubuntu/trimester2/big-data-project
cd /home/ubuntu/trimester2/big-data-project

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install Flask==2.3.3
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install joblib==1.3.2
pip install Werkzeug==2.3.7
pip install Jinja2==3.1.2

# Create systemd service for auto-start
echo "⚙️ Creating systemd service..."
sudo cp ati-bigdata.service /etc/systemd/system/ati-bigdata.service

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable ati-bigdata
sudo systemctl start ati-bigdata

# Install Caddy if not present
echo "🌐 Setting up Caddy web server..."
if ! command -v caddy &> /dev/null; then
    sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
    sudo apt update
    sudo apt install caddy
fi

# Copy Caddyfile
echo "📝 Configuring Caddy..."
sudo cp Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy

# Create log directory
sudo mkdir -p /var/log/caddy
sudo chown caddy:caddy /var/log/caddy

# Setup firewall
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8504/tcp
sudo ufw --force enable

echo "✅ Deployment completed!"
echo ""
echo "🌟 Your Adelaide Traffic Intelligence Dashboard is now available at:"
echo "   https://ati-bigdata.devshubh.me"
echo ""
echo "📊 Service Status:"
echo "   App Service: $(sudo systemctl is-active ati-bigdata)"
echo "   Caddy Service: $(sudo systemctl is-active caddy)"
echo ""
echo "🔍 Useful Commands:"
echo "   Check app logs: sudo journalctl -u ati-bigdata -f"
echo "   Check Caddy logs: sudo journalctl -u caddy -f"
echo "   Restart app: sudo systemctl restart ati-bigdata"
echo "   Restart Caddy: sudo systemctl restart caddy"
echo ""
echo "🎓 Portfolio: https://devshubh.me"
echo "💻 GitHub: https://github.com/shubharthaksangharsha/trimester2/"
echo "💼 LinkedIn: https://linkedin.com/in/shubharthaksangharsha" 