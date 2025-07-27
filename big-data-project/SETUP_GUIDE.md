# Adelaide Traffic Intelligence - Setup Guide

**Created by: Shubharthak Sangharsha ([Portfolio](https://devshubh.me))**

## 🚀 Quick Setup (Local Development)

### Prerequisites
- Python 3.8+ installed
- Git installed
- Web browser

### Step 1: Clone and Setup
```bash
# Clone repository
git clone https://github.com/shubharthaksangharsha/trimester2.git
cd trimester2/big-data-project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install Flask==2.3.3 pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 joblib==1.3.2
```

### Step 2: Run Application
```bash
# Start the Flask application
python app.py

# Open browser and navigate to:
http://localhost:8504
```

## 🌐 AWS Deployment

### Step 1: Server Setup
```bash
# Upload files to AWS EC2
scp -r * ubuntu@your-ec2-ip:/home/ubuntu/trimester2/big-data-project/

# SSH into server
ssh ubuntu@your-ec2-ip
cd /home/ubuntu/trimester2/big-data-project
```

### Step 2: Run Deployment Script
```bash
# Make deployment script executable
chmod +x deploy_aws.sh

# Run deployment (installs everything automatically)
./deploy_aws.sh
```

### Step 3: Manual Virtual Environment Setup (if needed)
```bash
# Create directory structure
mkdir -p /home/ubuntu/trimester2/big-data-project
cd /home/ubuntu/trimester2/big-data-project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install Flask==2.3.3
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install joblib==1.3.2
pip install Werkzeug==2.3.7
pip install Jinja2==3.1.2

# Test the application
python app.py
```

### Step 4: Systemd Service Setup
```bash
# Copy service file
sudo cp ati-bigdata.service /etc/systemd/system/ati-bigdata.service

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ati-bigdata
sudo systemctl start ati-bigdata

# Check service status
sudo systemctl status ati-bigdata
```

### Step 5: Caddy Web Server Setup
```bash
# Install Caddy (if not already installed)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# Copy Caddyfile
sudo cp Caddyfile /etc/caddy/Caddyfile

# Reload Caddy configuration
sudo systemctl reload caddy
```

## 🔧 Configuration Details

### Port Configuration
- **Application Port:** 8504
- **Web Server:** Caddy (ports 80/443)
- **Domain:** ati-bigdata.devshubh.me

### Service Files
- **Systemd Service:** `/etc/systemd/system/ati-bigdata.service`
- **Caddy Config:** `/etc/caddy/Caddyfile`
- **Application Path:** `/home/ubuntu/trimester2/big-data-project`

### Virtual Environment Structure
```
/home/ubuntu/trimester2/big-data-project/
├── venv/                    # Python virtual environment
│   ├── bin/                 # Executables
│   ├── lib/                 # Python packages
│   └── pyvenv.cfg          # Environment config
├── app.py                  # Flask application
├── templates/              # HTML templates
│   └── dashboard.html      # Main dashboard
├── requirements.txt        # Dependencies
├── ati-bigdata.service    # Systemd service file
├── Caddyfile              # Web server config
└── deploy_aws.sh          # Deployment script
```

## 🛠️ Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Check what's using port 8504
sudo lsof -i :8504

# Kill process if needed
sudo kill -9 <PID>
```

**2. Service Not Starting**
```bash
# Check service logs
sudo journalctl -u ati-bigdata -f

# Restart service
sudo systemctl restart ati-bigdata
```

**3. Virtual Environment Issues**
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Permission Issues**
```bash
# Fix ownership
sudo chown -R ubuntu:ubuntu /home/ubuntu/trimester2/big-data-project

# Fix permissions
chmod +x deploy_aws.sh
chmod +x app.py
```

### Service Management Commands
```bash
# Start service
sudo systemctl start ati-bigdata

# Stop service
sudo systemctl stop ati-bigdata

# Restart service
sudo systemctl restart ati-bigdata

# Check status
sudo systemctl status ati-bigdata

# View logs
sudo journalctl -u ati-bigdata -f

# Enable auto-start
sudo systemctl enable ati-bigdata

# Disable auto-start
sudo systemctl disable ati-bigdata
```

### Caddy Management
```bash
# Reload configuration
sudo systemctl reload caddy

# Restart Caddy
sudo systemctl restart caddy

# Check Caddy status
sudo systemctl status caddy

# View Caddy logs
sudo journalctl -u caddy -f
```

## 📊 Features

### ✨ Latest Features (v2.0)
- 🌓 **Light/Dark Theme Toggle**
- 🗺️ **Google Maps Integration** (with fallback)
- 🔮 **Interactive Prediction Modal**
- 📱 **Mobile Responsive Design**
- ⚡ **Real-time Updates**

### 🔧 Technical Stack
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **3D Graphics:** Three.js
- **Charts:** Chart.js
- **Maps:** Google Maps API (with fallback)
- **Deployment:** AWS EC2, Caddy, systemd

## 🌐 Live Demo
- **URL:** [https://ati-bigdata.devshubh.me](https://ati-bigdata.devshubh.me)
- **GitHub:** [Repository Link](https://github.com/shubharthaksangharsha/trimester2/tree/main/big-data-project)

## 🤝 Support
For issues or questions, contact:
- **Portfolio:** [devshubh.me](https://devshubh.me)
- **GitHub:** [shubharthaksangharsha](https://github.com/shubharthaksangharsha)
- **LinkedIn:** [in/shubharthaksangharsha](https://linkedin.com/in/shubharthaksangharsha) 