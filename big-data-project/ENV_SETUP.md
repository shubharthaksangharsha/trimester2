# 🔐 Environment Variables Setup Guide

## 📝 Create Your .env File

Since `.env` files are automatically ignored for security, you need to create it manually:

### **Windows (PowerShell):**
```powershell
# Navigate to your project directory
cd C:\Users\shubh\OneDrive\Desktop\trimester2\big-data-project

# Create .env file
@"
GOOGLE_MAPS_API_KEY=AIzaSyA--sX1QWCo1s_YBokc57ZkWq7m2aHWiOU
FLASK_ENV=development
FLASK_DEBUG=true
FLASK_HOST=0.0.0.0
FLASK_PORT=8504
SECRET_KEY=adelaide_traffic_intelligence_secret_key_2024
"@ | Out-File -FilePath .env -Encoding UTF8
```

### **Alternative: Manual Creation**
1. **Right-click** in your project folder
2. **New → Text Document**
3. **Rename** from `New Text Document.txt` to `.env` (remove the .txt extension)
4. **Open** the `.env` file in notepad
5. **Copy and paste** this content:

```
GOOGLE_MAPS_API_KEY=AIzaSyA--sX1QWCo1s_YBokc57ZkWq7m2aHWiOU
FLASK_ENV=development
FLASK_DEBUG=true
FLASK_HOST=0.0.0.0
FLASK_PORT=8504
SECRET_KEY=adelaide_traffic_intelligence_secret_key_2024
```

### **Linux/Mac:**
```bash
cat > .env << 'EOF'
GOOGLE_MAPS_API_KEY=AIzaSyA--sX1QWCo1s_YBokc57ZkWq7m2aHWiOU
FLASK_ENV=development
FLASK_DEBUG=true
FLASK_HOST=0.0.0.0
FLASK_PORT=8504
SECRET_KEY=adelaide_traffic_intelligence_secret_key_2024
EOF
```

## ✅ Verify Setup

After creating the `.env` file, restart your Flask app:

```bash
# Install python-dotenv if not already installed
pip install python-dotenv==1.0.0

# Start the application
python app.py
```

You should see:
```
🚀 Starting Adelaide Traffic Intelligence Dashboard...
   📍 URL: http://0.0.0.0:8504
   🔑 Google Maps API: ✅ Configured
```

## 🔍 Troubleshooting

### **If Google Maps API shows ❌ Missing:**
1. Check that `.env` file exists in the root directory
2. Verify the API key is correct (no extra spaces)
3. Ensure `python-dotenv` is installed
4. Restart the Flask application

### **If Maps Still Don't Load:**
1. Check browser console for API errors
2. Verify API key restrictions in Google Cloud Console
3. Ensure your domain/localhost is whitelisted

## 🚀 Final Test

1. **Start Flask app**: `python app.py`
2. **Open browser**: `http://localhost:8504`
3. **Click "Map View"** tab
4. **Verify** Google Maps loads with traffic markers

**Success! Your environment variables are configured correctly!** 🎉 