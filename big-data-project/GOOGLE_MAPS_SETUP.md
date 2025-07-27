# 🗺️ Google Maps API Setup Guide

## 🔑 How to Get Google Maps API Key

### Step 1: Go to Google Cloud Console
1. **Visit**: [Google Cloud Console](https://console.cloud.google.com/)
2. **Sign in** with your Google account

### Step 2: Create or Select a Project
1. **Click** the project dropdown (top left)
2. **Create New Project** or select existing one
3. **Name your project**: "Adelaide Traffic Intelligence" (or any name)
4. **Click "Create"**

### Step 3: Enable Maps JavaScript API
1. **Go to**: "APIs & Services" → "Library"
2. **Search for**: "Maps JavaScript API"
3. **Click** on "Maps JavaScript API"
4. **Click "Enable"**

### Step 4: Create API Key
1. **Go to**: "APIs & Services" → "Credentials"
2. **Click**: "+ CREATE CREDENTIALS"
3. **Select**: "API Key"
4. **Copy** the generated API key

### Step 5: Restrict API Key (Important for Security)
1. **Click** on your newly created API key
2. **Application restrictions**:
   - Select "HTTP referrers (web sites)"
   - Add these referrers:
     ```
     https://ati-bigdata.devshubh.me/*
     http://localhost:8504/*
     https://localhost:8504/*
     ```
3. **API restrictions**:
   - Select "Restrict key"
   - Choose "Maps JavaScript API"
4. **Click "Save"**

## 📝 Where to Place Your API Key

### In Your HTML File (`templates/dashboard.html`):

**Find this line** (around line 23):
```javascript
// 🗝️ REPLACE WITH YOUR GOOGLE MAPS API KEY HERE:
script.src = 'https://maps.googleapis.com/maps/api/js?key=YOUR_GOOGLE_MAPS_API_KEY_HERE&loading=async&callback=initMap';
```

**Replace `YOUR_GOOGLE_MAPS_API_KEY_HERE`** with your actual API key:
```javascript
// 🗝️ REPLACE WITH YOUR GOOGLE MAPS API KEY HERE:
script.src = 'https://maps.googleapis.com/maps/api/js?key=AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxx&loading=async&callback=initMap';
```

## 🔒 API Key Security Best Practices

### 1. Domain Restrictions
Always restrict your API key to specific domains:
- `ati-bigdata.devshubh.me`
- `localhost:8504` (for development)

### 2. API Restrictions
Only enable the APIs you need:
- Maps JavaScript API (for your traffic dashboard)
- Optional: Places API (if you add location search)

### 3. Usage Monitoring
- Set up billing alerts in Google Cloud Console
- Monitor API usage to avoid unexpected charges

### 4. Environment Variables (Advanced)
For production deployments, consider using environment variables:

**In your Flask app.py**:
```python
import os
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'your-default-key')

@app.route('/config')
def get_config():
    return {'maps_api_key': GOOGLE_MAPS_API_KEY}
```

**In your HTML**:
```javascript
fetch('/config')
    .then(response => response.json())
    .then(config => {
        script.src = `https://maps.googleapis.com/maps/api/js?key=${config.maps_api_key}&loading=async&callback=initMap`;
    });
```

## 💰 Google Maps Pricing

### Free Tier
- **$200 monthly credit** (covers ~28,000 map loads)
- **First 28,000 map loads**: FREE each month
- Perfect for development and small traffic dashboards

### Pay-as-you-go
- **After free tier**: $7 per 1,000 additional map loads
- **Dynamic Maps**: $7 per 1,000 loads
- **Static Maps**: $2 per 1,000 loads

## 🛠️ Troubleshooting

### Common Errors:

**1. `InvalidKeyMapError`**
- ✅ **Solution**: Replace `YOUR_GOOGLE_MAPS_API_KEY_HERE` with your actual API key

**2. `RefererNotAllowedMapError`**
- ✅ **Solution**: Add your domain to HTTP referrer restrictions in Google Cloud Console

**3. `RequestDeniedMapError`**
- ✅ **Solution**: Enable "Maps JavaScript API" in Google Cloud Console

**4. Map shows "This page didn't load Google Maps correctly"**
- ✅ **Solution**: Check browser console for specific error messages
- ✅ **Check**: API key is correct and has proper restrictions

### Testing Your Setup:
1. **Open browser console** (F12)
2. **Look for errors** mentioning "Google Maps"
3. **Check network tab** for failed requests to `maps.googleapis.com`

## 🚀 Quick Setup Commands

### For Local Development:
```bash
# Open the HTML file
code templates/dashboard.html

# Find line with YOUR_GOOGLE_MAPS_API_KEY_HERE
# Replace with your actual API key
# Save file

# Restart your Flask app
python app.py
```

### For Production (AWS):
```bash
# SSH into your server
ssh ubuntu@your-ec2-ip

# Edit the HTML file
nano /home/ubuntu/trimester2/big-data-project/templates/dashboard.html

# Find and replace the API key
# Save with Ctrl+X, Y, Enter

# Restart the service
sudo systemctl restart ati-bigdata
```

## ✅ Verification

Once set up correctly, you should see:
- ✅ **Interactive Google Map** in the "Map View" tab
- ✅ **Adelaide intersections** marked with colored circles
- ✅ **Info windows** when clicking markers
- ✅ **No console errors** about Google Maps

## 🎯 Expected Result

Your **Adelaide Traffic Intelligence Dashboard** will show:
- 🌐 **3D View**: Interactive Three.js city visualization
- 🗺️ **Map View**: Real Google Maps with traffic data
- 📍 **20 Adelaide intersections** with live traffic volumes
- 🔴 **Color-coded markers** based on traffic congestion
- 📊 **Click for details** showing traffic stats

**Your teacher will be impressed by the professional Google Maps integration!** 🚀✨ 