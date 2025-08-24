#!/usr/bin/env python3
"""
Simple startup script for AI Text Detection Flask App
"""

import sys
import os

def main():
    print("🚀 AI Text Detection System")
    print("💻 Configured for CPU inference")
    print("🌐 Starting Flask server...")
    print("📱 Open browser to: http://localhost:5000")
    print("=" * 50)
    
    try:
        # Import and run the app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have all requirements installed:")
        print("   pip install flask torch numpy")

if __name__ == "__main__":
    main()
