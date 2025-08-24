#!/usr/bin/env python3
"""
Quick run script for AI Text Detection Flask App
"""

import os
import sys

def check_models():
    """Check if model files exist"""
    models_dir = "../models-ml"
    required_models = [
        "best_ai_detector.pth",
        "best_optimized_roberta.pth", 
        "best_roberta_large_detector.pth",
        "ensemble_model_1.pth",
        "ensemble_model_2.pth",
        "ensemble_model_3.pth"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        print(f"⚠️ Missing model files in {models_dir}:")
        for model in missing_models:
            print(f"   - {model}")
        print("\n📝 Note: The app will work but some models may not load")
    else:
        print("✅ All model files found")
    
    return len(missing_models) == 0

def main():
    print("🚀 Starting AI Text Detection App")
    print("💻 Using CPU for model inference")
    print("=" * 50)
    
    # Check models
    check_models()
    
    print("\n🌐 Starting Flask server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped gracefully")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Try running: python app.py")

if __name__ == "__main__":
    main()
