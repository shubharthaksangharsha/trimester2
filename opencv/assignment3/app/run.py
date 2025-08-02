#!/usr/bin/env python3
"""
Computer Vision Assignment 3 - Application Launcher
Simple script to start the Flask web application with proper configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required directories and files exist"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Check for models directory
    models_dir = project_root / "models"
    if not models_dir.exists():
        print("❌ Models directory not found!")
        print(f"   Expected: {models_dir}")
        print("   Please ensure your trained models are saved in the models/ directory")
        return False
    
    # Check for model files
    model_files = list(models_dir.glob("*.pth"))
    if len(model_files) == 0:
        print("❌ No model files found!")
        print(f"   Expected: *.pth files in {models_dir}")
        print("   Please run your training notebook to generate model files")
        return False
    
    print(f"✅ Found {len(model_files)} model files")
    
    # Check for data directory
    data_dir = project_root / "data"
    if not data_dir.exists():
        print("❌ Data directory not found!")
        print(f"   Expected: {data_dir}")
        print("   Please ensure Fashion-MNIST data is downloaded")
        return False
    
    # Check for Fashion-MNIST
    fashion_mnist_dir = data_dir / "FashionMNIST"
    if not fashion_mnist_dir.exists():
        print("❌ Fashion-MNIST data not found!")
        print(f"   Expected: {fashion_mnist_dir}")
        print("   Please run your training notebook to download the dataset")
        return False
    
    print("✅ Fashion-MNIST dataset found")
    
    return True

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'flask',
        'torch',
        'torchvision', 
        'numpy',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Please install with: pip install -r requirements.txt")
        return False
    
    return True

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting Computer Vision Assignment 3 Web App...")
    print("="*60)
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = 'True'
    
    try:
        # Import and run the Flask app
        from app import app
        
        print("📱 Application URL: http://localhost:5000")
        print("🔧 Debug mode: Enabled")
        print("⚡ Auto-reload: Enabled")
        print("\n💡 Tips:")
        print("   - Upload images or draw to test models")
        print("   - Explore 3D visualizations with mouse controls")
        print("   - Compare different model architectures")
        print("   - Press Ctrl+C to stop the server")
        print("\n" + "="*60)
        
        # Start the Flask development server
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=True,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        print("   Please check that all dependencies are installed")
        return False
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Application error: {e}")
        return False

def main():
    """Main function"""
    print("🧠 Computer Vision Assignment 3 - Neural Network Explorer")
    print("=" * 60)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Dependency check failed!")
        print("   Please resolve the issues above before running the app")
        sys.exit(1)
    
    # Check Python packages
    print("\n📦 Checking Python packages...")
    if not check_python_packages():
        print("\n❌ Package check failed!")
        print("   Please install missing packages before running the app")
        sys.exit(1)
    
    # Start application
    print("\n✅ All checks passed!")
    start_application()

if __name__ == '__main__':
    main()