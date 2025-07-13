import os
import shutil
import argparse
import subprocess
import sys

def setup_directories():
    """Create necessary directory structure"""
    dirs = [
        'A2_smvs',
        'A2_smvs/book_covers',
        'A2_smvs/book_covers/Reference',
        'A2_smvs/book_covers/Query',
        'A2_smvs/landmarks',
        'A2_smvs/landmarks/Reference',
        'A2_smvs/landmarks/Query',
        'A2_smvs/museum_paintings',
        'A2_smvs/museum_paintings/Reference',
        'A2_smvs/museum_paintings/Query'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create placeholder book.png if it doesn't exist
    if not os.path.exists('book.png'):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create a simple image
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.text(0.5, 0.5, 'Computer\nVision\nExplorer', 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    fontsize=28, 
                    fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            
            # Save the image
            plt.savefig('book.png')
            plt.close()
            print("✓ Created placeholder book.png")
        except Exception as e:
            print(f"Could not create book.png: {str(e)}")

def check_requirements():
    """Check and install requirements"""
    try:
        # Try to import key packages
        import streamlit
        import numpy
        import cv2
        import matplotlib.pyplot
        from PIL import Image
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        print("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✓ Requirements installed successfully")
            return True
        except Exception as e:
            print(f"Error installing requirements: {str(e)}")
            return False

def setup_conda_env():
    """Setup conda environment if conda is available"""
    try:
        # Check if conda is available
        subprocess.check_call(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Create environment
        print("Creating conda environment...")
        subprocess.check_call(["conda", "env", "create", "-f", "environment.yml"])
        print("✓ Conda environment created")
        print("\nTo activate the environment, run:")
        print("conda activate cv-explorer")
        print("Then run:")
        print("streamlit run app.py")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Conda not available or error creating environment")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup Computer Vision Explorer app")
    parser.add_argument("--conda", action="store_true", help="Setup using conda environment")
    parser.add_argument("--pip", action="store_true", help="Setup using pip requirements")
    args = parser.parse_args()
    
    print("Setting up Computer Vision Explorer...")
    setup_directories()
    
    if args.conda:
        setup_conda_env()
    elif args.pip:
        check_requirements()
    else:
        # Try conda first, fall back to pip
        if not setup_conda_env():
            check_requirements()
    
    print("\nSetup complete!")
    print("To run the app:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main() 