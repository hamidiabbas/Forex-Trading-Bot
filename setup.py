# setup.py - Quick Setup Script
"""
Quick setup script to create all required modules and install dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required dependencies"""
    requirements = [
        "stable-baselines3[extra]",
        "transformers",
        "torch",
        "gymnasium", 
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "yfinance",
        "scikit-learn",
        "scipy"
    ]
    
    for requirement in requirements:
        try:
            print(f"Installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"✅ {requirement} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {requirement}: {e}")

def create_directories():
    """Create required directories"""
    directories = [
        "logs/production_training",
        "models",
        "results",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

if __name__ == "__main__":
    print("🚀 Setting up Forex Trading Bot...")
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Installing dependencies...")
    install_requirements()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Create the core modules (datahandler.py, marketintelligence.py, etc.)")
    print("2. Run: python enhanced_train_rl_model.py")
