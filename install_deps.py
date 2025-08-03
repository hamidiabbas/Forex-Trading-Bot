# install_deps.py - Install all required dependencies
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def main():
    print("Installing required dependencies...")
    
    packages = [
        "torch",
        "stable-baselines3[extra]",
        "gymnasium",
        "yfinance",
        "pandas",
        "numpy",
        "matplotlib",
        "transformers"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstallation complete: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("All dependencies installed successfully!")
        print("You can now run: python enhanced_train_rl_model.py")
    else:
        print("Some packages failed to install. Please install manually.")

if __name__ == "__main__":
    main()
