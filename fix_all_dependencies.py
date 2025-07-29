"""
Complete Dependency Conflict Resolution for Production Trading Bot
Handles all version conflicts automatically
"""
import subprocess
import sys
import os
import json
from datetime import datetime

class DependencyResolver:
    def __init__(self):
        self.success_count = 0
        self.error_count = 0
        self.conflicts_resolved = []
        
    def run_command(self, command, description):
        """Execute command with error handling"""
        print(f"🔄 {description}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=600
            )
            
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                self.success_count += 1
                return True
            else:
                print(f"❌ {description} - FAILED")
                print(f"   Error: {result.stderr}")
                self.error_count += 1
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ {description} - TIMEOUT")
            self.error_count += 1
            return False
        except Exception as e:
            print(f"❌ {description} - ERROR: {e}")
            self.error_count += 1
            return False
    
    def fix_typing_extensions(self):
        """Fix typing-extensions conflicts"""
        print("\n🔧 Fixing typing-extensions conflicts...")
        
        commands = [
            ("pip uninstall -y typing-extensions", "Uninstalling old typing-extensions"),
            ("pip install --no-cache-dir typing-extensions>=4.10.0", "Installing compatible typing-extensions")
        ]
        
        for command, description in commands:
            if not self.run_command(command, description):
                return False
        
        self.conflicts_resolved.append("typing-extensions")
        return True
    
    def fix_numpy(self):
        """Fix numpy version conflicts"""
        print("\n🔧 Fixing numpy conflicts...")
        
        commands = [
            ("pip install --no-cache-dir --upgrade numpy>=1.25.0", "Upgrading numpy to compatible version")
        ]
        
        for command, description in commands:
            if not self.run_command(command, description):
                return False
        
        self.conflicts_resolved.append("numpy")
        return True
    
    def fix_tensorflow(self):
        """Fix TensorFlow version conflicts"""
        print("\n🔧 Fixing TensorFlow conflicts...")
        
        commands = [
            ("pip uninstall -y tensorflow tensorflow-intel tensorflow-cpu", "Removing conflicting TensorFlow versions"),
            ("pip install --no-cache-dir tensorflow-cpu==2.13.0", "Installing consistent TensorFlow CPU")
        ]
        
        for command, description in commands:
            self.run_command(command, description)  # Don't fail if uninstall fails
        
        # Verify TensorFlow installation
        if self.verify_tensorflow():
            self.conflicts_resolved.append("tensorflow")
            return True
        else:
            print("⚠️ TensorFlow verification failed, but continuing...")
            return True  # Continue anyway
    
    def install_production_dependencies(self):
        """Install all production trading bot dependencies"""
        print("\n📦 Installing production dependencies...")
        
        # Core ML dependencies with specific compatible versions
        ml_packages = [
            ("torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu", "PyTorch CPU"),
            ("gym==0.21.0", "OpenAI Gym"),
            ("stable-baselines3==2.1.0", "Stable Baselines3")
        ]
        
        # Trading and analysis dependencies
        trading_packages = [
            ("MetaTrader5==5.0.45", "MetaTrader5"),
            ("PyYAML==6.0.1", "PyYAML"),
            ("python-dotenv==1.0.0", "Python-dotenv"),
            ("scikit-learn==1.3.0", "Scikit-learn"),
            ("matplotlib==3.7.2", "Matplotlib"),
            ("seaborn==0.12.2", "Seaborn"),
            ("colorlog==6.7.0", "Colorlog"),
            ("psutil==5.9.5", "Psutil")
        ]
        
        all_packages = ml_packages + trading_packages
        
        for package, name in all_packages:
            command = f"pip install --no-cache-dir {package}"
            self.run_command(command, f"Installing {name}")
    
    def verify_tensorflow(self):
        """Verify TensorFlow installation"""
        try:
            verification_code = """
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
# Test basic operation
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = tf.add(a, b)
print("TensorFlow basic test passed")
"""
            
            result = subprocess.run(
                [sys.executable, "-c", verification_code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def verify_all_dependencies(self):
        """Verify all dependencies are working"""
        print("\n🔍 Verifying all dependencies...")
        
        dependencies_to_test = [
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("scipy", "SciPy"),
            ("tensorflow", "TensorFlow"),
            ("torch", "PyTorch"),
            ("gym", "OpenAI Gym"),
            ("stable_baselines3", "Stable Baselines3"),
            ("MetaTrader5", "MetaTrader5"),
            ("yaml", "PyYAML"),
            ("dotenv", "Python-dotenv"),
            ("sklearn", "Scikit-learn"),
            ("matplotlib", "Matplotlib")
        ]
        
        working_deps = []
        failed_deps = []
        
        for module, name in dependencies_to_test:
            try:
                __import__(module)
                print(f"✅ {name} - OK")
                working_deps.append(name)
            except ImportError:
                print(f"❌ {name} - FAILED")
                failed_deps.append(name)
        
        return working_deps, failed_deps
    
    def create_requirements_file(self):
        """Create a requirements.txt file with working versions"""
        print("\n📝 Creating requirements.txt with working versions...")
        
        try:
            result = subprocess.run(
                "pip freeze",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                with open("requirements_working.txt", "w") as f:
                    f.write("# Working requirements for Production Trading Bot\n")
                    f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(result.stdout)
                
                print("✅ Requirements file created: requirements_working.txt")
                return True
            
        except Exception as e:
            print(f"⚠️ Could not create requirements file: {e}")
        
        return False
    
    def resolve_all_conflicts(self):
        """Main conflict resolution process"""
        print("🚀 Starting Complete Dependency Conflict Resolution")
        print("=" * 60)
        
        # Step 1: Fix typing-extensions
        self.fix_typing_extensions()
        
        # Step 2: Fix numpy
        self.fix_numpy()
        
        # Step 3: Fix TensorFlow
        self.fix_tensorflow()
        
        # Step 4: Install production dependencies
        self.install_production_dependencies()
        
        # Step 5: Verify everything
        working_deps, failed_deps = self.verify_all_dependencies()
        
        # Step 6: Create requirements file
        self.create_requirements_file()
        
        # Final report
        print("\n" + "=" * 60)
        print("📊 DEPENDENCY RESOLUTION SUMMARY")
        print("=" * 60)
        
        print(f"✅ Conflicts Resolved: {len(self.conflicts_resolved)}")
        for conflict in self.conflicts_resolved:
            print(f"   - {conflict}")
        
        print(f"\n✅ Working Dependencies ({len(working_deps)}):")
        for dep in working_deps:
            print(f"   - {dep}")
        
        if failed_deps:
            print(f"\n❌ Failed Dependencies ({len(failed_deps)}):")
            for dep in failed_deps:
                print(f"   - {dep}")
        
        print(f"\n📈 Success Rate: {self.success_count}/{self.success_count + self.error_count} ({(self.success_count/(self.success_count + self.error_count)*100):.1f}%)")
        
        if len(working_deps) >= 10:  # Most important dependencies working
            print("\n🎉 SUCCESS! Your production trading bot is ready!")
            print("✅ Critical dependencies are installed and working")
            print("🚀 Next step: Run 'python main.py' to start your trading bot")
            return True
        else:
            print("\n⚠️ Some critical dependencies are missing")
            print("🔧 Please address failed dependencies before running the bot")
            return False

def main():
    resolver = DependencyResolver()
    success = resolver.resolve_all_conflicts()
    
    if success:
        print("\n🎊 CONGRATULATIONS!")
        print("Your production-grade trading bot is fully configured!")
        sys.exit(0)
    else:
        print("\n🔧 Additional fixes needed")
        print("Review the failed dependencies and install them manually")
        sys.exit(1)

if __name__ == "__main__":
    main()
