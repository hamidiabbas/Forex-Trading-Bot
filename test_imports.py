# test_imports.py
import os
import sys

print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))
print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nFiles in current directory:")
for file in os.listdir('.'):
    if file.endswith('.py'):
        print(f"  ✓ {file}")

print("\nTesting imports...")
try:
    from enhanced_feature_engineering import FeatureManager
    print("✅ enhanced_feature_engineering imported successfully")
except ImportError as e:
    print(f"❌ enhanced_feature_engineering failed: {e}")

try:
    from enhanced_datahandler import EnhancedDataHandler
    print("✅ enhanced_datahandler imported successfully")
except ImportError as e:
    print(f"❌ enhanced_datahandler failed: {e}")
