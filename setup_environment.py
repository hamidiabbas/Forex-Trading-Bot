# setup_environment.py
import os
import sys
from pathlib import Path

def setup_project_structure():
    """Fix project structure and imports"""
    project_root = Path(__file__).parent
    print(f"🔧 Setting up project structure in: {project_root}")
    
    # Required directories
    required_dirs = ['config', 'core', 'strategies', 'utils', 'tests', 'logs']
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        
        # Create directory if it doesn't exist
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"📁 Created directory: {dir_name}/")
        else:
            print(f"✅ Directory exists: {dir_name}/")
        
        # Create __init__.py if it doesn't exist
        init_file = dir_path / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"📝 Created: {dir_name}/__init__.py")
        else:
            print(f"✅ __init__.py exists: {dir_name}/")
    
    # Check critical files
    critical_files = [
        'config/config.yaml',
        'config/configmanager.py',
        'core/marketintelligence.py',
        'core/riskmanager.py',
        'core/executionengine.py',
        'main.py'
    ]
    
    print("\n🔍 Checking critical files:")
    for file_path in critical_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ MISSING: {file_path}")
    
    print("\n✅ Project structure setup complete!")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports()

def test_imports():
    """Test if imports work correctly"""
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Test imports
        from config.configmanager import ConfigManager
        print("✅ config.configmanager import successful")
        
        from core.marketintelligence import EnhancedMarketIntelligence
        print("✅ core.marketintelligence import successful")
        
        from core.riskmanager import EnhancedRiskManager
        print("✅ core.riskmanager import successful")
        
        print("🎉 All critical imports working!")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        print("💡 You may need to check the actual file contents")

if __name__ == "__main__":
    setup_project_structure()
