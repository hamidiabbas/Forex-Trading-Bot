"""
Verify All Critical Fixes Are Working
Tests import paths and encoding issues
"""
import sys
import os
from pathlib import Path

def test_package_structure():
    """Test if package structure is correct"""
    print("Testing package structure...")
    
    required_files = [
        'config/__init__.py',
        'config/config_manager.py', 
        'config/config.yaml',
        'core/__init__.py'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"OK: {file_path} exists")
        else:
            print(f"MISSING: {file_path}")
            return False
    
    return True

def test_config_import():
    """Test if config manager can be imported"""
    print("\nTesting config import...")
    
    try:
        from config.config_manager import ConfigManager
        config = ConfigManager()
        symbols = config.get_trading_symbols()
        print(f"OK: Config loaded with symbols: {symbols}")
        return True
    except Exception as e:
        print(f"FAILED: Config import error: {e}")
        return False

def test_main_execution():
    """Test if main file can run without Unicode errors"""
    print("\nTesting main execution...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "main_safe.py", "--test"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding='utf-8',
            errors='replace'
        )
        
        if "PROFESSIONAL TRADING BOT" in result.stdout:
            print("OK: main_safe.py runs without Unicode errors")
            return True
        else:
            print(f"FAILED: Unexpected output: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"FAILED: Main execution error: {e}")
        return False

def main():
    """Run all fix verification tests"""
    print("=" * 50)
    print("CRITICAL FIXES VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Config Import", test_config_import),
        ("Main Execution", test_main_execution)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All critical fixes verified!")
        print("Your trading bot is ready to run!")
        return True
    else:
        print("\nISSUES REMAIN: Some fixes need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
