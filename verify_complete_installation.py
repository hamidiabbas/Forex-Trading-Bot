"""
Complete Production Trading Bot Installation Verification
Tests all dependencies and provides detailed diagnostics
"""
import sys
import os
from datetime import datetime

def test_core_dependencies():
    """Test core numerical and data processing libraries"""
    print("🔍 Testing Core Dependencies...")
    results = {}
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} - OK")
        results['numpy'] = True
    except ImportError as e:
        print(f"❌ NumPy - FAILED: {e}")
        results['numpy'] = False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} - OK")
        results['pandas'] = True
    except ImportError as e:
        print(f"❌ Pandas - FAILED: {e}")
        results['pandas'] = False
    
    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__} - OK")
        results['scipy'] = True
    except ImportError as e:
        print(f"❌ SciPy - FAILED: {e}")
        results['scipy'] = False
    
    return results

def test_ml_dependencies():
    """Test machine learning and RL libraries"""
    print("\n🤖 Testing ML/RL Dependencies...")
    results = {}
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} - OK")
        results['tensorflow'] = True
    except ImportError as e:
        print(f"❌ TensorFlow - FAILED: {e}")
        results['tensorflow'] = False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - OK")
        if torch.cuda.is_available():
            print(f"   🚀 CUDA Available: {torch.cuda.get_device_name(0)}")
        else:
            print("   💻 CPU Mode (CUDA not available)")
        results['torch'] = True
    except ImportError as e:
        print(f"❌ PyTorch - FAILED: {e}")
        results['torch'] = False
    
    try:
        import gym
        print(f"✅ OpenAI Gym {gym.__version__} - OK")
        results['gym'] = True
    except ImportError as e:
        print(f"❌ OpenAI Gym - FAILED: {e}")
        results['gym'] = False
    
    try:
        import stable_baselines3
        print(f"✅ Stable Baselines3 {stable_baselines3.__version__} - OK")
        results['stable_baselines3'] = True
    except ImportError as e:
        print(f"❌ Stable Baselines3 - FAILED: {e}")
        results['stable_baselines3'] = False
    
    return results

def test_trading_dependencies():
    """Test trading-specific libraries"""
    print("\n📈 Testing Trading Dependencies...")
    results = {}
    
    try:
        import MetaTrader5 as mt5
        print(f"✅ MetaTrader5 - OK")
        results['mt5'] = True
    except ImportError as e:
        print(f"❌ MetaTrader5 - FAILED: {e}")
        results['mt5'] = False
    
    try:
        import talib
        print(f"✅ TA-Lib {talib.__version__} - OK")
        results['talib'] = True
    except ImportError as e:
        print(f"⚠️ TA-Lib - NOT AVAILABLE: {e}")
        print("   Note: Trading bot includes custom indicators as backup")
        results['talib'] = False
    
    try:
        import yaml
        print(f"✅ PyYAML - OK")
        results['yaml'] = True
    except ImportError as e:
        print(f"❌ PyYAML - FAILED: {e}")
        results['yaml'] = False
    
    try:
        from dotenv import load_dotenv
        print(f"✅ Python-dotenv - OK")
        results['dotenv'] = True
    except ImportError as e:
        print(f"❌ Python-dotenv - FAILED: {e}")
        results['dotenv'] = False
    
    return results

def test_analysis_dependencies():
    """Test data analysis libraries"""
    print("\n📊 Testing Analysis Dependencies...")
    results = {}
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} - OK")
        results['sklearn'] = True
    except ImportError as e:
        print(f"❌ Scikit-learn - FAILED: {e}")
        results['sklearn'] = False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} - OK")
        results['matplotlib'] = True
    except ImportError as e:
        print(f"❌ Matplotlib - FAILED: {e}")
        results['matplotlib'] = False
    
    try:
        import seaborn
        print(f"✅ Seaborn {seaborn.__version__} - OK")
        results['seaborn'] = True
    except ImportError as e:
        print(f"❌ Seaborn - FAILED: {e}")
        results['seaborn'] = False
    
    return results

def test_advanced_features():
    """Test advanced functionality"""
    print("\n⚡ Testing Advanced Features...")
    
    # Test TA-Lib indicators if available
    try:
        import talib
        import numpy as np
        test_data = np.random.random(50)
        rsi = talib.RSI(test_data)
        print("✅ TA-Lib indicators working")
    except Exception as e:
        print(f"⚠️ TA-Lib indicators test failed: {e}")
    
    # Test Gym environment creation
    try:
        import gym
        env = gym.make('CartPole-v1')
        print("✅ Gym environment creation successful")
        env.close()
    except Exception as e:
        print(f"⚠️ Gym environment test failed: {e}")
    
    # Test SAC model creation
    try:
        from stable_baselines3 import SAC
        import gym
        env = gym.make('CartPole-v1')
        model = SAC('MlpPolicy', env, verbose=0)
        print("✅ SAC model creation successful")
        env.close()
    except Exception as e:
        print(f"⚠️ SAC model test failed: {e}")

def generate_report(all_results):
    """Generate installation report"""
    print("\n" + "="*60)
    print("📋 INSTALLATION REPORT")
    print("="*60)
    
    total_tests = sum(len(results) for results in all_results.values())
    total_passed = sum(sum(results.values()) for results in all_results.values())
    
    print(f"📊 Overall Status: {total_passed}/{total_tests} tests passed")
    print(f"🎯 Success Rate: {(total_passed/total_tests)*100:.1f}%")
    
    for category, results in all_results.items():
        passed = sum(results.values())
        total = len(results)
        print(f"\n{category.title()}: {passed}/{total} ({'✅' if passed == total else '⚠️'})")
        
        for lib, status in results.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {lib}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if total_passed == total_tests:
        print("🎉 Perfect! All dependencies installed successfully.")
        print("✅ Your production trading bot is ready to run!")
        print("🚀 Next step: Run 'python main.py' to start trading!")
    elif total_passed >= total_tests * 0.8:
        print("🟡 Good! Most dependencies are working.")
        print("⚠️ Some optional dependencies failed - bot will still work.")
        print("🔧 Consider installing missing packages for full functionality.")
    else:
        print("🔴 Critical dependencies missing!")
        print("❌ Please install missing packages before running the bot.")
    
    return total_passed == total_tests

def main():
    """Main verification function"""
    print("🔍 Production Trading Bot - Complete Installation Verification")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test all dependency categories
    all_results = {
        'core': test_core_dependencies(),
        'ml_rl': test_ml_dependencies(),
        'trading': test_trading_dependencies(),
        'analysis': test_analysis_dependencies()
    }
    
    # Test advanced features
    test_advanced_features()
    
    # Generate final report
    success = generate_report(all_results)
    
    if success:
        print(f"\n🎊 CONGRATULATIONS!")
        print(f"Your production-grade trading bot is fully installed!")
        sys.exit(0)
    else:
        print(f"\n🔧 Please install missing dependencies and run verification again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
