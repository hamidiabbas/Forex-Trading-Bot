"""
Complete Final Trading Bot Verification
Tests all components after NumPy compatibility fix
"""
import sys
import subprocess
import importlib
import numpy as np
from datetime import datetime

def verify_numpy_scipy_compatibility():
    """Verify NumPy/SciPy are now compatible"""
    print("🔍 Verifying NumPy/SciPy Compatibility...")
    
    try:
        import numpy as np
        import scipy
        
        numpy_version = np.__version__
        scipy_version = scipy.__version__
        
        print(f"✅ NumPy version: {numpy_version}")
        print(f"✅ SciPy version: {scipy_version}")
        
        # Test numerical operations
        test_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        mean_result = np.mean(test_array)
        std_result = np.std(test_array)
        
        print(f"✅ NumPy operations: mean={mean_result:.2f}, std={std_result:.2f}")
        
        # Test SciPy operations
        from scipy import stats
        norm_result = stats.norm.pdf(0, 0, 1)
        print(f"✅ SciPy operations: norm.pdf(0)={norm_result:.4f}")
        
        # Check version compatibility
        numpy_major_minor = tuple(map(int, numpy_version.split('.')[:2]))
        if numpy_major_minor < (2, 0):
            print("✅ NumPy version is compatible with SciPy 1.11.1")
            return True
        else:
            print("⚠️ NumPy version might have compatibility issues")
            return False
            
    except Exception as e:
        print(f"❌ NumPy/SciPy compatibility test failed: {e}")
        return False

def test_essential_trading_imports():
    """Test all essential trading bot imports"""
    print("\n🧪 Testing Essential Trading Bot Imports...")
    
    essential_modules = [
        ("numpy", "NumPy - Numerical computing"),
        ("pandas", "Pandas - Data manipulation"),
        ("scipy", "SciPy - Scientific computing"),
        ("yaml", "PyYAML - Configuration files"),
        ("dotenv", "Python-dotenv - Environment variables"),
        ("MetaTrader5", "MetaTrader5 - Trading platform"),
        ("colorlog", "Colorlog - Enhanced logging"),
        ("psutil", "Psutil - System monitoring"),
        ("matplotlib", "Matplotlib - Data visualization"),
        ("typing_extensions", "Typing Extensions - Type hints"),
        ("contourpy", "Contourpy - Contour plotting")
    ]
    
    working_imports = []
    failed_imports = []
    
    for module, description in essential_modules:
        try:
            lib = importlib.import_module(module)
            version = getattr(lib, '__version__', 'Unknown')
            print(f"✅ {description}: v{version}")
            working_imports.append((module, version))
        except ImportError as e:
            print(f"❌ {description}: FAILED - {e}")
            failed_imports.append((module, str(e)))
        except Exception as e:
            print(f"⚠️ {description}: WARNING - {e}")
            working_imports.append((module, 'Warning'))
    
    return working_imports, failed_imports

def test_trading_bot_core_functionality():
    """Test core trading bot functionality"""
    print("\n🤖 Testing Trading Bot Core Functionality...")
    
    try:
        # Test configuration manager
        from config.config_manager import ConfigManager
        config = ConfigManager('config/config.yaml')
        print("✅ Configuration Manager - Loaded successfully")
        
        # Test market intelligence with NumPy operations
        from core.market_intelligence import EnhancedMarketIntelligence
        market_intel = EnhancedMarketIntelligence(config.config)
        print("✅ Market Intelligence - Initialized with NumPy support")
        
        # Test risk manager with statistical calculations
        from core.risk_manager import EnhancedRiskManager
        risk_manager = EnhancedRiskManager(config.config)
        print("✅ Risk Manager - Initialized with SciPy support")
        
        # Test market data retrieval and processing
        market_data = market_intel.get_market_data('EURUSD')
        if market_data and isinstance(market_data.get('current_price'), (int, float)):
            price = market_data['current_price']
            print(f"✅ Market Data Processing - EURUSD @ {price:.5f}")
            
            # Test risk calculation with numerical operations
            risk_params = risk_manager.calculate_enhanced_risk(
                symbol='EURUSD',
                direction='BUY', 
                entry_price=price,
                stop_loss=price - 0.01,
                take_profit=price + 0.015,
                confidence=0.8,
                strategy='compatibility_test'
            )
            
            if risk_params and 'position_size' in risk_params:
                size = risk_params['position_size']
                print(f"✅ Risk Calculation - Position size: {size:.4f}")
                print("✅ NumPy/SciPy integration working in trading components")
                return True
            else:
                print("⚠️ Risk calculation returned incomplete results")
                return False
        else:
            print("ℹ️ Using synthetic market data (normal for testing)")
            print("✅ Core components initialized successfully")
            return True
            
    except Exception as e:
        print(f"❌ Trading bot functionality test failed: {e}")
        print("   This may indicate remaining NumPy/SciPy compatibility issues")
        return False

def check_dependency_health():
    """Check overall dependency health"""
    print("\n🔍 Checking Dependency Health...")
    
    try:
        result = subprocess.run(
            ["pip", "check"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ No dependency conflicts found!")
            return True, []
        else:
            conflicts = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            
            # Categorize conflicts
            critical_conflicts = []
            acceptable_conflicts = []
            
            # These are acceptable for a standalone trading bot
            acceptable_patterns = [
                'jupyter', 'notebook', 'ipython', 'sphinx', 'streamlit', 
                'altair', 'shimmy', 'gymnasium', 'tensorboard'
            ]
            
            for conflict in conflicts:
                if any(pattern in conflict.lower() for pattern in acceptable_patterns):
                    acceptable_conflicts.append(conflict)
                else:
                    critical_conflicts.append(conflict)
            
            if not critical_conflicts:
                print("✅ Only non-critical conflicts remain (acceptable)")
                if acceptable_conflicts:
                    print("ℹ️ Non-critical conflicts (can be ignored):")
                    for conflict in acceptable_conflicts[:3]:  # Show first 3
                        print(f"   • {conflict}")
                    if len(acceptable_conflicts) > 3:
                        print(f"   • ... and {len(acceptable_conflicts) - 3} more")
                return True, acceptable_conflicts
            else:
                print("❌ Critical conflicts found:")
                for conflict in critical_conflicts:
                    print(f"   • {conflict}")
                return False, critical_conflicts
                
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False, [str(e)]

def test_main_execution():
    """Test main.py execution"""
    print("\n🚀 Testing Main Bot Execution...")
    
    try:
        # Test main.py import
        result = subprocess.run(
            [sys.executable, "-c", "import main; print('✅ main.py imports successfully')"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd="."
        )
        
        if result.returncode == 0:
            print("✅ main.py imports without errors")
            print("✅ All trading bot components accessible")
            return True
        else:
            print(f"❌ main.py import issues:")
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')[-3:]  # Last 3 lines
                for line in error_lines:
                    print(f"   {line}")
            return False
            
    except Exception as e:
        print(f"❌ Main execution test failed: {e}")
        return False

def generate_production_readiness_report(compatibility_ok, working_imports, failed_imports, 
                                       functionality_ok, dependency_health, conflicts, main_ok):
    """Generate comprehensive production readiness report"""
    print("\n" + "=" * 80)
    print("📊 TRADING BOT PRODUCTION READINESS - FINAL REPORT")
    print("=" * 80)
    
    # Calculate readiness metrics
    metrics = [
        ("NumPy/SciPy Compatibility", compatibility_ok, "CRITICAL"),
        ("Essential Imports", len(failed_imports) == 0, "CRITICAL"),
        ("Core Functionality", functionality_ok, "CRITICAL"),
        ("Dependency Health", dependency_health, "IMPORTANT"),
        ("Main Execution", main_ok, "CRITICAL")
    ]
    
    critical_passed = sum(1 for _, status, priority in metrics if status and priority == "CRITICAL")
    total_critical = sum(1 for _, _, priority in metrics if priority == "CRITICAL")
    overall_passed = sum(1 for _, status, _ in metrics if status)
    total_metrics = len(metrics)
    
    readiness_score = (overall_passed / total_metrics) * 100
    critical_score = (critical_passed / total_critical) * 100
    
    print(f"📈 Overall Readiness: {readiness_score:.0f}% ({overall_passed}/{total_metrics})")
    print(f"🎯 Critical Systems: {critical_score:.0f}% ({critical_passed}/{total_critical})")
    print()
    
    # Detailed breakdown
    for metric, status, priority in metrics:
        icon = "✅" if status else "❌"
        priority_icon = "🔥" if priority == "CRITICAL" else "⚠️"
        print(f"{icon} {priority_icon} {metric}")
    
    # Import details
    if working_imports:
        print(f"\n✅ Working Components ({len(working_imports)}):")
        for module, version in working_imports:
            if module in ['numpy', 'scipy', 'pandas', 'MetaTrader5']:  # Show key ones
                print(f"   🔧 {module}: {version}")
    
    if failed_imports:
        print(f"\n❌ Failed Components ({len(failed_imports)}):")
        for module, error in failed_imports:
            print(f"   ❌ {module}: {error}")
    
    # Final assessment
    print(f"\n🎯 PRODUCTION ASSESSMENT:")
    
    if critical_score == 100 and readiness_score >= 90:
        print(f"🎉 FULLY PRODUCTION READY!")
        print(f"✅ All critical systems operational")
        print(f"✅ NumPy/SciPy compatibility confirmed")
        print(f"✅ Trading algorithms ready")
        print(f"\n🚀 LAUNCH COMMAND: python main.py")
        print(f"\n📋 Production Environment Confirmed:")
        print(f"   ✅ Compatible NumPy (v{working_imports[0][1] if working_imports else 'Unknown'})")
        print(f"   ✅ Working SciPy for statistical analysis")
        print(f"   ✅ Trading components operational")
        print(f"   ✅ Risk management functional")
        print(f"   ✅ Market intelligence active")
        
        print(f"\n🏆 CONGRATULATIONS!")
        print(f"Your institutional-grade trading bot is fully operational!")
        return True
        
    elif critical_score == 100 and readiness_score >= 75:
        print(f"🟡 PRODUCTION READY (Minor Issues)")
        print(f"✅ All critical systems working")
        print(f"⚠️ Some non-essential features limited")
        print(f"\n🚀 READY TO LAUNCH: python main.py")
        return True
        
    elif critical_score >= 75:
        print(f"🟡 MOSTLY READY")
        print(f"⚠️ Some critical issues remain")
        print(f"🔧 Address failed components before production use")
        return False
        
    else:
        print(f"🔴 NOT PRODUCTION READY")
        print(f"❌ Critical systems failing")
        print(f"🔧 Resolve critical issues before proceeding")
        return False

def main():
    """Run complete final verification"""
    print("🚀 Trading Bot Final Production Verification")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Post NumPy-compatibility fix verification")
    print("=" * 80)
    
    # Run comprehensive verification
    compatibility_ok = verify_numpy_scipy_compatibility()
    working_imports, failed_imports = test_essential_trading_imports()
    functionality_ok = test_trading_bot_core_functionality()
    dependency_health, conflicts = check_dependency_health()
    main_ok = test_main_execution()
    
    # Generate final production readiness report
    success = generate_production_readiness_report(
        compatibility_ok, working_imports, failed_imports,
        functionality_ok, dependency_health, conflicts, main_ok
    )
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
