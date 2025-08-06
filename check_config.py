"""
Configuration Diagnostic Script
Run this to see exactly what's in your config
"""

import sys
import os

def check_config():
    print("=" * 60)
    print("CONFIGURATION DIAGNOSTIC")
    print("=" * 60)
    
    # Check if config.py exists
    if os.path.exists('config.py'):
        print("✅ config.py file exists")
    else:
        print("❌ config.py file NOT found!")
        return
    
    # Try to import config
    try:
        import config
        print("✅ config module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return
    
    # Check what attributes config has
    print("\n📋 Config attributes found:")
    config_attrs = [attr for attr in dir(config) if not attr.startswith('_')]
    for attr in config_attrs:
        try:
            value = getattr(config, attr)
            if 'PASSWORD' in attr.upper():
                print(f"   {attr} = ****** (hidden)")
            else:
                print(f"   {attr} = {value}")
        except Exception as e:
            print(f"   {attr} = ERROR: {e}")
    
    # Specifically check MT5 credentials
    print("\n🔑 MT5 Credentials Check:")
    required_attrs = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
    
    for attr in required_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value and value != '':
                if 'PASSWORD' in attr:
                    print(f"   ✅ {attr} = ****** (set)")
                else:
                    print(f"   ✅ {attr} = {value}")
            else:
                print(f"   ❌ {attr} = {value} (empty or None)")
        else:
            print(f"   ❌ {attr} = NOT FOUND")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_config()
