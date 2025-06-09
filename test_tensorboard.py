#!/usr/bin/env python3
"""
Test TensorBoard accessibility
"""

import requests
import time

def test_tensorboard_access():
    """Test if TensorBoard instances are accessible"""
    
    ports = {
        6007: "DQN Locked",
        6008: "DQN Hierarchical", 
        6009: "DQN Movement",
        6006: "DREAM"
    }
    
    print("🔍 Testing TensorBoard Accessibility")
    print("=" * 50)
    
    for port, name in ports.items():
        try:
            response = requests.get(f"http://localhost:{port}", timeout=3)
            if response.status_code == 200:
                print(f"✅ {name:18} - http://localhost:{port} - ACCESSIBLE")
            else:
                print(f"⚠️  {name:18} - http://localhost:{port} - RESPONSE: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ {name:18} - http://localhost:{port} - NOT RUNNING")
        except Exception as e:
            print(f"💥 {name:18} - http://localhost:{port} - ERROR: {e}")

if __name__ == "__main__":
    test_tensorboard_access() 