#!/usr/bin/env python3
"""
Simple test to verify the imports work correctly
"""

try:
    print("Testing imports...")
    from localMultiplayerTetris.rl_utils.unified_trainer import UnifiedTrainer
    print("✅ UnifiedTrainer import successful")
    
    from localMultiplayerTetris.rl_utils.staged_unified_trainer import StagedTrainingConfig
    print("✅ StagedTrainingConfig import successful")
    
    from localMultiplayerTetris.rl_utils.staged_unified_trainer import StagedUnifiedTrainer
    print("✅ StagedUnifiedTrainer import successful")
    
    print("🎉 All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc() 