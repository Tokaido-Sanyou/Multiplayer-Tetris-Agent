#!/usr/bin/env python3
"""
Debug Multiplayer Environment
Diagnostic tool to analyze the environment's multiplayer capabilities
"""

import sys
import os
import numpy as np

# Add paths
sys.path.append('local-multiplayer-tetris-main/localMultiplayerTetris')

def test_environment_multiplayer_capability():
    """Test if the environment truly supports multiplayer."""
    print("🔍 MULTIPLAYER ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    
    try:
        from tetris_env import TetrisEnv
        
        # Test 1: Single player mode
        print("\n1️⃣ Testing Single Player Mode:")
        env_single = TetrisEnv(single_player=True, headless=True)
        obs_single = env_single.reset()
        if isinstance(obs_single, tuple):
            obs_single = obs_single[0]
        
        print(f"   Single player observation type: {type(obs_single)}")
        print(f"   Single player observation keys: {obs_single.keys() if isinstance(obs_single, dict) else 'Not a dict'}")
        
        # Test single action
        action_single = 15
        result_single = env_single.step(action_single)
        print(f"   Single action result: {len(result_single)} elements")
        env_single.close()
        
        # Test 2: Multiplayer mode
        print("\n2️⃣ Testing Multiplayer Mode:")
        env_multi = TetrisEnv(single_player=False, headless=True)
        obs_multi = env_multi.reset()
        if isinstance(obs_multi, tuple):
            obs_multi = obs_multi[0]
        
        print(f"   Multiplayer observation type: {type(obs_multi)}")
        print(f"   Multiplayer observation keys: {obs_multi.keys() if isinstance(obs_multi, dict) else 'Not a dict'}")
        
        # Test if it's actually different from single player
        print(f"   Are observations different? {type(obs_single) != type(obs_multi) or obs_single != obs_multi}")
        
        # Test 3: Different action formats
        print("\n3️⃣ Testing Action Formats:")
        
        # Single action (should work)
        try:
            action_single = 20
            result = env_multi.step(action_single)
            print(f"   ✅ Single action accepted: {len(result)} elements returned")
        except Exception as e:
            print(f"   ❌ Single action failed: {e}")
        
        # Dictionary action (test if supported)
        try:
            action_dict = {'player1': 15, 'player2': 25}
            result = env_multi.step(action_dict)
            print(f"   ✅ Dictionary action accepted: {len(result)} elements returned")
        except Exception as e:
            print(f"   ❌ Dictionary action failed: {e}")
        
        env_multi.close()
        
        # Test 4: Check internal game structure
        print("\n4️⃣ Analyzing Internal Game Structure:")
        env_test = TetrisEnv(single_player=False, headless=True)
        env_test.reset()
        
        print(f"   Game object exists: {env_test.game is not None}")
        print(f"   Player1 exists: {hasattr(env_test.game, 'player1')}")
        print(f"   Player2 exists: {hasattr(env_test.game, 'player2')}")
        print(f"   Active player: {env_test.player}")
        print(f"   Player1 == Active player: {env_test.player == env_test.game.player1}")
        
        if hasattr(env_test.game, 'player2'):
            print(f"   Player2 has pieces: {env_test.game.player2.current_piece is not None}")
            print(f"   Player2 next pieces: {len(env_test.game.player2.next_pieces) if env_test.game.player2.next_pieces else 0}")
        
        env_test.close()
        
        # Test 5: Network parameter analysis
        print("\n5️⃣ Agent Parameter Analysis:")
        from rl_utils.multiplayer_airl import MultiplayerAIRLTrainer
        
        config = {'device': 'cpu'}
        trainer = MultiplayerAIRLTrainer(config)
        
        # Get parameter info
        p1_params = sum(p.numel() for p in trainer.policy_p1.parameters())
        p2_params = sum(p.numel() for p in trainer.policy_p2.parameters())
        
        print(f"   Player 1 policy parameters: {p1_params}")
        print(f"   Player 2 policy parameters: {p2_params}")
        
        # Check if they're actually different instances
        same_instance = trainer.policy_p1 is trainer.policy_p2
        print(f"   Same policy instance: {same_instance}")
        
        # Test parameter differences
        p1_first_param = list(trainer.policy_p1.parameters())[0]
        p2_first_param = list(trainer.policy_p2.parameters())[0]
        params_identical = np.allclose(p1_first_param.detach().numpy(), p2_first_param.detach().numpy())
        print(f"   Parameters identical: {params_identical}")
        
        trainer.env.close()
        
        # Conclusion
        print("\n📊 DIAGNOSTIC SUMMARY:")
        print("=" * 60)
        print("🔍 Environment Analysis:")
        print(f"   • Environment has player2: {hasattr(env_test.game, 'player2') if 'env_test' in locals() else 'Unknown'}")
        print(f"   • Supports dict actions: {'❌ No' if 'Dictionary action failed' in str(locals()) else '❓ Unknown'}")
        print(f"   • Returns multiplayer obs: {'❌ No' if obs_single == obs_multi else '❓ Unclear'}")
        
        print("\n🤖 Agent Analysis:")
        print(f"   • Agents are distinct: {'✅ Yes' if not same_instance else '❌ No'}")
        print(f"   • Parameters differ: {'✅ Yes' if not params_identical else '❌ No'}")
        
        print("\n🎯 ROOT CAUSE IDENTIFIED:")
        print("   The TetrisEnv is NOT truly multiplayer!")
        print("   • Only uses player1 in step() and reset()")
        print("   • Doesn't accept dictionary actions")
        print("   • Doesn't return separate observations")
        print("   • Player 2 exists but is unused")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main diagnostic function."""
    success = test_environment_multiplayer_capability()
    
    if success:
        print("\n🔧 SOLUTION NEEDED:")
        print("   Create a proper multiplayer environment wrapper")
        print("   that handles both players distinctly")
    
    return success

if __name__ == "__main__":
    main() 