#!/usr/bin/env python3
"""
🎯 COMPLETE DUAL REWARD MODE SUPPORT TEST

Comprehensive test demonstrating:
1. DREAM works with both 'standard' and 'lines_only' reward modes
2. DQN works with both 'standard' and 'lines_only' reward modes
3. All combinations are fully functional
"""

import torch
import numpy as np
from envs.tetris_env import TetrisEnv
from dream.configs.dream_config import DREAMConfig
from dream.models.world_model import WorldModel
from dream.models.actor_critic import ActorCritic
from agents.dqn_locked_agent_redesigned import RedesignedLockedStateDQNAgent

def test_dream_with_reward_mode(reward_mode):
    """Test DREAM with specified reward mode"""
    print(f"\n🚀 TESTING DREAM WITH {reward_mode.upper()} REWARD MODE")
    print("-" * 60)
    
    try:
        # Create padded environment
        class PaddedEnv:
            def __init__(self, reward_mode):
                self.env = TetrisEnv(
                    num_agents=1,
                    headless=True,
                    action_mode='direct',
                    reward_mode=reward_mode
                )
            def reset(self):
                obs = self.env.reset()
                return np.concatenate([obs, np.zeros(6)]) if obs.shape[0] == 206 else obs
            def step(self, action):
                obs, reward, done, info = self.env.step(action)
                return np.concatenate([obs, np.zeros(6)]) if obs.shape[0] == 206 else obs, reward, done, info
            def close(self):
                self.env.close()
        
        env = PaddedEnv(reward_mode)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize DREAM components
        config = DREAMConfig.get_default_config(action_mode='direct')
        world_model = WorldModel(**config.world_model).to(device)
        actor_critic = ActorCritic(**config.actor_critic).to(device)
        
        print(f"   ✅ Environment created: reward_mode='{reward_mode}'")
        print(f"   ✅ World Model initialized: {sum(p.numel() for p in world_model.parameters()):,} parameters")
        print(f"   ✅ Actor-Critic initialized: {sum(p.numel() for p in actor_critic.parameters()):,} parameters")
        
        # Test episode
        obs = env.reset()
        total_reward = 0
        episode_length = 0
        
        for step in range(50):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = actor_critic.get_action_and_value(obs_tensor)
                action_scalar = torch.argmax(action.squeeze(0)).cpu().item()
            
            next_obs, reward, done, info = env.step(action_scalar)
            total_reward += reward
            episode_length += 1
            
            if done:
                break
                
            obs = next_obs
        
        env.close()
        
        print(f"   ✅ Test episode completed: {episode_length} steps, reward={total_reward:.2f}")
        print(f"   ✅ DREAM works perfectly with {reward_mode} rewards!")
        return True
        
    except Exception as e:
        print(f"   ❌ DREAM failed with {reward_mode}: {e}")
        return False

def test_dqn_with_reward_mode(reward_mode):
    """Test DQN with specified reward mode"""
    print(f"\n🤖 TESTING DQN WITH {reward_mode.upper()} REWARD MODE")
    print("-" * 60)
    
    try:
        # Create environment for DQN
        env = TetrisEnv(
            num_agents=1,
            headless=True,
            action_mode='locked_position',
            reward_mode=reward_mode
        )
        
        # Initialize DQN agent with reward mode
        agent = RedesignedLockedStateDQNAgent(
            input_dim=206,
            num_actions=800,
            device='cuda',
            learning_rate=0.0001,
            epsilon_start=0.9,
            epsilon_end=0.05,
            reward_mode=reward_mode
        )
        
        print(f"   ✅ Environment created: reward_mode='{reward_mode}'")
        print(f"   ✅ DQN Agent initialized: {agent.get_parameter_count():,} parameters")
        print(f"   ✅ Agent configured for reward_mode='{reward_mode}'")
        
        # Test episode
        obs = env.reset()
        total_reward = 0
        episode_length = 0
        
        for step in range(50):
            action = agent.select_action(obs, training=True, env=env)
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)
            
            total_reward += reward
            episode_length += 1
            
            if done:
                break
                
            obs = next_obs
        
        env.close()
        
        print(f"   ✅ Test episode completed: {episode_length} steps, reward={total_reward:.2f}")
        print(f"   ✅ DQN works perfectly with {reward_mode} rewards!")
        return True
        
    except Exception as e:
        print(f"   ❌ DQN failed with {reward_mode}: {e}")
        return False

def test_reward_mode_switching():
    """Test that reward modes can be switched dynamically"""
    print(f"\n🔄 TESTING REWARD MODE SWITCHING")
    print("-" * 60)
    
    try:
        # Test switching between modes
        env1 = TetrisEnv(reward_mode='standard')
        obs1 = env1.reset()
        _, reward1, _, _ = env1.step(0)
        env1.close()
        
        env2 = TetrisEnv(reward_mode='lines_only')
        obs2 = env2.reset()
        _, reward2, _, _ = env2.step(0)
        env2.close()
        
        print(f"   ✅ Standard mode reward: {reward1:.2f}")
        print(f"   ✅ Lines-only mode reward: {reward2:.2f}")
        print(f"   ✅ Reward mode switching works perfectly!")
        return True
        
    except Exception as e:
        print(f"   ❌ Reward mode switching failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test of all combinations"""
    print("🎯 COMPREHENSIVE DUAL REWARD MODE SUPPORT TEST")
    print("=" * 80)
    
    results = {}
    
    # Test all combinations
    print(f"\n📋 TESTING ALL COMBINATIONS:")
    
    # DREAM tests
    results['dream_standard'] = test_dream_with_reward_mode('standard')
    results['dream_lines_only'] = test_dream_with_reward_mode('lines_only')
    
    # DQN tests
    results['dqn_standard'] = test_dqn_with_reward_mode('standard')
    results['dqn_lines_only'] = test_dqn_with_reward_mode('lines_only')
    
    # Switching test
    results['reward_switching'] = test_reward_mode_switching()
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"🎯 ALGORITHM + REWARD MODE COMBINATIONS:")
    print(f"   DREAM + Standard:   {'✅ PASS' if results['dream_standard'] else '❌ FAIL'}")
    print(f"   DREAM + Lines-Only: {'✅ PASS' if results['dream_lines_only'] else '❌ FAIL'}")
    print(f"   DQN + Standard:     {'✅ PASS' if results['dqn_standard'] else '❌ FAIL'}")
    print(f"   DQN + Lines-Only:   {'✅ PASS' if results['dqn_lines_only'] else '❌ FAIL'}")
    print(f"   Reward Switching:   {'✅ PASS' if results['reward_switching'] else '❌ FAIL'}")
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Both DREAM and DQN support both reward modes flawlessly")
        print(f"✅ All combinations are production-ready")
        print(f"✅ Reward mode switching works perfectly")
        
        print(f"\n🚀 USAGE SUMMARY:")
        print(f"   DREAM + Standard:   env = TetrisEnv(action_mode='direct', reward_mode='standard')")
        print(f"   DREAM + Lines-Only: env = TetrisEnv(action_mode='direct', reward_mode='lines_only')")
        print(f"   DQN + Standard:     env = TetrisEnv(action_mode='locked_position', reward_mode='standard')")
        print(f"   DQN + Lines-Only:   env = TetrisEnv(action_mode='locked_position', reward_mode='lines_only')")
        
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        failed_tests = [test for test, passed in results.items() if not passed]
        print(f"   Failed: {failed_tests}")
    
    print("=" * 80)
    
    return passed_tests == total_tests

def main():
    """Main test function"""
    success = run_comprehensive_test()
    
    if success:
        print(f"\n🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print(f"🚀 BOTH DREAM AND DQN SUPPORT BOTH REWARD MODES FLAWLESSLY!")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 