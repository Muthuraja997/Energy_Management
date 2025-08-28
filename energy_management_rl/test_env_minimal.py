import numpy as np
from env import EnergyManagementEnv


def test_environment():
    """Test the basic functionality of the environment."""
    print("Testing EnergyManagementEnv...")
    
    # Create the environment
    env = EnergyManagementEnv()
    
    # Test reset
    state, _ = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state values: {state}")
    
    # Verify observation space
    assert env.observation_space.contains(state), "Initial state is not in observation space!"
    
    # Test step with each action
    for action in range(env.action_space.n):
        state, _ = env.reset()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nAction {action} ({['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar+Battery'][action]}):")
        print(f"Reward: {reward}")
        print(f"Solar used: {info['solar_used']:.2f} kW")
        print(f"Battery used: {info['battery_used']:.2f} kW")
        print(f"Grid used: {info['grid_used']:.2f} kW")
        print(f"Grid cost: ${info['grid_cost']:.2f}")
        print(f"Battery SOC: {info['battery_soc']:.2f}%")
        
        # Verify that next_state is valid
        assert env.observation_space.contains(next_state), f"Next state for action {action} is not in observation space!"
    
    # Test a full episode
    print("\nRunning a full episode with random actions...")
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    while not (done or truncated):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        state = next_state
    
    print(f"Episode completed after {step_count} steps with total reward: {total_reward:.2f}")
    print("Environment test completed successfully!")


if __name__ == "__main__":
    test_environment()
