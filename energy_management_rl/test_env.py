import numpy as np
import matplotlib.pyplot as plt
from env import EnergyManagementEnv


def run_random_agent(env, episodes=1):
    """
    Run a random agent on the environment to test it.
    
    Args:
        env: The environment to run on
        episodes: Number of episodes to run
    """
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        print(f"Episode {episode+1}/{episodes}")
        
        while not (done or truncated):
            # Choose a random action
            action = env.action_space.sample()
            
            # Take the action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Print information
            env.render()
            print(f"Action taken: {['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar+Battery'][action]}")
            print(f"Solar used: {info['solar_used']:.2f} kW")
            print(f"Battery used: {info['battery_used']:.2f} kW")
            print(f"Grid used: {info['grid_used']:.2f} kW")
            print(f"Grid cost: ${info['grid_cost']:.2f}")
            print(f"New battery SOC: {info['battery_soc']:.2f}%")
            print(f"Reward: {reward:.2f}")
            print("-" * 50)
            
            # Update total reward
            total_reward += reward
            
            # Update state
            state = next_state
        
        print(f"Episode {episode+1} completed with total reward: {total_reward:.2f}")
        print("=" * 50)


def plot_environment_profiles():
    """Plot the sample profiles used in the environment."""
    env = EnergyManagementEnv()
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot demand profile
    axs[0].plot(range(24), env.demand_profile, 'b-o')
    axs[0].set_xlabel('Hour of Day')
    axs[0].set_ylabel('Demand (kW)')
    axs[0].set_title('Daily Demand Profile')
    axs[0].grid(True)
    
    # Plot solar profile
    axs[1].plot(range(24), env.solar_profile, 'g-o')
    axs[1].set_xlabel('Hour of Day')
    axs[1].set_ylabel('Solar Generation (kW)')
    axs[1].set_title('Daily Solar Generation Profile')
    axs[1].grid(True)
    
    # Plot price profile
    axs[2].plot(range(24), env.price_profile, 'r-o')
    axs[2].set_xlabel('Hour of Day')
    axs[2].set_ylabel('Price ($/kWh)')
    axs[2].set_title('Daily Electricity Price Profile')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("environment_profiles.png")
    plt.show()


if __name__ == "__main__":
    # Create the environment
    env = EnergyManagementEnv()
    
    # Plot the environment profiles
    plot_environment_profiles()
    
    # Run a random agent to test the environment
    run_random_agent(env, episodes=1)
