import numpy as np
from stable_baselines3 import DQN
import matplotlib.pyplot as plt

from env import EnergyManagementEnv

def load_model(model_path):
    """
    Load a trained DQN model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    try:
        model = DQN.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def run_inference(model, env, episodes=5, render=True):
    """
    Run inference with the trained model.
    
    Args:
        model: Trained model
        env: Environment to run inference on
        episodes: Number of episodes to run
        render: Whether to render the environment
        
    Returns:
        Dictionary of metrics
    """
    episode_rewards = []
    episode_costs = []
    episode_solar_usage = []
    episode_battery_usage = []
    episode_grid_usage = []
    episode_battery_soc = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        
        episode_reward = 0
        episode_cost = 0
        episode_solar = 0
        episode_battery = 0
        episode_grid = 0
        episode_soc = []
        
        done = False
        truncated = False
        
        print(f"Episode {episode+1}/{episodes}")
        
        while not (done or truncated):
            # Get action from model
            action, _states = model.predict(state, deterministic=True)
            
            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Render if specified
            if render:
                env.render()
                print(f"Action taken: {['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar+Battery'][action]}")
                print(f"Solar used: {info['solar_used']:.2f} kW")
                print(f"Battery used: {info['battery_used']:.2f} kW")
                print(f"Grid used: {info['grid_used']:.2f} kW")
                print(f"Grid cost: ${info['grid_cost']:.2f}")
                print(f"New battery SOC: {info['battery_soc']:.2f}%")
                print(f"Reward: {reward:.2f}")
                print("-" * 50)
            
            # Update metrics
            episode_reward += reward
            episode_cost += info['grid_cost']
            episode_solar += info['solar_used']
            episode_battery += info['battery_used']
            episode_grid += info['grid_used']
            episode_soc.append(info['battery_soc'])
            
            # Update state
            state = next_state
        
        # Collect episode statistics
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_solar_usage.append(episode_solar)
        episode_battery_usage.append(episode_battery)
        episode_grid_usage.append(episode_grid)
        episode_battery_soc.append(episode_soc)
        
        print(f"Episode {episode+1} completed with total reward: {episode_reward:.2f}")
        print(f"Total electricity cost: ${episode_cost:.2f}")
        print(f"Total solar used: {episode_solar:.2f} kWh")
        print(f"Total battery used: {episode_battery:.2f} kWh")
        print(f"Total grid used: {episode_grid:.2f} kWh")
        print("=" * 50)
    
    # Calculate averages
    avg_reward = np.mean(episode_rewards)
    avg_cost = np.mean(episode_costs)
    avg_solar = np.mean(episode_solar_usage)
    avg_battery = np.mean(episode_battery_usage)
    avg_grid = np.mean(episode_grid_usage)
    
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Average electricity cost: ${avg_cost:.2f}")
    print(f"Average solar usage: {avg_solar:.2f} kWh")
    print(f"Average battery usage: {avg_battery:.2f} kWh")
    print(f"Average grid usage: {avg_grid:.2f} kWh")
    
    # Return metrics
    return {
        'rewards': episode_rewards,
        'costs': episode_costs,
        'solar_usage': episode_solar_usage,
        'battery_usage': episode_battery_usage,
        'grid_usage': episode_grid_usage,
        'battery_soc': episode_battery_soc
    }


def plot_metrics(metrics, save_path=None):
    """
    Plot metrics from inference.
    
    Args:
        metrics: Dictionary of metrics from run_inference
        save_path: Path to save the plots
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot energy sources
    axs[0, 0].bar(range(len(metrics['solar_usage'])), metrics['solar_usage'], label='Solar')
    axs[0, 0].bar(range(len(metrics['battery_usage'])), metrics['battery_usage'], 
             bottom=metrics['solar_usage'], label='Battery')
    axs[0, 0].bar(range(len(metrics['grid_usage'])), metrics['grid_usage'], 
             bottom=np.array(metrics['solar_usage']) + np.array(metrics['battery_usage']), label='Grid')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Energy (kWh)')
    axs[0, 0].set_title('Energy Sources by Episode')
    axs[0, 0].legend()
    
    # Plot costs
    axs[0, 1].plot(metrics['costs'], marker='o')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Cost ($)')
    axs[0, 1].set_title('Electricity Cost by Episode')
    axs[0, 1].grid(True)
    
    # Plot rewards
    axs[1, 0].plot(metrics['rewards'], marker='o')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Reward')
    axs[1, 0].set_title('Rewards by Episode')
    axs[1, 0].grid(True)
    
    # Plot battery SOC for the last episode
    axs[1, 1].plot(metrics['battery_soc'][-1], marker='.')
    axs[1, 1].axhline(y=20, color='r', linestyle='--', label='Min SOC Threshold')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Battery SOC (%)')
    axs[1, 1].set_title('Battery SOC (Last Episode)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plots saved to {save_path}")
    
    plt.show()


def example_usage():
    """Example of how to use the inference code."""
    # Path to the trained model
    model_path = "models/best/best_model.zip"
    
    # Create environment
    env = EnergyManagementEnv()
    
    # Load model
    model = load_model(model_path)
    
    if model:
        # Run inference
        metrics = run_inference(model, env, episodes=3, render=True)
        
        # Plot metrics
        plot_metrics(metrics, save_path="inference_results.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with trained DQN agent')
    parser.add_argument('--model_path', type=str, default="models/best/best_model.zip", 
                        help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=3, 
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', 
                        help='Whether to render the environment')
    parser.add_argument('--output', type=str, default="inference_results.png", 
                        help='Path to save the output plots')
    args = parser.parse_args()
    
    # Create environment
    env = EnergyManagementEnv()
    
    # Load model
    model = load_model(args.model_path)
    
    if model:
        # Run inference
        metrics = run_inference(model, env, episodes=args.episodes, render=args.render)
        
        # Plot metrics
        plot_metrics(metrics, save_path=args.output)
