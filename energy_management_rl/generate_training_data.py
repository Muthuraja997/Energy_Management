import numpy as np
import pandas as pd
import os
import pickle
from env import EnergyManagementEnv
from tqdm import tqdm
import random

# Action names for reference
ACTION_NAMES = ['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar + Battery']

def rule_based_policy(state):
    """
    Rule-based policy to determine the best energy source.
    This will be used to generate training data for the ML model.
    
    Parameters:
    - state: Numpy array containing [demand, solar, battery_soc, price, time_of_day]
    
    Returns:
    - action: The recommended action (0-3)
    """
    demand, solar, battery_soc, price, time_of_day = state
    
    # First priority: Use solar if available
    solar_available = solar > 0
    
    # Second priority: Use battery if SOC is good and price is high
    battery_usable = battery_soc > 20  # Don't go below 20% SOC
    price_is_high = price > 0.25
    
    # Calculate how much energy we need
    energy_needed = demand
    
    # Decision making
    if solar_available and solar >= demand:
        # Solar can meet all demand
        return 0  # Use solar
    elif solar_available and battery_usable and solar + (battery_soc/100 * 10) >= demand:
        # Solar + battery can meet demand
        if price_is_high or (time_of_day >= 18 or time_of_day <= 6):
            # High price or evening/night: use solar + battery
            return 3  # Use solar + battery
        else:
            # Use solar and save battery
            return 0  # Use solar
    elif battery_usable and (battery_soc/100 * 10) >= demand and (price_is_high or solar < 0.5):
        # Battery can meet demand and price is high or solar is low
        return 1  # Use battery
    elif solar_available and solar > 0.5:
        # Some solar available, but not enough - use solar anyway to offset grid
        return 0  # Use solar
    else:
        # Default to grid if other options not viable
        return 2  # Use grid

def generate_training_data(num_samples=10000):
    """
    Generate training data for the ML model by simulating the environment
    and using the rule-based policy to make decisions.
    
    Parameters:
    - num_samples: Number of samples to generate
    
    Returns:
    - pandas DataFrame containing states and actions
    """
    print(f"Generating {num_samples} samples...")
    
    # Create environment
    env = EnergyManagementEnv()
    
    # Lists to store data
    states = []
    actions = []
    
    # Generate data
    for _ in tqdm(range(num_samples)):
        # Reset environment to get a random state
        obs = env.reset()
        if isinstance(obs, tuple):
            state = obs[0]  # In newer gym, reset returns (state, info)
        else:
            state = obs
        
        # Sometimes create custom states to ensure all actions are represented
        if random.random() < 0.3:
            # Generate custom state that would lead to different actions
            if random.random() < 0.25:
                # State that favors battery usage (high price, low solar, good battery)
                state = np.array([
                    random.uniform(2.0, 5.0),  # moderate demand
                    random.uniform(0.0, 1.0),  # low solar
                    random.uniform(50.0, 90.0),  # high battery SOC
                    random.uniform(0.3, 0.5),  # high price
                    random.uniform(18.0, 22.0)  # evening time
                ])
                action = 1  # Battery
            elif random.random() < 0.5:
                # State that favors grid usage (high demand, low solar, low battery)
                state = np.array([
                    random.uniform(8.0, 15.0),  # high demand
                    random.uniform(0.0, 1.0),  # low solar
                    random.uniform(0.0, 30.0),  # low battery SOC
                    random.uniform(0.1, 0.2),  # low price
                    random.uniform(0.0, 23.0)  # any time
                ])
                action = 2  # Grid
            elif random.random() < 0.75:
                # State that favors solar+battery (medium demand, medium solar, good battery)
                state = np.array([
                    random.uniform(5.0, 8.0),  # medium-high demand
                    random.uniform(3.0, 5.0),  # medium solar
                    random.uniform(40.0, 90.0),  # good battery SOC
                    random.uniform(0.3, 0.5),  # high price
                    random.uniform(10.0, 14.0)  # daytime
                ])
                action = 3  # Solar + Battery
            else:
                # State that favors solar (low demand, high solar)
                state = np.array([
                    random.uniform(1.0, 3.0),  # low demand
                    random.uniform(5.0, 10.0),  # high solar
                    random.uniform(20.0, 90.0),  # any battery SOC
                    random.uniform(0.1, 0.3),  # any price
                    random.uniform(10.0, 14.0)  # daytime
                ])
                action = 0  # Solar
        else:
            # Get action from rule-based policy for random state
            action = rule_based_policy(state)
        
        # Store state and action
        states.append(state)
        actions.append(action)
        
    # Create DataFrame
    df = pd.DataFrame(states, columns=['demand', 'solar', 'battery_soc', 'price', 'time_of_day'])
    df['action'] = actions
    
    # Check if all actions are represented
    action_counts = df['action'].value_counts()
    print("Action distribution:")
    for action in range(4):
        count = action_counts.get(action, 0)
        print(f"Action {action} ({ACTION_NAMES[action]}): {count} samples ({count/num_samples*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate training data
    df = generate_training_data(10000)
    
    # Save to CSV
    df.to_csv("data/xgboost_training_data.csv", index=False)
    print("Training data saved to 'data/xgboost_training_data.csv'")
