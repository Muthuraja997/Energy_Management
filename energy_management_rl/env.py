import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnergyManagementEnv(gym.Env):
    """
    Custom Gymnasium environment for energy management system.
    
    State space:
    - current demand (kW)
    - solar generation (kW)
    - battery SOC (%)
    - electricity price ($/kWh)
    - time of day (hour)
    
    Actions:
    - 0: use solar
    - 1: use battery
    - 2: use grid
    - 3: use solar + battery
    
    Reward:
    - Negative reward for electricity cost if grid is used
    - Positive reward for using solar
    - Penalty for draining battery below 20%
    
    Episode:
    - One day of operation (24 hours in discrete time steps)
    """
    
    def __init__(self):
        super(EnergyManagementEnv, self).__init__()
        
        # Define action and observation space
        # Action space: 0 = use solar, 1 = use battery, 2 = use grid, 3 = use solar + battery
        self.action_space = spaces.Discrete(4)
        
        # Observation space
        # [current demand (kW), solar generation (kW), battery SOC (%), electricity price ($/kWh), time of day]
        
        # Define bounds for each observation
        self.demand_low, self.demand_high = 0.0, 20.0  # kW
        self.solar_low, self.solar_high = 0.0, 15.0  # kW
        self.battery_soc_low, self.battery_soc_high = 0.0, 100.0  # %
        self.price_low, self.price_high = 0.05, 0.5  # $/kWh
        self.time_low, self.time_high = 0, 23  # hour of day
        
        self.observation_space = spaces.Box(
            low=np.array([self.demand_low, self.solar_low, self.battery_soc_low, self.price_low, self.time_low]),
            high=np.array([self.demand_high, self.solar_high, self.battery_soc_high, self.price_high, self.time_high]),
            dtype=np.float32
        )
        
        # Environment parameters
        self.battery_capacity = 30.0  # kWh
        self.battery_max_discharge_rate = 5.0  # kW
        self.battery_efficiency = 0.9  # 90% efficiency
        self.min_battery_soc = 20.0  # minimum battery state of charge (%)
        
        # Initialize state
        self.current_step = 0
        self.max_steps = 24  # 24 hours in a day
        self.state = None
        
        # Load sample data profiles (in real application, this would be actual data)
        self.demand_profile = self._generate_sample_demand_profile()
        self.solar_profile = self._generate_sample_solar_profile()
        self.price_profile = self._generate_sample_price_profile()
    
    def _generate_sample_demand_profile(self):
        """Generate a sample 24-hour demand profile."""
        # Morning peak (7-9 AM), evening peak (6-10 PM)
        base_demand = np.ones(24) * 2.0
        morning_peak = np.zeros(24)
        morning_peak[7:10] = 3.0
        evening_peak = np.zeros(24)
        evening_peak[18:22] = 5.0
        
        profile = base_demand + morning_peak + evening_peak
        # Add some randomness
        profile += np.random.normal(0, 0.5, 24)
        return np.clip(profile, self.demand_low, self.demand_high)
    
    def _generate_sample_solar_profile(self):
        """Generate a sample 24-hour solar generation profile."""
        # Solar generation between 6 AM and 6 PM with peak at noon
        profile = np.zeros(24)
        for i in range(6, 19):
            profile[i] = 10.0 * np.sin(np.pi * (i - 6) / 12)
        
        # Add some randomness (clouds etc.)
        profile += np.random.normal(0, 1.0, 24)
        return np.clip(profile, self.solar_low, self.solar_high)
    
    def _generate_sample_price_profile(self):
        """Generate a sample 24-hour electricity price profile."""
        # Higher prices during peak demand hours
        base_price = np.ones(24) * 0.1
        peak_price = np.zeros(24)
        # Morning peak (7-9 AM)
        peak_price[7:10] = 0.1
        # Evening peak (6-10 PM)
        peak_price[18:22] = 0.2
        
        profile = base_price + peak_price
        # Add some randomness
        profile += np.random.normal(0, 0.02, 24)
        return np.clip(profile, self.price_low, self.price_high)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize battery at 50% SOC
        self.battery_soc = 50.0
        
        # Set initial state
        self.state = np.array([
            self.demand_profile[self.current_step],
            self.solar_profile[self.current_step],
            self.battery_soc,
            self.price_profile[self.current_step],
            self.current_step
        ], dtype=np.float32)
        
        # Return the initial state and info
        info = {}
        return self.state, info
    
    def step(self, action):
        """
        Take a step in the environment given an action.
        
        Args:
            action (int): 
                0 = use solar
                1 = use battery
                2 = use grid
                3 = use solar + battery
        
        Returns:
            state, reward, terminated, truncated, info
        """
        # Extract current state values
        current_demand = self.state[0]
        available_solar = self.state[1]
        current_battery_soc = self.state[2]
        current_price = self.state[3]
        
        # Available battery energy in kWh
        available_battery_energy = (current_battery_soc / 100.0) * self.battery_capacity
        # Maximum energy that can be discharged this step (1 hour)
        max_battery_discharge = min(self.battery_max_discharge_rate, available_battery_energy * self.battery_efficiency)
        
        # Initialize values
        solar_used = 0.0
        battery_used = 0.0
        grid_used = 0.0
        
        # Implement different actions
        if action == 0:  # Use solar only
            solar_used = min(available_solar, current_demand)
            grid_used = max(0, current_demand - solar_used)
        
        elif action == 1:  # Use battery only
            battery_used = min(max_battery_discharge, current_demand)
            grid_used = max(0, current_demand - battery_used)
        
        elif action == 2:  # Use grid only
            grid_used = current_demand
        
        elif action == 3:  # Use solar + battery
            solar_used = min(available_solar, current_demand)
            remaining_demand = current_demand - solar_used
            battery_used = min(max_battery_discharge, remaining_demand)
            grid_used = max(0, remaining_demand - battery_used)
        
        # Update battery SOC
        energy_from_battery = battery_used / self.battery_efficiency  # Account for battery efficiency
        new_battery_energy = available_battery_energy - energy_from_battery
        new_battery_soc = (new_battery_energy / self.battery_capacity) * 100.0
        
        # Calculate reward
        reward = 0.0
        
        # Cost of electricity from the grid
        grid_cost = grid_used * current_price
        reward -= grid_cost
        
        # Reward for using solar
        reward += solar_used * 0.1
        
        # Penalty for draining battery below threshold
        if new_battery_soc < self.min_battery_soc:
            reward -= 5.0  # Significant penalty for breaking the constraint
        
        # Update the battery SOC
        self.battery_soc = new_battery_soc
        
        # Move to the next time step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # If not done, update the state
        if not terminated:
            self.state = np.array([
                self.demand_profile[self.current_step],
                self.solar_profile[self.current_step],
                self.battery_soc,
                self.price_profile[self.current_step],
                self.current_step
            ], dtype=np.float32)
        
        # Additional info
        info = {
            'solar_used': solar_used,
            'battery_used': battery_used,
            'grid_used': grid_used,
            'grid_cost': grid_cost,
            'battery_soc': self.battery_soc
        }
        
        return self.state, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        print(f"Step: {self.current_step}")
        print(f"Demand: {self.state[0]:.2f} kW")
        print(f"Solar: {self.state[1]:.2f} kW")
        print(f"Battery SOC: {self.state[2]:.2f}%")
        print(f"Price: ${self.state[3]:.2f}/kWh")
        print(f"Time: {int(self.state[4])}:00")
    
    def close(self):
        """Close the environment."""
        pass
