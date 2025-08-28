# Energy Management System with Reinforcement Learning

This project implements a DQN (Deep Q-Network) reinforcement learning agent using Stable-Baselines3 for optimizing energy management decisions. The agent learns to make decisions about whether to use solar power, battery power, or grid power based on current conditions.

The project now includes an interactive Streamlit dashboard for visualizing the agent's behavior in real-time!

## Environment Details

The custom Gym environment models a household energy management system with:

- **State space**: [current demand (kW), solar generation (kW), battery SOC (%), electricity price ($/kWh), time of day]
- **Actions**: 
  - 0 = use solar
  - 1 = use battery
  - 2 = use grid
  - 3 = use solar + battery
- **Reward function**: 
  - Negative reward for electricity cost when grid is used
  - Positive reward for using solar energy
  - Penalty for draining battery below 20% SOC
- **Episode**: One day of operation (24 hours in discrete time steps)

## Project Structure

- `env.py`: Custom Gymnasium environment for the energy management system
- `train.py`: Script for training the DQN agent
- `inference.py`: Script for running and visualizing the trained agent
- `test_env.py` and `test_env_minimal.py`: Scripts to test the environment
- `run.py`: Wrapper script to run the entire pipeline
- `setup.py`: Script to install dependencies
- `prediction.py`: Module for making predictions with the trained model
- `dashboard/`: Interactive Streamlit dashboard for visualizing the agent's behavior
- `dashboard/app.py`: The main Streamlit dashboard
- `dashboard/predictor.py`: Standalone predictor application
- `dashboard/demo_app.py`: Demo version of the dashboard using rule-based logic
- `dashboard/demo_predictor.py`: Demo version of the predictor using rule-based logic
- `requirements.txt`: Dependencies required for the project

## Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python setup.py
# Or install directly with pip
pip install -r requirements.txt
```

## Usage

### Training the Agent

```bash
python train.py --timesteps 100000 --learning_rate 1e-4 --gamma 0.99
```

This will:
1. Create the custom environment
2. Train a DQN agent
3. Save model checkpoints and the best model
4. Create training logs and learning curves

### Running Inference

```bash
python inference.py --model_path "models/best/best_model.zip" --episodes 3 --render
```

This will:
1. Load the trained model
2. Run the agent on the environment for multiple episodes
3. Display and visualize the agent's performance
4. Calculate metrics such as energy usage, costs, and rewards

### Testing the Environment

```bash
python test_env_minimal.py
```

This will run basic tests on the environment to ensure it's working correctly.

### Running the Complete Pipeline

```bash
python run.py --mode all --timesteps 100000 --episodes 3 --render
```

This will run through the entire pipeline from testing the environment to training the agent to running inference.

## Interactive Dashboard

This project includes an interactive Streamlit dashboard for visualizing the agent's behavior and monitoring its performance in real-time.

### Running the Dashboard

Due to NumPy compatibility issues, there are two versions of the dashboard:

#### Full Dashboard (requires compatible NumPy version)

```bash
# Run the full dashboard with trained DQN model
python -m streamlit run dashboard/app.py

# OR use the batch file
run_dashboard.bat
```

#### Demo Dashboard (works with any NumPy version)

```bash
# Run the demo dashboard with rule-based policy
python -m streamlit run dashboard/demo_app.py

# OR use the batch file
run_demo_dashboard.bat
```

### Using the Predictor

The predictor allows you to input a system state and get the recommended energy source.

#### Full Predictor (requires compatible NumPy version)

```bash
# Run the predictor with trained DQN model
python -m streamlit run dashboard/predictor.py

# OR use the batch file
run_predictor.bat
```

#### Demo Predictor (works with any NumPy version)

```bash
# Run the demo predictor with rule-based policy
python -m streamlit run dashboard/demo_predictor.py

# OR use the batch file
run_demo_predictor.bat
```

### Dashboard Features

- **Real-time Visualization**: See the current state of the environment and the agent's actions.
- **Interactive Charts**: Monitor demand, solar generation, battery state of charge, and more.
- **Simulation Controls**: Step through a simulation one step at a time or run a complete episode.
- **Flexible Model Loading**: Load different trained models to compare their performance.
- **Detailed Results**: View comprehensive statistics and metrics about the agent's performance.

## Customization

You can modify parameters in the following ways:

- Adjust the training parameters in `train.py` (e.g., learning rate, exploration rate)
- Modify the environment parameters in `env.py` (e.g., battery capacity, solar generation profile)
- Change the number of episodes and visualization options in `inference.py`

## Requirements

- Python 3.7+
- Gymnasium
- Stable-Baselines3
- NumPy
- Streamlit
- Pandas
- Matplotlib
- Plotly

## NumPy Compatibility Issue

The project was developed with NumPy 1.x, but you're currently using NumPy 2.3.2. Some dependencies like TensorFlow and Stable-Baselines3 require NumPy 1.x. To fix this, you can:

1. Downgrade NumPy: `pip install numpy<2`
2. Use the demo versions of the dashboard and predictor (they don't depend on Stable-Baselines3)
3. Create a separate virtual environment with compatible dependencies

## Predictor Tool

The predictor tool allows you to:

1. Input the current state of your system (demand, solar, battery SOC, price, time of day)
2. Get a prediction of the best energy source to use
3. View the Q-values and confidence levels for each action
4. Track your prediction history

This is useful for real-time decision support or for understanding how the agent makes decisions in different scenarios.
- Matplotlib
- Streamlit (for dashboard)
- Pandas (for dashboard)
- Plotly (for dashboard)
