import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import sys
import tempfile
from PIL import Image
from pathlib import Path
import pickle

# Add parent directory to path so we can import our modules
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import our environment and DQN components
from env import EnergyManagementEnv
from stable_baselines3 import DQN


# Page configuration
st.set_page_config(
    page_title="Energy Management RL Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model paths
MODEL_PATHS = {
    "DQN": "models/final_model.zip",
    "Random Forest": "models/random_forest_model.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
        border: none;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .header-text {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Session state initialization
if 'env' not in st.session_state:
    st.session_state.env = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = "DQN"
if 'state' not in st.session_state:
    st.session_state.state = None
if 'episode_history' not in st.session_state:
    st.session_state.episode_history = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'done' not in st.session_state:
    st.session_state.done = False
if 'action_history' not in st.session_state:
    st.session_state.action_history = []
if 'reward_history' not in st.session_state:
    st.session_state.reward_history = []
if 'solar_used_history' not in st.session_state:
    st.session_state.solar_used_history = []
if 'battery_used_history' not in st.session_state:
    st.session_state.battery_used_history = []
if 'grid_used_history' not in st.session_state:
    st.session_state.grid_used_history = []
if 'grid_cost_history' not in st.session_state:
    st.session_state.grid_cost_history = []
if 'battery_soc_history' not in st.session_state:
    st.session_state.battery_soc_history = []
if 'running_simulation' not in st.session_state:
    st.session_state.running_simulation = False
if 'cumulative_reward' not in st.session_state:
    st.session_state.cumulative_reward = 0

# Action names
ACTION_NAMES = ['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar+Battery']

# Function to reset the environment
def reset_environment():
    env = EnergyManagementEnv()
    state, _ = env.reset()
    
    # Reset session state
    st.session_state.env = env
    st.session_state.state = state
    st.session_state.current_step = 0
    st.session_state.done = False
    st.session_state.episode_history = []
    st.session_state.action_history = []
    st.session_state.reward_history = []
    st.session_state.solar_used_history = []
    st.session_state.battery_used_history = []
    st.session_state.grid_used_history = []
    st.session_state.grid_cost_history = []
    st.session_state.battery_soc_history = []
    st.session_state.cumulative_reward = 0
    
    # Add initial state to history
    add_state_to_history(state)
    
    return env, state

# Function to load a model
def load_model(model_type, model_path):
    try:
        if model_type == "DQN":
            model = DQN.load(model_path)
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        st.session_state.model = model
        st.session_state.model_type = model_type
        st.success(f"{model_type} model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading {model_type} model: {e}")
        return None

# Function to add a state to the history
def add_state_to_history(state):
    # Extract values from state
    demand = state[0]
    solar = state[1]
    battery_soc = state[2]
    price = state[3]
    time_of_day = int(state[4])
    
    # Create a state dictionary
    state_dict = {
        'step': st.session_state.current_step,
        'demand': demand,
        'solar': solar,
        'battery_soc': battery_soc,
        'price': price,
        'time_of_day': time_of_day
    }
    
    # Add to history
    st.session_state.episode_history.append(state_dict)
    st.session_state.battery_soc_history.append(battery_soc)

# Function to take a step in the environment
def take_step(action):
    env = st.session_state.env
    state = st.session_state.state
    
    # Take step in environment
    next_state, reward, done, truncated, info = env.step(action)
    
    # Update session state
    st.session_state.state = next_state
    st.session_state.current_step += 1
    st.session_state.done = done or truncated
    st.session_state.cumulative_reward += reward
    
    # Record action and results
    st.session_state.action_history.append(action)
    st.session_state.reward_history.append(reward)
    st.session_state.solar_used_history.append(info['solar_used'])
    st.session_state.battery_used_history.append(info['battery_used'])
    st.session_state.grid_used_history.append(info['grid_used'])
    st.session_state.grid_cost_history.append(info['grid_cost'])
    
    # Add new state to history
    add_state_to_history(next_state)
    
    return next_state, reward, done, truncated, info

# Function to get action from model
def get_model_action(state):
    model = st.session_state.model
    model_type = st.session_state.model_type
    if model:
        if model_type == "DQN":
            action, _ = model.predict(state, deterministic=True)
            return action
        elif model_type in ["Random Forest", "XGBoost"]:
            # Model expects 2D array
            action = model.predict(np.array(state).reshape(1, -1))[0]
            return action
        else:
            st.warning("Unknown model type. Using random action.")
            return np.random.randint(0, 4)
    else:
        st.warning("No model loaded. Using random action.")
        return np.random.randint(0, 4)

# Function to run a complete episode
def run_episode(auto_run=False, delay=0.5):
    if not st.session_state.env:
        reset_environment()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.session_state.running_simulation = True
    
    for step in range(24):  # One day = 24 hours
        if st.session_state.done:
            break
        
        # Get action from model
        state = st.session_state.state
        action = get_model_action(state)
        
        # Take step
        next_state, reward, done, truncated, info = take_step(action)
        
        # Update progress
        progress = (step + 1) / 24
        progress_bar.progress(progress)
        status_text.text(f"Step {step+1}/24 - Reward: {reward:.2f}")
        
        if auto_run and not done and not truncated:
            time.sleep(delay)
        else:
            break
    
    st.session_state.running_simulation = False
    
    if st.session_state.done:
        status_text.text("Episode complete!")
    else:
        status_text.text(f"Step {st.session_state.current_step}/24")

# Function to step once
def step_once():
    if not st.session_state.env:
        reset_environment()
    
    if st.session_state.done:
        st.warning("Episode is already complete. Please reset.")
        return
    
    # Get action from model
    state = st.session_state.state
    action = get_model_action(state)
    
    # Take step
    next_state, reward, done, truncated, info = take_step(action)
    
    if done or truncated:
        st.success("Episode complete!")

# Function to create demand vs solar chart
def create_demand_solar_chart():
    if len(st.session_state.episode_history) == 0:
        return None
    
    df = pd.DataFrame(st.session_state.episode_history)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add demand line
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['demand'], name="Demand (kW)", line=dict(color='blue')),
        secondary_y=False,
    )
    
    # Add solar generation line
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['solar'], name="Solar Generation (kW)", line=dict(color='orange')),
        secondary_y=False,
    )
    
    # Add electricity price
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['price'], name="Electricity Price ($/kWh)", line=dict(color='red')),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title_text="Demand vs. Solar Generation vs. Price",
        height=400,
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Price ($/kWh)", secondary_y=True)
    
    return fig

# Function to create energy sources chart
def create_energy_sources_chart():
    if len(st.session_state.action_history) == 0:
        return None
    
    steps = list(range(st.session_state.current_step))
    
    fig = go.Figure()
    
    # Add solar used
    fig.add_trace(
        go.Bar(x=steps, y=st.session_state.solar_used_history, name="Solar Used (kW)", marker_color='yellow')
    )
    
    # Add battery used
    fig.add_trace(
        go.Bar(x=steps, y=st.session_state.battery_used_history, name="Battery Used (kW)", marker_color='green')
    )
    
    # Add grid used
    fig.add_trace(
        go.Bar(x=steps, y=st.session_state.grid_used_history, name="Grid Used (kW)", marker_color='red')
    )
    
    # Set titles and layout
    fig.update_layout(
        title_text="Energy Sources Used",
        barmode='stack',
        height=400,
    )
    
    return fig

# Function to create battery SOC chart
def create_battery_soc_chart():
    if len(st.session_state.battery_soc_history) == 0:
        return None
    
    steps = list(range(len(st.session_state.battery_soc_history)))
    
    fig = go.Figure()
    
    # Add battery SOC line
    fig.add_trace(
        go.Scatter(x=steps, y=st.session_state.battery_soc_history, name="Battery SOC (%)", line=dict(color='green'))
    )
    
    # Add min battery SOC threshold
    fig.add_shape(
        type="line",
        x0=0,
        y0=20,  # 20% minimum SOC
        x1=max(steps) if steps else 0,
        y1=20,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Set titles and layout
    fig.update_layout(
        title_text="Battery State of Charge (SOC)",
        height=400,
        yaxis=dict(
            title="SOC (%)",
            range=[0, 100],
        )
    )
    
    return fig

# Function to create grid cost chart
def create_grid_cost_chart():
    if len(st.session_state.grid_cost_history) == 0:
        return None
    
    steps = list(range(st.session_state.current_step))
    
    fig = go.Figure()
    
    # Add grid cost bars
    fig.add_trace(
        go.Bar(x=steps, y=st.session_state.grid_cost_history, name="Grid Cost ($)", marker_color='red')
    )
    
    # Set titles and layout
    fig.update_layout(
        title_text="Electricity Cost from Grid",
        height=400,
        yaxis=dict(title="Cost ($)")
    )
    
    return fig

# Function to create reward chart
def create_reward_chart():
    if len(st.session_state.reward_history) == 0:
        return None
    
    steps = list(range(st.session_state.current_step))
    cumulative_rewards = np.cumsum(st.session_state.reward_history)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add step reward bars
    fig.add_trace(
        go.Bar(x=steps, y=st.session_state.reward_history, name="Step Reward", marker_color='blue'),
        secondary_y=False,
    )
    
    # Add cumulative reward line
    fig.add_trace(
        go.Scatter(x=steps, y=cumulative_rewards, name="Cumulative Reward", line=dict(color='green')),
        secondary_y=True,
    )
    
    # Set titles and layout
    fig.update_layout(
        title_text="Rewards per Step",
        height=400,
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Step Reward", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Reward", secondary_y=True)
    
    return fig

# Function to create a state metrics visualization
def display_current_state_metrics():
    if not st.session_state.state is not None:
        return
    
    state = st.session_state.state
    
    # Extract values
    demand = state[0]
    solar = state[1]
    battery_soc = state[2]
    price = state[3]
    time_of_day = int(state[4])
    
    # Create layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Demand", f"{demand:.2f} kW")
        st.metric("Solar Generation", f"{solar:.2f} kW")
    
    with col2:
        st.metric("Battery SOC", f"{battery_soc:.1f}%")
        st.metric("Electricity Price", f"${price:.3f}/kWh")
    
    with col3:
        st.metric("Time of Day", f"{time_of_day}:00")
        
        if len(st.session_state.action_history) > 0:
            last_action = st.session_state.action_history[-1]
            st.metric("Last Action", ACTION_NAMES[last_action])
        
        if len(st.session_state.reward_history) > 0:
            last_reward = st.session_state.reward_history[-1]
            st.metric("Last Reward", f"{last_reward:.3f}")

# Function to create a detailed results table
def create_results_table():
    if len(st.session_state.episode_history) == 0:
        return None
    
    # Create DataFrame
    data = []
    
    for i in range(len(st.session_state.action_history)):
        step_data = {
            "Step": i + 1,
            "Time": f"{st.session_state.episode_history[i]['time_of_day']}:00",
            "Demand (kW)": f"{st.session_state.episode_history[i]['demand']:.2f}",
            "Solar (kW)": f"{st.session_state.episode_history[i]['solar']:.2f}",
            "Action": ACTION_NAMES[st.session_state.action_history[i]],
            "Solar Used (kW)": f"{st.session_state.solar_used_history[i]:.2f}",
            "Battery Used (kW)": f"{st.session_state.battery_used_history[i]:.2f}",
            "Grid Used (kW)": f"{st.session_state.grid_used_history[i]:.2f}",
            "Grid Cost ($)": f"{st.session_state.grid_cost_history[i]:.2f}",
            "Battery SOC (%)": f"{st.session_state.battery_soc_history[i+1]:.1f}",
            "Reward": f"{st.session_state.reward_history[i]:.3f}"
        }
        data.append(step_data)
    
    df = pd.DataFrame(data)
    return df

# Main dashboard layout
st.title("⚡ Energy Management RL Dashboard")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    # Model selection and loading
    st.subheader("Select Model")
    model_types = list(MODEL_PATHS.keys())
    selected_model_type = st.selectbox("Model Type", model_types, index=model_types.index(st.session_state.model_type) if 'model_type' in st.session_state else 0)
    model_path = MODEL_PATHS[selected_model_type]
    if st.button(f"Load {selected_model_type} Model"):
        load_model(selected_model_type, model_path)
    st.divider()
    
    st.divider()
    
    # Simulation controls
    st.subheader("Simulation")
    
    if st.button("Reset Environment"):
        reset_environment()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Step Once"):
            step_once()
    
    with col2:
        if st.button("Run Episode"):
            auto_run = st.checkbox("Auto-run", value=True)
            delay = st.slider("Delay (seconds)", 0.1, 2.0, 0.5)
            run_episode(auto_run, delay)
    
    st.divider()
    
    # Display summary metrics
    st.subheader("Summary")
    
    if len(st.session_state.reward_history) > 0:
        total_reward = sum(st.session_state.reward_history)
        st.metric("Total Reward", f"{total_reward:.2f}")
    
    if len(st.session_state.grid_cost_history) > 0:
        total_cost = sum(st.session_state.grid_cost_history)
        st.metric("Total Grid Cost", f"${total_cost:.2f}")
    
    if len(st.session_state.solar_used_history) > 0:
        total_solar = sum(st.session_state.solar_used_history)
        st.metric("Total Solar Used", f"{total_solar:.2f} kWh")
    
    if len(st.session_state.battery_used_history) > 0:
        total_battery = sum(st.session_state.battery_used_history)
        st.metric("Total Battery Used", f"{total_battery:.2f} kWh")
    
    if len(st.session_state.grid_used_history) > 0:
        total_grid = sum(st.session_state.grid_used_history)
        st.metric("Total Grid Used", f"{total_grid:.2f} kWh")

# Main content area
# Current state visualization
st.header("Current State")
st.markdown('<div class="metric-card">', unsafe_allow_html=True)
display_current_state_metrics()
st.markdown('</div>', unsafe_allow_html=True)

# Initialize environment if not already done
if st.session_state.env is None:
    reset_environment()

# Charts
st.header("Monitoring")

# First row of charts
col1, col2 = st.columns(2)

with col1:
    demand_solar_chart = create_demand_solar_chart()
    if demand_solar_chart:
        st.plotly_chart(demand_solar_chart, use_container_width=True)
    else:
        st.info("Run the simulation to see demand and solar data.")

with col2:
    energy_sources_chart = create_energy_sources_chart()
    if energy_sources_chart:
        st.plotly_chart(energy_sources_chart, use_container_width=True)
    else:
        st.info("Run the simulation to see energy sources data.")

# Second row of charts
col1, col2 = st.columns(2)

with col1:
    battery_soc_chart = create_battery_soc_chart()
    if battery_soc_chart:
        st.plotly_chart(battery_soc_chart, use_container_width=True)
    else:
        st.info("Run the simulation to see battery SOC data.")

with col2:
    reward_chart = create_reward_chart()
    if reward_chart:
        st.plotly_chart(reward_chart, use_container_width=True)
    else:
        st.info("Run the simulation to see reward data.")

# Grid cost chart
grid_cost_chart = create_grid_cost_chart()
if grid_cost_chart:
    st.plotly_chart(grid_cost_chart, use_container_width=True)
else:
    st.info("Run the simulation to see grid cost data.")

# Results table
st.header("Detailed Results")
results_table = create_results_table()
if results_table is not None:
    st.dataframe(results_table, use_container_width=True)
else:
    st.info("Run the simulation to see detailed results.")

# Add footer
st.markdown("---")
st.markdown("Energy Management RL Dashboard - Developed with Streamlit")
