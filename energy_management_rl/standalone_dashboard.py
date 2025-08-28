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
from datetime import datetime, timedelta
import random

# Action names for reference
ACTION_NAMES = ['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar + Battery']

# Constants for the simulation
BATTERY_CAPACITY = 10.0  # kWh
MAX_SOLAR = 10.0  # kW
MAX_DEMAND = 20.0  # kW
MAX_PRICE = 0.5  # $/kWh
CO2_PER_KWH = 0.5  # kg CO2 per kWh from grid

# Page configuration
st.set_page_config(
    page_title="Energy Management Dashboard (Standalone)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.reset = True
    st.session_state.step = 0
    st.session_state.total_reward = 0
    st.session_state.episode_rewards = []
    st.session_state.actions = []
    st.session_state.states = []
    st.session_state.demands = []
    st.session_state.solar_values = []
    st.session_state.battery_socs = []
    st.session_state.prices = []
    st.session_state.times = []
    st.session_state.battery_powers = []
    st.session_state.grid_powers = []
    st.session_state.solar_powers = []
    st.session_state.costs = []
    st.session_state.co2_emissions = []
    st.session_state.action_confidences = {i: [] for i in range(4)}
    st.session_state.auto_run = False
    st.session_state.auto_run_speed = 1.0
    st.session_state.episode_count = 0
    st.session_state.last_action = None
    st.session_state.last_reward = None
    st.session_state.last_state = None
    st.session_state.custom_mode = False
    st.session_state.custom_demand = 5.0
    st.session_state.custom_solar = 2.0
    st.session_state.custom_battery_soc = 50.0
    st.session_state.custom_price = 0.2
    st.session_state.custom_time = 12.0
    st.session_state.hour = 0
    st.session_state.battery_soc = 50.0

# Title and description
st.title("⚡ Energy Management System Dashboard")
st.markdown("""
This dashboard demonstrates a rule-based energy management system that makes decisions 
about whether to use solar power, battery power, grid power, or a combination.
""")

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    # Reset environment
    if st.button("Reset Environment"):
        st.session_state.reset = True
        st.session_state.step = 0
        st.session_state.total_reward = 0
        st.session_state.actions = []
        st.session_state.states = []
        st.session_state.demands = []
        st.session_state.solar_values = []
        st.session_state.battery_socs = []
        st.session_state.prices = []
        st.session_state.times = []
        st.session_state.battery_powers = []
        st.session_state.grid_powers = []
        st.session_state.solar_powers = []
        st.session_state.costs = []
        st.session_state.co2_emissions = []
        st.session_state.last_action = None
        st.session_state.last_reward = None
        st.session_state.last_state = None
    
    # Step forward or automatically run
    col1, col2 = st.columns(2)
    with col1:
        step_button = st.button("Step Forward")
    with col2:
        auto_run = st.checkbox("Auto Run", value=st.session_state.auto_run)
        
    if auto_run != st.session_state.auto_run:
        st.session_state.auto_run = auto_run
    
    if st.session_state.auto_run:
        st.session_state.auto_run_speed = st.slider("Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    
    # Custom mode
    st.session_state.custom_mode = st.checkbox("Custom State", value=st.session_state.custom_mode)
    
    if st.session_state.custom_mode:
        st.session_state.custom_demand = st.slider("Demand (kW)", min_value=0.0, max_value=20.0, value=st.session_state.custom_demand, step=0.5)
        st.session_state.custom_solar = st.slider("Solar (kW)", min_value=0.0, max_value=10.0, value=st.session_state.custom_solar, step=0.5)
        st.session_state.custom_battery_soc = st.slider("Battery SOC (%)", min_value=0.0, max_value=100.0, value=st.session_state.custom_battery_soc, step=5.0)
        st.session_state.custom_price = st.slider("Price ($/kWh)", min_value=0.0, max_value=0.5, value=st.session_state.custom_price, step=0.01)
        st.session_state.custom_time = st.slider("Time (hour)", min_value=0.0, max_value=23.0, value=st.session_state.custom_time, step=1.0)
    
    st.markdown("---")
    st.header("Simulation Info")
    st.write(f"Episode: {st.session_state.episode_count}")
    st.write(f"Step: {st.session_state.step}")
    st.write(f"Total Reward: {st.session_state.total_reward:.2f}")

# Function to generate demand profile
def generate_demand(hour, base=5.0, morning_peak=3.0, evening_peak=8.0):
    # Morning peak around 7-9 AM
    morning = np.exp(-0.5 * ((hour - 8) / 1.5) ** 2) * morning_peak
    # Evening peak around 6-9 PM
    evening = np.exp(-0.5 * ((hour - 19) / 2) ** 2) * evening_peak
    # Base load
    base_load = base + np.sin(hour / 24 * np.pi) * 1.0
    return max(0, base_load + morning + evening)

# Function to generate solar profile
def generate_solar(hour, max_solar=5.0):
    # Solar generation between 6 AM and 6 PM with peak at noon
    if 6 <= hour <= 18:
        return max_solar * np.sin(np.pi * (hour - 6) / 12)
    else:
        return 0.0

# Function to generate price profile
def generate_price(hour, base=0.12, peak_factor=1.5):
    # Price is higher during peak demand hours
    if 7 <= hour <= 9 or 17 <= hour <= 21:
        return base * peak_factor
    else:
        return base

# Define the rule-based policy function
def rule_based_policy(state):
    demand, solar, battery_soc, price, time_of_day = state
    
    # Calculate confidence scores
    confidences = {
        0: 0.1,  # Solar
        1: 0.1,  # Battery
        2: 0.1,  # Grid
        3: 0.1   # Solar + Battery
    }
    
    # First priority: Use solar if available
    solar_available = solar > 0
    
    # Second priority: Use battery if SOC is good and price is high
    battery_usable = battery_soc > 20  # Don't go below 20% SOC
    price_is_high = price > 0.25
    
    # Calculate how much energy we need
    energy_needed = demand
    
    # Decision making with confidence scoring
    if solar_available and solar >= demand:
        # Solar can meet all demand
        confidences[0] = 0.9  # High confidence for solar
        best_action = 0  # Use solar
    elif solar_available and battery_usable and solar + (battery_soc/100 * 10) >= demand:
        # Solar + battery can meet demand
        if price_is_high or (time_of_day >= 18 or time_of_day <= 6):
            # High price or evening/night: use solar + battery
            confidences[3] = 0.8  # High confidence for solar+battery
            best_action = 3  # Use solar + battery
        else:
            # Use solar and save battery
            confidences[0] = 0.7  # Medium-high confidence for solar
            best_action = 0  # Use solar
    elif battery_usable and (battery_soc/100 * 10) >= demand and (price_is_high or solar < 0.5):
        # Battery can meet demand and price is high or solar is low
        confidences[1] = 0.8  # High confidence for battery
        best_action = 1  # Use battery
    elif solar_available and solar > 0.5:
        # Some solar available, but not enough - use solar anyway to offset grid
        confidences[0] = 0.6  # Medium confidence for solar
        best_action = 0  # Use solar
    else:
        # Default to grid if other options not viable
        confidences[2] = 0.7  # Medium-high confidence for grid
        best_action = 2  # Use grid
    
    return best_action, confidences

# Function to simulate taking a step in the environment
def simulate_step(state, action):
    demand, solar, battery_soc, price, time_of_day = state
    
    # Initialize returns
    reward = 0
    info = {}
    
    # Calculate energy flows based on action
    if action == 0:  # Use solar
        solar_power = min(solar, demand)
        battery_power = 0
        grid_power = demand - solar_power if demand > solar_power else 0
        
        # Update battery (charge with excess solar if any)
        if solar > demand:
            excess_solar = solar - demand
            battery_power = -min(excess_solar, (100 - battery_soc) * BATTERY_CAPACITY / 100)
            battery_soc = min(100, battery_soc - battery_power * 100 / BATTERY_CAPACITY)
        
    elif action == 1:  # Use battery
        solar_power = 0
        battery_power = min(demand, battery_soc * BATTERY_CAPACITY / 100)
        grid_power = demand - battery_power if demand > battery_power else 0
        
        # Update battery
        battery_soc = max(0, battery_soc - battery_power * 100 / BATTERY_CAPACITY)
        
    elif action == 2:  # Use grid
        solar_power = 0
        battery_power = 0
        grid_power = demand
        
        # Update battery (charge with solar if any)
        if solar > 0:
            battery_power = -min(solar, (100 - battery_soc) * BATTERY_CAPACITY / 100)
            battery_soc = min(100, battery_soc - battery_power * 100 / BATTERY_CAPACITY)
        
    elif action == 3:  # Use solar + battery
        solar_power = min(solar, demand)
        remaining_demand = demand - solar_power
        battery_power = min(remaining_demand, battery_soc * BATTERY_CAPACITY / 100)
        grid_power = remaining_demand - battery_power if remaining_demand > battery_power else 0
        
        # Update battery
        battery_soc = max(0, battery_soc - battery_power * 100 / BATTERY_CAPACITY)
    
    # Calculate costs and emissions
    cost = grid_power * price
    co2 = grid_power * CO2_PER_KWH
    
    # Calculate reward
    # Reward for using renewable energy
    reward += solar_power * 0.1
    
    # Penalty for using grid
    reward -= grid_power * price
    
    # Penalty for low battery
    if battery_soc < 20:
        reward -= (20 - battery_soc) * 0.05
    
    # Update time of day for next step
    next_time = (time_of_day + 1) % 24
    
    # For next state, generate new values or use custom if in custom mode
    if st.session_state.custom_mode:
        next_demand = st.session_state.custom_demand
        next_solar = st.session_state.custom_solar
        next_price = st.session_state.custom_price
    else:
        next_demand = generate_demand(next_time)
        next_solar = generate_solar(next_time)
        next_price = generate_price(next_time)
    
    # Construct next state
    next_state = np.array([next_demand, next_solar, battery_soc, next_price, next_time])
    
    # Check if episode is done (24 hours)
    done = (next_time == 0)
    
    # Populate info dictionary
    info = {
        'solar_power': solar_power,
        'battery_power': battery_power,
        'grid_power': grid_power,
        'cost': cost,
        'co2': co2
    }
    
    return next_state, reward, done, info

# Main content
main_content = st.container()

# Create visualization placeholders
current_state_container = st.container()
with current_state_container:
    st.header("Current State")
    col1, col2, col3, col4, col5 = st.columns(5)
    demand_metric = col1.empty()
    solar_metric = col2.empty()
    battery_metric = col3.empty()
    price_metric = col4.empty()
    time_metric = col5.empty()

action_container = st.container()
with action_container:
    st.header("Agent Action")
    action_placeholder = st.empty()

history_container = st.container()
with history_container:
    st.header("Simulation History")
    col1, col2 = st.columns(2)
    
    with col1:
        demand_solar_plot = st.empty()
    
    with col2:
        battery_plot = st.empty()
    
    col3, col4 = st.columns(2)
    
    with col3:
        actions_plot = st.empty()
    
    with col4:
        rewards_plot = st.empty()

# Function to update the visualization
def update_visualization():
    # Get current state
    if st.session_state.custom_mode:
        # Use custom values
        state = np.array([
            st.session_state.custom_demand,
            st.session_state.custom_solar,
            st.session_state.custom_battery_soc,
            st.session_state.custom_price,
            st.session_state.custom_time
        ])
    else:
        # Generate values based on current hour
        hour = st.session_state.hour
        demand = generate_demand(hour)
        solar = generate_solar(hour)
        price = generate_price(hour)
        battery_soc = st.session_state.battery_soc
        
        state = np.array([demand, solar, battery_soc, price, hour])
    
    demand, solar, battery_soc, price, time_of_day = state
    
    # Update metrics
    demand_metric.metric("Demand", f"{demand:.2f} kW")
    solar_metric.metric("Solar", f"{solar:.2f} kW")
    battery_metric.metric("Battery SOC", f"{battery_soc:.1f}%")
    price_metric.metric("Price", f"${price:.3f}/kWh")
    time_metric.metric("Time", f"{int(time_of_day):02d}:00")
    
    # Predict action
    action, confidences = rule_based_policy(state)
    
    # Update action display
    action_md = f"""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'>
        <h3 style='margin-top: 0;'>Recommended Action: <span style='color: #4e8cff;'>{ACTION_NAMES[action]}</span></h3>
        <div style='display: flex; margin-bottom: 10px;'>
            <div style='width: 25%; text-align: center; padding: 10px; margin: 5px; border-radius: 5px; background-color: {"#e6f7ff" if action == 0 else "#f5f5f5"};'>
                <h4 style='margin-top: 0;'>Use Solar</h4>
                <p style='font-size: 24px; margin: 0;'>{confidences[0]*100:.1f}%</p>
            </div>
            <div style='width: 25%; text-align: center; padding: 10px; margin: 5px; border-radius: 5px; background-color: {"#e6f7ff" if action == 1 else "#f5f5f5"};'>
                <h4 style='margin-top: 0;'>Use Battery</h4>
                <p style='font-size: 24px; margin: 0;'>{confidences[1]*100:.1f}%</p>
            </div>
            <div style='width: 25%; text-align: center; padding: 10px; margin: 5px; border-radius: 5px; background-color: {"#e6f7ff" if action == 2 else "#f5f5f5"};'>
                <h4 style='margin-top: 0;'>Use Grid</h4>
                <p style='font-size: 24px; margin: 0;'>{confidences[2]*100:.1f}%</p>
            </div>
            <div style='width: 25%; text-align: center; padding: 10px; margin: 5px; border-radius: 5px; background-color: {"#e6f7ff" if action == 3 else "#f5f5f5"};'>
                <h4 style='margin-top: 0;'>Solar+Battery</h4>
                <p style='font-size: 24px; margin: 0;'>{confidences[3]*100:.1f}%</p>
            </div>
        </div>
        <p><strong>Explanation:</strong> {get_action_explanation(state, action)}</p>
    </div>
    """
    action_placeholder.markdown(action_md, unsafe_allow_html=True)
    
    # Update history plots if there's data
    if len(st.session_state.demands) > 0:
        # Demand and solar
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig1.add_trace(
            go.Scatter(x=list(range(len(st.session_state.demands))), y=st.session_state.demands, name="Demand", line=dict(color='#ff9900', width=3)),
            secondary_y=False,
        )
        
        fig1.add_trace(
            go.Scatter(x=list(range(len(st.session_state.solar_values))), y=st.session_state.solar_values, name="Solar", line=dict(color='#66cc00', width=3)),
            secondary_y=False,
        )
        
        fig1.update_layout(
            title_text="Demand and Solar Generation",
            xaxis_title="Step",
            yaxis_title="Power (kW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
        )
        
        demand_solar_plot.plotly_chart(fig1, use_container_width=True)
        
        # Battery SOC
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Scatter(x=list(range(len(st.session_state.battery_socs))), y=st.session_state.battery_socs, name="Battery SOC", line=dict(color='#3366ff', width=3))
        )
        
        fig2.update_layout(
            title_text="Battery State of Charge",
            xaxis_title="Step",
            yaxis_title="SOC (%)",
            yaxis=dict(range=[0, 100]),
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
        )
        
        battery_plot.plotly_chart(fig2, use_container_width=True)
        
        # Actions
        fig3 = go.Figure()
        
        action_names = ['Solar', 'Battery', 'Grid', 'Solar+Battery']
        colors = ['#66cc00', '#3366ff', '#ff9900', '#9933cc']
        
        # Create a more visually clear representation of actions
        action_data = []
        for i, action in enumerate(st.session_state.actions):
            for j in range(4):
                if j == action:
                    action_data.append(action_names[j])
                    break
        
        # Count occurrences of each action
        action_counts = {name: action_data.count(name) for name in action_names}
        
        fig3 = px.bar(
            x=list(action_counts.keys()),
            y=list(action_counts.values()),
            color=list(action_counts.keys()),
            color_discrete_map={
                'Solar': '#66cc00',
                'Battery': '#3366ff',
                'Grid': '#ff9900',
                'Solar+Battery': '#9933cc'
            }
        )
        
        fig3.update_layout(
            title_text="Action Distribution",
            xaxis_title="Action",
            yaxis_title="Count",
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
        )
        
        actions_plot.plotly_chart(fig3, use_container_width=True)
        
        # Rewards
        cumulative_rewards = np.cumsum(st.session_state.episode_rewards)
        
        fig4 = go.Figure()
        
        fig4.add_trace(
            go.Scatter(x=list(range(len(cumulative_rewards))), y=cumulative_rewards, name="Cumulative Reward", line=dict(color='#ff3366', width=3))
        )
        
        fig4.update_layout(
            title_text="Cumulative Reward",
            xaxis_title="Step",
            yaxis_title="Reward",
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
        )
        
        rewards_plot.plotly_chart(fig4, use_container_width=True)

# Function to get explanation for an action
def get_action_explanation(state, action):
    demand, solar, battery_soc, price, time_of_day = state
    
    if action == 0:  # Use Solar
        if solar >= demand:
            return f"Solar generation ({solar:.2f} kW) is sufficient to meet the demand ({demand:.2f} kW)."
        else:
            return f"Using available solar ({solar:.2f} kW) to partially offset demand ({demand:.2f} kW), reducing grid dependency."
    
    elif action == 1:  # Use Battery
        if price > 0.25:
            return f"Electricity price (${price:.3f}/kWh) is high, so using battery to avoid expensive grid power."
        else:
            return f"Battery state of charge ({battery_soc:.1f}%) is good and solar is limited, so using stored energy."
    
    elif action == 2:  # Use Grid
        if solar < 0.5 and battery_soc < 30:
            return f"Low solar availability ({solar:.2f} kW) and battery charge ({battery_soc:.1f}%) necessitates grid usage."
        else:
            return f"Grid is the most economical option given current conditions."
    
    elif action == 3:  # Use Solar + Battery
        return f"Combining solar ({solar:.2f} kW) and battery power to meet high demand ({demand:.2f} kW) without using the grid."

# Main loop
if st.session_state.reset:
    st.session_state.reset = False
    st.session_state.hour = 0
    st.session_state.battery_soc = 50.0

# Initialize state if first run
if st.session_state.custom_mode:
    # Use custom values
    state = np.array([
        st.session_state.custom_demand,
        st.session_state.custom_solar,
        st.session_state.custom_battery_soc,
        st.session_state.custom_price,
        st.session_state.custom_time
    ])
else:
    # Generate values based on current hour
    hour = st.session_state.hour
    demand = generate_demand(hour)
    solar = generate_solar(hour)
    price = generate_price(hour)
    battery_soc = st.session_state.battery_soc
    
    state = np.array([demand, solar, battery_soc, price, hour])

# Update visualization
update_visualization()

# Check if step button is pressed or auto run is enabled
if step_button or st.session_state.auto_run:
    # Get current state
    if st.session_state.custom_mode:
        # Use custom values
        state = np.array([
            st.session_state.custom_demand,
            st.session_state.custom_solar,
            st.session_state.custom_battery_soc,
            st.session_state.custom_price,
            st.session_state.custom_time
        ])
    else:
        # Generate values based on current hour
        hour = st.session_state.hour
        demand = generate_demand(hour)
        solar = generate_solar(hour)
        price = generate_price(hour)
        battery_soc = st.session_state.battery_soc
        
        state = np.array([demand, solar, battery_soc, price, hour])
    
    # Get action from policy
    action, confidences = rule_based_policy(state)
    
    # Take step in simulated environment
    next_state, reward, done, info = simulate_step(state, action)
    
    # Update state
    st.session_state.battery_soc = next_state[2]  # Update battery SOC
    st.session_state.hour = int(next_state[4])    # Update hour
    
    # Update state
    state = next_state
    
    # Update session state
    st.session_state.step += 1
    st.session_state.total_reward += reward
    st.session_state.episode_rewards.append(reward)
    st.session_state.actions.append(action)
    st.session_state.states.append(state.tolist())
    st.session_state.demands.append(state[0])
    st.session_state.solar_values.append(state[1])
    st.session_state.battery_socs.append(state[2])
    st.session_state.prices.append(state[3])
    st.session_state.times.append(state[4])
    
    # Update action confidences
    for a, conf in confidences.items():
        st.session_state.action_confidences[a].append(conf)
    
    # Extract additional info
    st.session_state.battery_powers.append(info.get('battery_power', 0))
    st.session_state.grid_powers.append(info.get('grid_power', 0))
    st.session_state.solar_powers.append(info.get('solar_power', 0))
    st.session_state.costs.append(info.get('cost', 0))
    st.session_state.co2_emissions.append(info.get('co2', 0))
    
    # Update visualization
    update_visualization()
    
    # Check if episode is done
    if done:
        st.session_state.reset = True
    
    # Add a small delay if auto running
    if st.session_state.auto_run:
        time.sleep(1.0 / st.session_state.auto_run_speed)
        st.rerun()
