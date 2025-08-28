import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import os
import sys
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Energy Source Predictor (Standalone)",
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
    .prediction-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .header-text {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .solar-box {
        background-color: #FFF59D;
        border-left: 6px solid #FDD835;
        padding: 1rem;
        border-radius: 5px;
    }
    .battery-box {
        background-color: #C8E6C9;
        border-left: 6px solid #66BB6A;
        padding: 1rem;
        border-radius: 5px;
    }
    .grid-box {
        background-color: #FFCDD2;
        border-left: 6px solid #EF5350;
        padding: 1rem;
        border-radius: 5px;
    }
    .hybrid-box {
        background-color: #BBDEFB;
        border-left: 6px solid #42A5F5;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Action names
ACTION_NAMES = ['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar+Battery']

# Session state initialization
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'previous_states' not in st.session_state:
    st.session_state.previous_states = []

# Trained model logic (simplified decision rules based on the actual DQN model behavior)
def predict_action(demand, solar, battery_soc, price, time_of_day):
    """Predict the best energy source based on the current state.
    This is a simplified version of what the DQN model would predict."""
    
    # Decision logic based on observations of the trained model
    if solar >= demand:
        # If solar can cover demand, use solar
        action = 0  # Use solar
        confidence = 0.95
        q_values = [5.0, -1.0, -2.0, 2.0]
    elif price > 0.15 and battery_soc > 30 and demand > solar:
        if battery_soc > 60:
            # If it's expensive electricity and we have plenty of battery, use battery
            action = 1  # Use battery
            confidence = 0.85
            q_values = [-1.0, 4.0, -2.0, 2.0]
        else:
            # If battery is not very high but price is high, combine solar and battery
            action = 3  # Use solar + battery
            confidence = 0.80
            q_values = [1.0, 2.0, -2.0, 4.0]
    elif 6 <= time_of_day <= 18 and solar > 0:
        # During daytime with some solar, consider combining
        if battery_soc > 50:
            # If we have decent battery, combine solar and battery
            action = 3  # Use solar + battery
            confidence = 0.75
            q_values = [2.0, 1.0, -1.0, 3.0]
        else:
            # With lower battery but some solar, use solar
            action = 0  # Use solar
            confidence = 0.70
            q_values = [3.0, 0.0, -1.0, 2.0]
    elif battery_soc > 40 and price > 0.12:
        # If we have decent battery and price is somewhat high, use battery
        action = 1  # Use battery
        confidence = 0.65
        q_values = [-1.0, 3.0, 0.0, 2.0]
    else:
        # Default to grid when other conditions don't apply
        action = 2  # Use grid
        confidence = 0.60
        q_values = [-2.0, -1.0, 3.0, -1.0]
    
    # Add some randomness to q_values to simulate a real model (but keep the winner the same)
    for i in range(4):
        if i != action:
            q_values[i] += np.random.uniform(-0.5, 0.5)
    
    # Normalize q_values for confidence calculation
    exp_q = np.exp(np.array(q_values) - np.max(q_values))
    confidence = exp_q[action] / exp_q.sum()
    
    return {
        "action": action,
        "action_name": ACTION_NAMES[action],
        "q_values": q_values,
        "confidence": float(confidence)
    }

# Function to make a prediction
def make_prediction(demand, solar, battery_soc, price, time_of_day):
    # Validate inputs
    try:
        demand = float(demand)
        solar = float(solar)
        battery_soc = float(battery_soc)
        price = float(price)
        time_of_day = int(time_of_day)
        
        # Validate ranges
        if demand < 0:
            st.error("Demand must be non-negative")
            return None
        if solar < 0:
            st.error("Solar generation must be non-negative")
            return None
        if battery_soc < 0 or battery_soc > 100:
            st.error("Battery SOC must be between 0 and 100")
            return None
        if price < 0:
            st.error("Price must be non-negative")
            return None
        if time_of_day < 0 or time_of_day > 23:
            st.error("Time of day must be between 0 and 23")
            return None
            
    except ValueError:
        st.error("All inputs must be numeric values")
        return None
        
    # Make prediction using rules that approximate the DQN model
    prediction = predict_action(demand, solar, battery_soc, price, time_of_day)
    
    # Store prediction in session state
    st.session_state.prediction_result = prediction
    
    # Add to previous states
    state_dict = {
        'demand': demand,
        'solar': solar,
        'battery_soc': battery_soc,
        'price': price,
        'time_of_day': time_of_day,
        'action': prediction['action_name'],
        'confidence': prediction['confidence']
    }
    st.session_state.previous_states.append(state_dict)
    
    # Keep only the last 10 predictions
    if len(st.session_state.previous_states) > 10:
        st.session_state.previous_states.pop(0)
        
    return prediction

# Function to display prediction result
def display_prediction_result(prediction):
    if prediction is None:
        return
    
    action = prediction['action']
    action_name = prediction['action_name']
    confidence = prediction['confidence']
    q_values = prediction['q_values']
    
    # Get appropriate style class based on action
    style_class = ""
    if action == 0:
        style_class = "solar-box"
    elif action == 1:
        style_class = "battery-box"
    elif action == 2:
        style_class = "grid-box"
    elif action == 3:
        style_class = "hybrid-box"
    
    # Display result
    st.markdown(f'<div class="{style_class}">', unsafe_allow_html=True)
    st.markdown(f"### Recommended Energy Source: {action_name}")
    st.markdown(f"**Confidence:** {confidence:.2%}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display Q-values
    st.subheader("Q-Values for All Actions")
    
    q_value_data = {
        'Action': ['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar+Battery'],
        'Q-Value': q_values,
        'Normalized': [np.exp(q - max(q_values)) / sum(np.exp(qv - max(q_values)) for qv in q_values) for q in q_values]
    }
    
    q_df = pd.DataFrame(q_value_data)
    
    # Highlight the selected action
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
    
    styled_df = q_df.style.apply(highlight_max, subset=['Q-Value'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(styled_df, width="stretch")
    
    with col2:
        # Create a bar chart of Q-values
        fig = go.Figure()
        
        colors = ['#FDD835', '#66BB6A', '#EF5350', '#42A5F5']
        
        for i, (action_name, q_value) in enumerate(zip(q_value_data['Action'], q_value_data['Q-Value'])):
            fig.add_trace(go.Bar(
                x=[action_name],
                y=[q_value],
                name=action_name,
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            title="Q-Values by Action",
            xaxis_title="Action",
            yaxis_title="Q-Value",
            height=300
        )
        
        st.plotly_chart(fig, width="stretch")

# Function to display previous predictions
def display_previous_predictions():
    if len(st.session_state.previous_states) == 0:
        st.info("No previous predictions yet.")
        return
        
    st.subheader("Previous Predictions")
    
    df = pd.DataFrame(st.session_state.previous_states)
    
    # Format the dataframe
    df['time_of_day'] = df['time_of_day'].apply(lambda x: f"{int(x)}:00")
    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
    
    # Reorder columns
    columns = ['demand', 'solar', 'battery_soc', 'price', 'time_of_day', 'action', 'confidence']
    df = df[columns]
    
    # Rename columns
    df.columns = ['Demand (kW)', 'Solar (kW)', 'Battery SOC (%)', 'Price ($/kWh)', 'Time', 'Recommended Action', 'Confidence']
    
    st.dataframe(df, width="stretch")

# Main dashboard layout
st.title("⚡ Energy Source Predictor (Standalone)")
st.markdown("This tool predicts the optimal energy source for your current system state based on the patterns learned by the DQN model.")
st.info("This is a standalone version that doesn't require the trained model. It uses decision rules that approximate what the DQN model would predict.")

# Main content
st.header("Input Current System State")

st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

# Input form
col1, col2 = st.columns(2)

with col1:
    demand = st.number_input("Demand (kW)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    solar = st.number_input("Solar Generation (kW)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
    battery_soc = st.number_input("Battery State of Charge (%)", min_value=0.0, max_value=100.0, value=60.0, step=5.0)

with col2:
    price = st.number_input("Electricity Price ($/kWh)", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    time_of_day = st.slider("Time of Day", min_value=0, max_value=23, value=12)
    st.markdown(f"Selected time: **{time_of_day}:00**")

# Prediction button
if st.button("Predict Best Energy Source"):
    with st.spinner("Making prediction..."):
        prediction = make_prediction(demand, solar, battery_soc, price, time_of_day)
        
st.markdown('</div>', unsafe_allow_html=True)

# Display prediction result
if st.session_state.prediction_result:
    st.header("Prediction Result")
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    display_prediction_result(st.session_state.prediction_result)
    st.markdown('</div>', unsafe_allow_html=True)

# Display previous predictions
st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
display_previous_predictions()
st.markdown('</div>', unsafe_allow_html=True)

# Explanation section
with st.expander("How It Works"):
    st.markdown("""
    ### How the Prediction Works
    
    This standalone predictor uses a set of decision rules that approximate what the trained Deep Q-Network (DQN) would predict.
    
    #### Inputs:
    - **Demand (kW)**: Current electricity demand
    - **Solar Generation (kW)**: Current solar power production
    - **Battery SOC (%)**: Current state of charge of the battery
    - **Electricity Price ($/kWh)**: Current grid electricity price
    - **Time of Day**: Current hour (0-23)
    
    #### Outputs:
    - **Recommended Energy Source**: The optimal source to use
    - **Confidence**: How confident the predictor is in its recommendation
    - **Q-Values**: The simulated Q-values for each action
    
    #### Actions:
    - **Use Solar**: Use solar power only
    - **Use Battery**: Use battery power only
    - **Use Grid**: Use grid power only
    - **Use Solar+Battery**: Use a combination of solar and battery power
    
    #### Decision Logic (approximating the DQN model):
    1. If solar can cover demand, use solar
    2. If electricity is expensive and battery has enough charge, use battery
    3. During daytime with some solar available, consider combining solar and battery
    4. With decent battery and high price, prefer battery
    5. Otherwise, use grid power
    """)

# Add footer
st.markdown("---")
st.markdown("Energy Management RL Project - Standalone Predictor")
