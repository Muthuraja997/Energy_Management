import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import our prediction module
from prediction import EnergyPrediction

# Page configuration
st.set_page_config(
    page_title="Energy Source Predictor",
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

# Session state initialization
if 'predictor' not in st.session_state:
    st.session_state.predictor = EnergyPrediction()
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'previous_states' not in st.session_state:
    st.session_state.previous_states = []

# Function to load the model
def load_model(model_path):
    predictor = st.session_state.predictor
    predictor.model_path = model_path
    success = predictor.load_model()
    st.session_state.model_loaded = success
    return success

# Function to make a prediction
def make_prediction(demand, solar, battery_soc, price, time_of_day):
    predictor = st.session_state.predictor
    
    # Validate inputs
    valid, result = predictor.validate_state(demand, solar, battery_soc, price, time_of_day)
    
    if not valid:
        st.error(result)
        return None
        
    # Make prediction
    prediction = predictor.predict_best_source(result)
    
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
st.title("⚡ Energy Source Predictor")
st.markdown("This tool predicts the optimal energy source for your current system state using the trained DQN model.")

# Sidebar for model loading
with st.sidebar:
    st.header("Model Configuration")
    
    model_path = st.text_input("Model Path", value="models/best/best_model.zip")
    
    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            success = load_model(model_path)
            if success:
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model. Check the path and try again.")
    
    st.markdown("---")
    
    if st.session_state.model_loaded:
        st.success("✓ Model is loaded and ready")
    else:
        st.warning("⚠ Model not loaded")

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
    if not st.session_state.model_loaded:
        st.warning("Please load the model first.")
    else:
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
    
    This tool uses a trained Deep Q-Network (DQN) to predict the optimal energy source based on the current state of your system.
    
    #### Inputs:
    - **Demand (kW)**: Current electricity demand
    - **Solar Generation (kW)**: Current solar power production
    - **Battery SOC (%)**: Current state of charge of the battery
    - **Electricity Price ($/kWh)**: Current grid electricity price
    - **Time of Day**: Current hour (0-23)
    
    #### Outputs:
    - **Recommended Energy Source**: The optimal source to use
    - **Confidence**: How confident the model is in its prediction
    - **Q-Values**: The estimated future rewards for each action
    
    #### Actions:
    - **Use Solar**: Use solar power only
    - **Use Battery**: Use battery power only
    - **Use Grid**: Use grid power only
    - **Use Solar+Battery**: Use a combination of solar and battery power
    
    The model was trained using reinforcement learning to minimize cost and maximize renewable energy usage while ensuring demand is always met.
    """)

# Add footer
st.markdown("---")
st.markdown("Energy Management Reinforcement Learning Project")
