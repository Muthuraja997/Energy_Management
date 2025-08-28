import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Source Predictor (XGBoost)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Action names for reference
ACTION_NAMES = ['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar + Battery']

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
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for history tracking
if 'history' not in st.session_state:
    st.session_state.history = []

# Title and description
st.title("⚡ Energy Source Predictor (XGBoost)")
st.markdown("""
This tool predicts the optimal energy source (solar, battery, grid, or solar+battery) 
based on your current energy system state. It uses an XGBoost machine learning model
trained on simulated data from a rule-based policy.
""")

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model_path = "models/xgboost_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        else:
            st.error(f"Model file not found at {model_path}. Please train the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the model
model = load_model()

# Check if model is loaded
if model is None:
    st.warning("Model not loaded. Using fallback rule-based policy.")
    use_rules = True
else:
    use_rules = False

# Sidebar inputs
st.sidebar.header("Energy System State")

# Demand
demand = st.sidebar.slider("Energy Demand (kW)", 0.0, 20.0, 5.0, 0.1)

# Solar
solar = st.sidebar.slider("Available Solar Power (kW)", 0.0, 15.0, 2.0, 0.1)

# Battery
battery_soc = st.sidebar.slider("Battery State of Charge (%)", 0.0, 100.0, 50.0, 1.0)

# Price
price = st.sidebar.slider("Electricity Price ($/kWh)", 0.05, 0.50, 0.15, 0.01)

# Initialize the time of day in session state if not already present
if 'time_of_day_hour' not in st.session_state:
    st.session_state.time_of_day_hour = 12.0

# Set up time of day input
st.sidebar.markdown("### Time of Day")
st.session_state.time_of_day_hour = st.sidebar.slider("Hour of Day", 0, 23, int(st.session_state.time_of_day_hour))
time_of_day = st.session_state.time_of_day_hour

# Rule-based policy as fallback
def rule_based_policy(state):
    """
    Rule-based policy to determine the best energy source.
    
    Parameters:
    - state: Array containing [demand, solar, battery_soc, price, time_of_day]
    
    Returns:
    - action: The recommended action (0-3)
    - confidences: Dictionary of confidence scores for each action
    """
    demand, solar, battery_soc, price, time_of_day = state
    
    # Initialize confidence scores
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

# Function to make prediction with XGBoost model
def make_prediction(demand, solar, battery_soc, price, time_of_day):
    """
    Make prediction using the XGBoost model or fallback to rule-based policy.
    
    Parameters:
    - demand: Current energy demand in kW
    - solar: Current solar generation in kW
    - battery_soc: Current battery state of charge as percentage
    - price: Current electricity price in $/kWh
    - time_of_day: Current hour of the day (0-23)
    
    Returns:
    - Dictionary containing the recommended action and probability scores
    """
    # Create input features
    features = np.array([[demand, solar, battery_soc, price, time_of_day]])
    
    # Make prediction
    if not use_rules and model is not None:
        # Use XGBoost model
        probabilities = model.predict_proba(features)[0]
        best_action = np.argmax(probabilities)
        
        # Convert to confidence dictionary
        confidences = {i: float(prob) for i, prob in enumerate(probabilities)}
    else:
        # Use rule-based policy as fallback
        best_action, confidences = rule_based_policy([demand, solar, battery_soc, price, time_of_day])
    
    # Create explanation
    explanation = get_action_explanation([demand, solar, battery_soc, price, time_of_day], best_action)
    
    # Create prediction result
    result = {
        'action': best_action,
        'action_name': ACTION_NAMES[best_action],
        'confidences': confidences,
        'explanation': explanation,
        'state': {
            'demand': demand,
            'solar': solar,
            'battery_soc': battery_soc,
            'price': price,
            'time_of_day': time_of_day
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': 'XGBoost' if not use_rules else 'Rule-based (fallback)'
    }
    
    return result

# Feature importance plot
def plot_feature_importance():
    if not use_rules and model is not None:
        # Get feature importance
        importance = model.feature_importances_
        features = ['Demand', 'Solar', 'Battery SOC', 'Price', 'Time of Day']
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=features,
            y=importance,
            marker_color=['#ff9900', '#66cc00', '#3366ff', '#ff3366', '#9933cc']
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Feature",
            yaxis_title="Importance",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    else:
        return None

# Make prediction
if st.button("Predict Best Source", type="primary"):
    with st.spinner("Making prediction..."):
        prediction = make_prediction(
            demand=demand,
            solar=solar,
            battery_soc=battery_soc,
            price=price,
            time_of_day=time_of_day
        )
        
        # Add to history
        st.session_state.history.append(prediction)
        
        # Show prediction
        st.markdown("## Prediction Result")
        
        # Model type info
        st.info(f"Using {prediction['model_type']} for prediction")
        
        # Create color-coded confidence display
        confidence_html = "<div style='display: flex; margin-bottom: 20px;'>"
        
        for i, action_name in enumerate(ACTION_NAMES):
            confidence = prediction['confidences'][i] * 100
            if i == prediction['action']:
                bg_color = "#e6f7ff"  # Highlight selected action
                border = "2px solid #4e8cff"
            else:
                bg_color = "#f5f5f5"
                border = "1px solid #ddd"
                
            if confidence >= 70:
                conf_class = "confidence-high"
            elif confidence >= 40:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
                
            confidence_html += f"""
            <div style='flex: 1; text-align: center; padding: 15px; margin: 5px; 
                       border-radius: 5px; background-color: {bg_color}; border: {border};'>
                <h4 style='margin-top: 0;'>{action_name}</h4>
                <p class='{conf_class}' style='font-size: 24px; margin: 0;'>{confidence:.1f}%</p>
            </div>
            """
        
        confidence_html += "</div>"
        
        # Show recommendation
        rec_html = f"""
        <div class='prediction-card'>
            <h3>Recommended Action: <span style='color: #4e8cff;'>{prediction['action_name']}</span></h3>
            {confidence_html}
            <p><strong>Explanation:</strong> {prediction['explanation']}</p>
        </div>
        """
        
        st.markdown(rec_html, unsafe_allow_html=True)
        
        # Show feature importance if using XGBoost
        if not use_rules and model is not None:
            st.subheader("Feature Importance")
            fig = plot_feature_importance()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Show history
        if len(st.session_state.history) > 1:
            st.markdown("## Prediction History")
            
            # Create a dataframe from history
            history_data = []
            for p in st.session_state.history:
                history_data.append({
                    'Time': p['timestamp'],
                    'Demand (kW)': p['state']['demand'],
                    'Solar (kW)': p['state']['solar'],
                    'Battery (%)': p['state']['battery_soc'],
                    'Price ($/kWh)': p['state']['price'],
                    'Hour': int(p['state']['time_of_day']),
                    'Action': p['action_name'],
                    'Confidence': f"{p['confidences'][p['action']]*100:.1f}%",
                    'Model': p.get('model_type', 'Unknown')
                })
            
            # Display history table
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)
            
            # Option to clear history
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
