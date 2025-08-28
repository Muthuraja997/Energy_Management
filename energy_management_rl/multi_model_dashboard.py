import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import random
import warnings
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import preprocess_obs
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Management Multi-Model Dashboard",
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
    .model-header {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
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
    .comparison-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for history tracking
if 'history' not in st.session_state:
    st.session_state.history = []

# Title and description
st.title("⚡ Energy Management Multi-Model Dashboard")
st.markdown("""
This dashboard allows you to compare different machine learning models for predicting 
the optimal energy source (solar, battery, grid, or solar+battery) based on your current 
energy system state.
""")

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    models_info = {}
    
    # Try to load XGBoost model
    try:
        xgboost_path = "models/xgboost_model.pkl"
        if os.path.exists(xgboost_path):
            with open(xgboost_path, "rb") as f:
                models["xgboost"] = pickle.load(f)
            models_info["xgboost"] = {"name": "XGBoost", "loaded": True}
        else:
            models_info["xgboost"] = {"name": "XGBoost", "loaded": False, "error": "Model file not found"}
    except Exception as e:
        models_info["xgboost"] = {"name": "XGBoost", "loaded": False, "error": str(e)}
    
    # Try to load Random Forest model
    try:
        rf_path = "models/random_forest_model.pkl"
        if os.path.exists(rf_path):
            with open(rf_path, "rb") as f:
                models["random_forest"] = pickle.load(f)
            models_info["random_forest"] = {"name": "Random Forest", "loaded": True}
        else:
            models_info["random_forest"] = {"name": "Random Forest", "loaded": False, "error": "Model file not found"}
    except Exception as e:
        models_info["random_forest"] = {"name": "Random Forest", "loaded": False, "error": str(e)}
    
    # Try to load DQN model
    try:
        dqn_path = "models/final_model.zip"
        if os.path.exists(dqn_path):
            models["dqn"] = DQN.load(dqn_path)
            models_info["dqn"] = {"name": "DQN", "loaded": True}
        else:
            models_info["dqn"] = {"name": "DQN", "loaded": False, "error": "Model file not found"}
    except Exception as e:
        models_info["dqn"] = {"name": "DQN", "loaded": False, "error": str(e)}
    
    return models, models_info

# Load models
models, models_info = load_models()

# Check loaded models and show status
st.sidebar.header("Models Status")
for model_key, info in models_info.items():
    if info["loaded"]:
        st.sidebar.success(f"✅ {info['name']} loaded successfully")
    else:
        st.sidebar.warning(f"⚠️ {info['name']} not loaded: {info.get('error', 'Unknown error')}")

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

# Time of day
time_of_day = st.sidebar.slider("Hour of Day", 0, 23, 12)

# Model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    options=["All Models"] + [info["name"] for model_key, info in models_info.items() if info["loaded"]],
    index=0
)

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

# Function to make prediction with selected model
def predict_with_selected_model(model_choice, state):
    """
    Make prediction using the selected model or all models.
    
    Parameters:
    - model_choice: The selected model name or "All Models"
    - state: Array containing [demand, solar, battery_soc, price, time_of_day]
    
    Returns:
    - Dictionary containing predictions from selected model(s)
    """
    demand, solar, battery_soc, price, time_of_day = state
    features = np.array([[demand, solar, battery_soc, price, time_of_day]])
    
    results = {}
    
    # Helper function for prediction
    def predict_single_model(model_key, model_name):
        if model_key == "rule_based":
            # Use rule-based policy
            best_action, confidences = rule_based_policy(state)
            probabilities = np.array([confidences[i] for i in range(4)])
        elif model_key == "dqn":
            # Use DQN model
            model = models.get(model_key)
            if model is not None:
                # For DQN, we need to convert the observation properly
                obs = np.array(state).reshape(1, -1)
                action, _ = model.predict(obs, deterministic=True)
                best_action = int(action)
                
                # Get Q-values (confidences) from the model
                q_values = model.q_net(model.q_net.obs_to_tensor(obs)[0])[0].detach().numpy()
                
                # Convert Q-values to probabilities using softmax
                q_values_exp = np.exp(q_values - np.max(q_values))
                probabilities = q_values_exp / q_values_exp.sum()
            else:
                # Fall back to rule-based
                best_action, confidences = rule_based_policy(state)
                probabilities = np.array([confidences[i] for i in range(4)])
        else:
            # Use supervised model (XGBoost or Random Forest)
            model = models.get(model_key)
            if model is not None:
                probabilities = model.predict_proba(features)[0]
                best_action = np.argmax(probabilities)
            else:
                # Fall back to rule-based
                best_action, confidences = rule_based_policy(state)
                probabilities = np.array([confidences[i] for i in range(4)])
        
        # Create confidences dictionary
        confidences = {i: float(prob) for i, prob in enumerate(probabilities)}
        
        # Create explanation
        explanation = get_action_explanation(state, best_action)
        
        return {
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
            'model_type': model_name
        }
    
    # Make predictions based on model choice
    if model_choice == "All Models":
        # Predict with all available models
        for model_key, info in models_info.items():
            if info["loaded"]:
                results[model_key] = predict_single_model(model_key, info["name"])
        
        # Also add rule-based prediction for comparison
        results["rule_based"] = predict_single_model("rule_based", "Rule-based")
    else:
        # Predict with selected model
        for model_key, info in models_info.items():
            if info["name"] == model_choice and info["loaded"]:
                results[model_key] = predict_single_model(model_key, info["name"])
                break
    
    return results

# Function to display prediction results
def display_predictions(predictions):
    """
    Display the prediction results in a visually appealing way.
    
    Parameters:
    - predictions: Dictionary of predictions from different models
    """
    if len(predictions) == 1:
        # Single model prediction
        model_key = list(predictions.keys())[0]
        prediction = predictions[model_key]
        
        st.markdown(f"## Prediction from {prediction['model_type']}")
        
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
        
    else:
        # Multiple model comparison
        st.markdown("## Model Comparison")
        
        # Create tabs for each model
        tabs = st.tabs([p["model_type"] for p in predictions.values()])
        
        # Display detailed prediction for each model in its tab
        for i, (model_key, prediction) in enumerate(predictions.items()):
            with tabs[i]:
                # Create color-coded confidence display
                confidence_html = "<div style='display: flex; margin-bottom: 20px;'>"
                
                for j, action_name in enumerate(ACTION_NAMES):
                    confidence = prediction['confidences'][j] * 100
                    if j == prediction['action']:
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
        
        # Summary comparison table
        st.markdown("## Side-by-Side Comparison")
        
        # Create comparison cards
        comparison_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
        
        for model_key, prediction in predictions.items():
            model_color = {
                "xgboost": "#9c27b0",  # Purple for XGBoost
                "random_forest": "#2196f3",  # Blue for Random Forest
                "dqn": "#ff9800",  # Orange for DQN
                "rule_based": "#4caf50"  # Green for Rule-based
            }.get(model_key, "#607d8b")  # Default gray
            
            comparison_html += f"""
            <div style='flex: 1; min-width: 200px;' class='comparison-card'>
                <h4 style='color: {model_color}; margin-top: 0;'>{prediction['model_type']}</h4>
                <h3>Action: <span style='color: #4e8cff;'>{prediction['action_name']}</span></h3>
                <p>Confidence: <span style='font-weight: bold;'>{prediction['confidences'][prediction['action']]*100:.1f}%</span></p>
            </div>
            """
        
        comparison_html += "</div>"
        
        st.markdown(comparison_html, unsafe_allow_html=True)
        
        # Create a bar chart comparing model confidences for each action
        st.markdown("## Confidence Comparison")
        
        # Prepare data for chart
        chart_data = []
        for model_key, prediction in predictions.items():
            for action_idx, action_name in enumerate(ACTION_NAMES):
                chart_data.append({
                    'Model': prediction['model_type'],
                    'Action': action_name,
                    'Confidence': prediction['confidences'][action_idx] * 100,
                    'Selected': action_idx == prediction['action']
                })
        
        chart_df = pd.DataFrame(chart_data)
        
        # Create chart
        fig = px.bar(
            chart_df,
            x='Action',
            y='Confidence',
            color='Model',
            barmode='group',
            title='Model Confidence Comparison by Action',
            labels={'Confidence': 'Confidence (%)'},
            pattern_shape='Selected',
            pattern_shape_sequence=['', '/'],
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Main app function
def run_prediction():
    """
    Main function to make predictions and display results.
    """
    state = [demand, solar, battery_soc, price, time_of_day]
    
    # Make predictions with selected model(s)
    predictions = predict_with_selected_model(model_choice, state)
    
    # Display predictions
    display_predictions(predictions)
    
    # Add to history
    for model_key, prediction in predictions.items():
        st.session_state.history.append(prediction)
    
    # Show history if available
    if len(st.session_state.history) > 0:
        with st.expander("Prediction History", expanded=False):
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
                    'Model': p['model_type']
                })
            
            # Display history table
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)
            
            # Option to clear history
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

# Make prediction
if st.button("Predict Best Source", type="primary"):
    with st.spinner("Making predictions..."):
        run_prediction()

# Information about the models
with st.expander("About the Models"):
    st.markdown("""
    ### XGBoost
    XGBoost is a gradient boosting algorithm known for its performance and efficiency. In this application, 
    it's been trained on simulated data to predict the best energy source based on current conditions.
    
    ### Random Forest
    Random Forest is an ensemble learning method that operates by constructing multiple decision trees.
    It provides good accuracy and handles non-linear relationships well.
    
    ### DQN (Deep Q Network)
    DQN is a reinforcement learning algorithm that learns optimal actions through experience.
    Unlike the supervised models, it learns by interacting with the environment to maximize rewards.
    
    ### Rule-based
    A simple rule-based policy that makes decisions based on predefined logic. This serves as a baseline
    for comparison with the machine learning models.
    """)
