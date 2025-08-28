import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import os

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

# Generate synthetic training data
np.random.seed(42)
n_samples = 10000

# Generate features
demand = np.random.uniform(0, 20, n_samples)
solar = np.random.uniform(0, 15, n_samples)
battery_soc = np.random.uniform(0, 100, n_samples)
price = np.random.uniform(0.05, 0.50, n_samples)
time_of_day = np.random.randint(0, 24, n_samples)

# Create feature matrix
X = np.column_stack([demand, solar, battery_soc, price, time_of_day])

# Generate labels based on rules similar to the rule-based policy
def generate_label(row):
    demand, solar, battery_soc, price, time_of_day = row
    
    # Solar sufficient
    if solar >= demand:
        return 0  # Use Solar
    
    # High price and good battery
    if price > 0.25 and battery_soc > 20:
        return 1  # Use Battery
    
    # Low solar and low battery
    if solar < 0.5 and battery_soc < 30:
        return 2  # Use Grid
    
    # Solar + Battery viable
    if solar + (battery_soc/100 * 10) >= demand and battery_soc > 20:
        return 3  # Use Solar + Battery
    
    # Default to grid
    return 2

# Generate labels
y = np.apply_along_axis(generate_label, 1, X)

# Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
xgb_model.fit(X, y)

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
rf_model.fit(X, y)

# Save models
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Models trained and saved successfully!")
print("- XGBoost model saved to: models/xgboost_model.pkl")
print("- Random Forest model saved to: models/random_forest_model.pkl")
