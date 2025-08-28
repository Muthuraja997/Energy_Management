import numpy as np
import pandas as pd
import pickle
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Action names for reference
ACTION_NAMES = ['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar + Battery']

def train_xgboost_model(data_path):
    """
    Train an XGBoost model on the generated data.
    
    Parameters:
    - data_path: Path to the CSV file containing training data
    
    Returns:
    - Trained XGBoost model
    """
    print("Loading training data...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Check value counts to ensure we have all classes
    action_counts = df['action'].value_counts()
    print("Action distribution:")
    print(action_counts)
    
    # Check if we have all actions represented
    missing_actions = []
    for action in range(4):
        count = action_counts.get(action, 0)
        if count == 0:
            missing_actions.append(action)
        print(f"Action {action} ({ACTION_NAMES[action]}): {count} samples ({count/len(df)*100:.1f}% if any)")
    
    if missing_actions:
        print(f"Warning: Actions {missing_actions} are not represented in the training data!")
        print("Adding synthetic samples for missing actions...")
        
        # Add synthetic samples for missing actions
        for action in missing_actions:
            print(f"Adding synthetic samples for action {action}")
            
            # Create synthetic features based on action type
            if action == 0:  # Solar
                synthetic_samples = pd.DataFrame({
                    'demand': np.random.uniform(1, 3, 100),
                    'solar': np.random.uniform(5, 10, 100),
                    'battery_soc': np.random.uniform(20, 90, 100),
                    'price': np.random.uniform(0.1, 0.3, 100),
                    'time_of_day': np.random.uniform(10, 14, 100),
                    'action': [action] * 100
                })
            elif action == 1:  # Battery
                synthetic_samples = pd.DataFrame({
                    'demand': np.random.uniform(2, 5, 100),
                    'solar': np.random.uniform(0, 1, 100),
                    'battery_soc': np.random.uniform(50, 90, 100),
                    'price': np.random.uniform(0.3, 0.5, 100),
                    'time_of_day': np.random.uniform(18, 22, 100),
                    'action': [action] * 100
                })
            elif action == 2:  # Grid
                synthetic_samples = pd.DataFrame({
                    'demand': np.random.uniform(8, 15, 100),
                    'solar': np.random.uniform(0, 1, 100),
                    'battery_soc': np.random.uniform(0, 30, 100),
                    'price': np.random.uniform(0.1, 0.2, 100),
                    'time_of_day': np.random.uniform(0, 23, 100),
                    'action': [action] * 100
                })
            elif action == 3:  # Solar + Battery
                synthetic_samples = pd.DataFrame({
                    'demand': np.random.uniform(5, 8, 100),
                    'solar': np.random.uniform(3, 5, 100),
                    'battery_soc': np.random.uniform(40, 90, 100),
                    'price': np.random.uniform(0.3, 0.5, 100),
                    'time_of_day': np.random.uniform(10, 14, 100),
                    'action': [action] * 100
                })
                
            # Add to the original dataframe
            df = pd.concat([df, synthetic_samples], ignore_index=True)
    
        # Check distribution again after adding synthetic samples
        print("\nAction distribution after adding synthetic samples:")
        action_counts = df['action'].value_counts()
        for action in range(4):
            count = action_counts.get(action, 0)
            print(f"Action {action} ({ACTION_NAMES[action]}): {count} samples ({count/len(df)*100:.1f}%)")
    
    # Split features and target
    X = df.drop('action', axis=1)
    y = df['action']
    
    # Double-check that all classes are represented
    unique_classes = np.unique(y)
    print(f"Unique action classes in dataset: {unique_classes}")
    
    if len(unique_classes) < 4:
        print("Error: Still not all action classes are represented. Cannot train model.")
        return None
    
    # Split into training and validation sets
    print("Splitting into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softproba',
        num_class=4,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model performance...")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=ACTION_NAMES))
    
    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Plot feature importance
    print("Generating feature importance plot...")
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx])
    plt.xlabel("XGBoost Feature Importance")
    plt.title("Feature Importance for Energy Source Selection")
    plt.tight_layout()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/feature_importance.png")
    plt.close()
    print("Feature importance plot saved to 'data/feature_importance.png'")
    
    return model

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train model
    model = train_xgboost_model("data/xgboost_training_data.csv")
    
    if model is not None:
        # Save model
        print("Saving model to 'models/xgboost_model.pkl'")
        with open("models/xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model saved successfully.")
    else:
        print("Model training failed. No model saved.")
