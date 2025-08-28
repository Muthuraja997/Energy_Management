import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Action names for reference
ACTION_NAMES = ['Use Solar', 'Use Battery', 'Use Grid', 'Use Solar + Battery']

def train_random_forest_model(data_path):
    """
    Train a Random Forest model on the generated data.
    
    Parameters:
    - data_path: Path to the CSV file containing training data
    
    Returns:
    - Trained Random Forest model
    """
    print("Loading training data...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Check value counts to ensure we have all classes
    action_counts = df['action'].value_counts()
    print("Action distribution:")
    print(action_counts)
    
    # Split features and target
    X = df.drop('action', axis=1)
    y = df['action']
    
    # Split into training and validation sets
    print("Splitting into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Use all available cores
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
    plt.xlabel("Random Forest Feature Importance")
    plt.title("Feature Importance for Energy Source Selection")
    plt.tight_layout()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/rf_feature_importance.png")
    plt.close()
    print("Feature importance plot saved to 'data/rf_feature_importance.png'")
    
    return model

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train model
    model = train_random_forest_model("data/xgboost_training_data.csv")
    
    if model is not None:
        # Save model
        print("Saving model to 'models/random_forest_model.pkl'")
        with open("models/random_forest_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model saved successfully.")
    else:
        print("Model training failed. No model saved.")
