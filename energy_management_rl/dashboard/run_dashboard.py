import subprocess
import sys
import os
from pathlib import Path

def setup_and_run_dashboard():
    """Set up the environment and run the Streamlit dashboard."""
    print("Setting up environment for Energy Management RL Dashboard...")
    
    # Get the dashboard directory
    dashboard_dir = Path(__file__).parent.absolute()
    
    # Install required packages
    requirements_path = os.path.join(dashboard_dir, "requirements.txt")
    print(f"Installing dependencies from {requirements_path}...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return
    
    # Run the Streamlit dashboard
    dashboard_app_path = os.path.join(dashboard_dir, "app.py")
    print(f"Starting Streamlit dashboard from {dashboard_app_path}...")
    
    try:
        # Use streamlit run command
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_app_path])
    except Exception as e:
        print(f"Error running dashboard: {e}")
        return

if __name__ == "__main__":
    setup_and_run_dashboard()
