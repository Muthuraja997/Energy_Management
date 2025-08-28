import subprocess
import sys
import os

def setup_environment():
    """Set up the environment by installing dependencies."""
    print("Setting up environment for Energy Management RL project...")
    
    # Check if pip is available
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        print("Error: pip is not available. Please install pip first.")
        sys.exit(1)
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    print("Installing required packages...")
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    
    # Create necessary directories
    print("Creating necessary directories...")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    
    print("Setup completed successfully!")
    print("\nYou can now run:")
    print("  python test_env_minimal.py  # Test the environment")
    print("  python train.py             # Train the agent")
    print("  python inference.py         # Run inference with a trained agent")
    print("  python run.py --mode all    # Run the entire pipeline")


if __name__ == "__main__":
    setup_environment()
