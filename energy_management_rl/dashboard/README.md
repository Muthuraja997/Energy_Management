# Energy Management RL Dashboard

This dashboard provides an interactive visualization of the reinforcement learning energy management system. It allows you to observe the agent's decision-making process in real-time and analyze its performance.

## Features

- **Real-time Visualization**: See the current state of the environment and the agent's actions.
- **Interactive Charts**: Monitor demand, solar generation, battery state of charge, and more.
- **Simulation Controls**: Step through a simulation one step at a time or run a complete episode.
- **Flexible Model Loading**: Load different trained models to compare their performance.
- **Detailed Results**: View comprehensive statistics and metrics about the agent's performance.

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

The dashboard requires Streamlit and other dependencies. These will be automatically installed when you run the dashboard.

### Running the Dashboard

1. Run the dashboard launcher:

```bash
python run_dashboard.py
```

2. This will:
   - Install all required dependencies
   - Start the Streamlit server
   - Open the dashboard in your default web browser

## Using the Dashboard

1. **Load a Model**:
   - Select a pre-trained model from the dropdown or upload your own
   - Click "Load Selected Model" or "Load Uploaded Model"

2. **Run a Simulation**:
   - Click "Reset Environment" to start fresh
   - Use "Step Once" to advance one step at a time
   - Or use "Run Episode" to run through a complete 24-hour episode

3. **Analyze Results**:
   - Explore the various charts to understand the agent's behavior
   - View the detailed results table for step-by-step analysis
   - Check the summary metrics for overall performance

## Troubleshooting

If you encounter any issues:

1. Make sure you've activated your virtual environment if you're using one
2. Verify that you have a trained model available (either in the models directory or uploaded)
3. Check the terminal output for any error messages

## Tech Stack

- **Streamlit**: For the interactive web interface
- **Plotly**: For dynamic, interactive charts
- **Pandas**: For data manipulation and tables
- **Stable-Baselines3**: For loading and using the trained RL model
- **Gymnasium**: For the reinforcement learning environment
