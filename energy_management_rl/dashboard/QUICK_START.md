# Quick Start Guide for the Energy Management Dashboard

## Running the Dashboard

1. Make sure you have installed all requirements:
   ```
   pip install -r dashboard/requirements.txt
   ```

2. Run the dashboard:
   ```
   streamlit run dashboard/app.py
   ```

3. The dashboard should open automatically in your browser. If not, go to:
   ```
   http://localhost:8501
   ```

## Using the Dashboard

1. **Load a Model**:
   - From the sidebar, select a pre-trained model or upload your own
   - Click the "Load Selected Model" button

2. **Run a Simulation**:
   - Click "Reset Environment" to start with a fresh environment
   - Use "Step Once" to see the agent take individual steps
   - Use "Run Episode" to automatically run a full 24-hour cycle

3. **View Results**:
   - Monitor real-time charts showing demand, solar generation, battery status, etc.
   - See the detailed breakdown of each step in the results table
   - Check summary statistics in the sidebar

## Troubleshooting

- If you get import errors, make sure all dependencies are installed
- If the model doesn't load, check that the path is correct
- If visualizations don't update, try clicking "Reset Environment" and run again

## Additional Information

For more detailed information about the dashboard and the reinforcement learning model, please refer to the main README.md file and the documentation in the code.
