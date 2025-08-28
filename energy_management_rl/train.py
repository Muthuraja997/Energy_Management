import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import matplotlib.pyplot as plt

from env import EnergyManagementEnv

# Create log directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)


def train_dqn(total_timesteps=100000, learning_rate=1e-4, gamma=0.99, exploration_fraction=0.2, 
              exploration_initial_eps=1.0, exploration_final_eps=0.05, train_freq=4, 
              gradient_steps=1, target_update_interval=1000, buffer_size=100000):
    """
    Train a DQN agent on the energy management environment.
    
    Args:
        total_timesteps: Total timesteps to train for
        learning_rate: Learning rate for the neural network
        gamma: Discount factor
        exploration_fraction: Fraction of total timesteps used for exploration
        exploration_initial_eps: Initial exploration rate
        exploration_final_eps: Final exploration rate
        train_freq: Update the model every `train_freq` steps
        gradient_steps: How many gradient steps to do after each rollout
        target_update_interval: Update the target network every `target_update_interval` steps
        buffer_size: Size of the replay buffer
    """
    # Create and wrap the environment
    env = EnergyManagementEnv()
    env = Monitor(env, log_dir)
    
    # Create evaluation environment
    eval_env = EnergyManagementEnv()
    eval_env = Monitor(eval_env, log_dir)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best/",
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{models_dir}/checkpoints/",
        name_prefix="dqn_energy_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Create the DQN agent
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        buffer_size=buffer_size,
        tensorboard_log=log_dir,
        verbose=1
    )
    
    # Train the agent
    print("Starting training...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="dqn_energy"
    )
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the final model
    model.save(f"{models_dir}/final_model")
    print(f"Final model saved to {models_dir}/final_model")
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model


def plot_learning_curve(log_file):
    """
    Plot the learning curve from the training logs.
    
    Args:
        log_file: Path to the monitor.csv log file
    """
    data = np.genfromtxt(log_file, delimiter=',', skip_header=1,
                         names=['r', 'l', 't'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['t'], data['r'], label='Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent for Energy Management')
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Total timesteps to train for')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate for the neural network')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Discount factor')
    args = parser.parse_args()
    
    # Train the agent
    model = train_dqn(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    # Plot learning curve
    log_file = os.path.join(log_dir, "monitor.csv")
    if os.path.exists(log_file):
        plot_learning_curve(log_file)
