#!/usr/bin/python

"""
Trains the DRL-ADAPT policy using PPO in a Gazebo simulation environment.
Sets up the Turtlebot Gym environment, runs reinforcement learning training.
"""

import os
import gym
import rospy
import numpy as np
import turtlebot_gym                                    # Custom Turtlebot Gym environment
from custom_cnn_full import *                           # Custom CNN feature extractor
from stable_baselines3 import PPO                       # PPO algorithm
from stable_baselines3.common.monitor import Monitor    # For logging
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


"""Callback to save the best model based on training reward."""
class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq                            # Check every N steps
        self.log_dir = log_dir                                  # Directory for logs/models
        self.save_path = os.path.join(log_dir, 'best_model')    # Save path
        self.best_mean_reward = -np.inf                         # Track best reward

    """Create save directory if needed."""
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    """Check and save model if reward improves."""
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')   # Load reward data
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])                     # Mean of last 100 episodes
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)                 # Save improved model
            if self.n_calls % 20000 == 0:                           # Save every 20k steps
                path = self.save_path + '_model' + str(self.n_calls)
                self.model.save(path)                               # Periodic checkpoint
        return True
    

# Initialize ROS node
rospy.init_node('drl_adapt_train', anonymous=True, log_level=rospy.WARN)


# Create log directory
log_dir = rospy.get_param('~log_dir', "./runs/")    # Get log directory path
os.makedirs(log_dir, exist_ok=True)                 # Create directory if needed


# Setup Gym environment
env = gym.make('drl-nav-v0')                        # Create Turtlebot Gym environment
env = Monitor(env, log_dir)                         # Monitor environment for logging
obs = env.reset()                                   # Reset environment to get initial observation


# Specify that CustomCNN is the feature extractor, processing observations into 256 features
policy_kwargs = dict(
    features_extractor_class=CustomCNN,                 # Use custom CNN
    features_extractor_kwargs=dict(features_dim=256),   # 256 output features
    net_arch=[dict(pi=[256], vf=[128])]                 # Actor and value network hidden layers size
)

# raw training:
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, learning_rate=1e-3, verbose=2, tensorboard_log=log_dir, n_steps=512, n_epochs=10, batch_size=128)

# Continue training: Loads pre-trained model and links it to the environment
# kwargs = {'tensorboard_log': log_dir, 'verbose': 2, 'n_epochs':10, 'n_steps': 512, 'batch_size': 128, 'learning_rate': 5e-5}
# model_file = rospy.get_param('~model_file', "./model/drl_pre_train.zip")    # Pre-trained model path
# model = PPO.load(model_file, env=env, policy_kwargs=policy_kwargs, **kwargs) # Load pre-trained model with env and policy_kwargs


# Train the model
callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)  # Save best model callback
model.learn(total_timesteps=2000000, log_interval=5, tb_log_name='drl_vo_policy', callback=callback, reset_num_timesteps=True)  # Run training


# Save final model
model.save("drl_vo_model")  # Save trained model
print("Training finished.")
env.close()  # Close environment