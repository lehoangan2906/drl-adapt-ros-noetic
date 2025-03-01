#!/usr/bin/python

"""
Runs inference for the DRL-ADAPT policy in real-time navigation.

Loads a trained model, processes CNN_data inputs, and publishes velocity commands
for the robot to navigate crowded scenes in the Gazebo simulation.
"""


import os
import tf 
import sys 
import rospy
import numpy as np
import message_filters
from custom_cnn_full import *
from cnn_msgs.msg import CNN_data
from stable_baselines3 import PPO
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan


# Dictionary holds keyword arguments for configuring a policy.
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)


class DrlInference:
    def __init__(self):
        """Initialize inference node and load trained model."""
        # Data storage
        self.ped_pos = []       # Pedestrian kinematic map
        self.scan = []          # Lidar scan data
        self.goal = []          # Sub-goal position
        self.vx = 0.0           # Linear velocity
        self.wz = 0.0           # Angular velocity
        self.model = None       # Trained model

        # Load pre-trained model
        model_file = rospy.get_param('~model_file', "./model/drl_adapt.zip") # get model file path
        self.model = PPO.load(model_file)  # Load PPO model
        print("Finish loading model.")

        # ROS subscriber and publisher
        self.cnn_data_sub = rospy.Subscriber("/cnn_data", CNN_data, self.cnn_data_callback)  # Subscribe to pre-processed sensory data
        self.cmd_vel_pub = rospy.Publisher('/drl_cmd_vel', Twist, queue_size=10, latch=False)  # Publish velocity commands


    """Process sensory data and generate velocity commands."""
    def cnn_data_callback(self, cnn_data_msg):

        # Extract sensory data
        self.ped_pos = cnn_data_msg.ped_pos_map  # 80x80x2 pedestrian kinematics
        self.scan = cnn_data_msg.scan            # Lidar scan history
        self.goal = cnn_data_msg.goal_cart       # Sub-goal in robot frame
        cmd_vel = Twist()                        # Velocity command message


        # Check minimum distance to obstacles
        scan = np.array(self.scan[-540:-180])  # Last 360 points of scan
        scan = scan[scan != 0]  # Filter out zeros
        min_scan_dist = np.amin(scan) if scan.size != 0 else 10  # Default to 10 if no valid points


        # Decision logic for velocity
        if np.linalg.norm(self.goal) <= 0.9:  # Goal reached (within 0.9m)
            cmd_vel.linear.x = 0
            cmd_vel.angular.z = 0


        elif min_scan_dist <= 0.4:            # Obstacle too close (0.4m)
            cmd_vel.linear.x = 0
            cmd_vel.angular.z = 0.7           # Turn to avoid


        else:
            # Process and normalize pedestrian data
            v_min, v_max = -2, 2
            self.ped_pos = np.array(self.ped_pos, dtype=np.float32)
            self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1) # Normalize to [-1, 1]


            # Process and normalize scan
            temp = np.array(self.scan, dtype=np.float32)
            scan_avg = np.zeros((20, 80))           # 10 timestamps, 80 bins

            for n in range(10):
                scan_tmp = temp[n*720:(n+1)*720]    # One timestamp
                for i in range(80):
                    scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])        # Min pooling
                    scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])     # Avg pooling

            scan_avg = scan_avg.reshape(1600)                               # Flatten
            scan_avg_map = np.matlib.repmat(scan_avg, 1, 4)                 # Repeat 4 times
            self.scan = scan_avg_map.reshape(6400)                          # 6400 elements
            s_min, s_max = 0, 30
            self.scan = 2 * (self.scan - s_min) / (s_max - s_min) + (-1)    # Normalize to [-1, 1]


            # Normalize goal
            g_min, g_max = -2, 2
            goal_original = np.array(self.goal, dtype=np.float32)
            self.goal = 2 * (goal_original - g_min) / (g_max - g_min) + (-1) # Normalize to [-1, 1]


            # Create observation for CNN
            self.observation = np.concatenate((self.ped_pos, self.scan, self.goal), axis=None)  # 19202 elements


            # Predict action with DRL-VO model
            action, _states = self.model.predict(self.observation)  # Outputs [vx, wz] in [-1, 1]


            # Denormalize action to robot velocities
            vx_min, vx_max = 0, 0.5
            vz_min, vz_max = -2, 2
            cmd_vel.linear.x = (action[0] + 1) * (vx_max - vx_min) / 2 + vx_min   # Scale to [0, 0.5] m/s
            cmd_vel.angular.z = (action[1] + 1) * (vz_max - vz_min) / 2 + vz_min  # Scale to [-2, 2] rad/s


        # Publish valid velocity commands
        if not np.isnan(cmd_vel.linear.x) and not np.isnan(cmd_vel.angular.z):
            self.cmd_vel_pub.publish(cmd_vel)  # Send to /drl_cmd_vel


if __name__ == '__main__':
    rospy.init_node('drl_inference')  # Initialize ROS node
    drl_infe = DrlInference()        # Create inference instance
    rospy.spin()                     # Keep node running