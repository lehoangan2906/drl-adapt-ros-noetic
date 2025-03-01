#!/usr/bin/python

"""
Defines the Gym environment (drl-nav-v0) for training the DRL policy.
Manages simulation, processes sensory data from CNN_data, computes rewards, and provides
observations for PPO to navigate the Turtlebot through crowded scenes with Pedsim.
"""

import numpy as np
import random
import math
from scipy.optimize import linprog, minimize
import threading
import rospy
import gym
from gym.utils import seeding
from gym import spaces
from .gazebo_connection import GazeboConnection  # Gazebo control
from std_msgs.msg import Float64, Empty, Bool
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import Pose, Twist, Point, PoseStamped, PoseWithCovarianceStamped
import time
from kobuki_msgs.msg import BumperEvent
from actionlib_msgs.msg import GoalStatusArray
from pedsim_msgs.msg import TrackedPersons, TrackedPerson
from cnn_msgs.msg import CNN_data

class DRLNavEnv(gym.Env):
    """Gym environment for DRL navigation training."""
    def __init__(self):
        """Initialize environment, simulation, and ROS interfaces."""
        rospy.logdebug("START init DRLNavEnv")
        self.seed()  # Set random seed
        
        # Robot and goal parameters
        self.ROBOT_RADIUS = 0.3  # Robot size
        self.GOAL_RADIUS = 0.3  # Goal proximity threshold
        self.DIST_NUM = 10      # Distance history length (for reward)
        self.pos_valid_flag = True  # Position validity flag
        self.bump_flag = False  # Collision flag
        self.bump_num = 0       # Collision count
        self.dist_to_goal_reg = np.zeros(self.DIST_NUM)  # Distance history
        self.num_iterations = 0  # Step counter

        # Action and observation spaces
        self.max_linear_speed = 0.5          # Max linear velocity
        self.max_angular_speed = 2           # Max angular velocity
        self.high_action = np.array([1, 1])  # Normalized action range
        self.low_action = np.array([-1, -1])
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)    # Action space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19202,), dtype=np.float32)           # Observation space

        # State variables
        self.cnn_data = CNN_data()  # Input data
        self.ped_pos = []  # Pedestrian kinematics
        self.scan = []     # Lidar scan
        self.goal = []     # Sub-goal
        self.init_pose = Pose()  # Initial robot pose
        self.curr_pose = Pose()  # Current robot pose
        self.curr_vel = Twist()  # Current velocity
        self.goal_position = Point()  # Global goal
        self.info = {}  # Episode info
        self._episode_done = False  # Done flag
        self._goal_reached = False  # Goal reached flag
        self._reset = True  # Reset flag
        self.mht_peds = TrackedPersons()  # Tracked pedestrians

        # Gazebo control
        self.gazebo = GazeboConnection(start_init_physics_parameters=True, reset_world_or_sim="WORLD") # Gazebo interface
        self.gazebo.unpauseSim()  # Start simulation

        # ROS subscribers
        self._map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_callback)  # Map data
        self._cnn_data_sub = rospy.Subscriber("/cnn_data", CNN_data, self._cnn_data_callback, queue_size=1, buff_size=2**24)  # CNN inputs
        self._robot_pos_sub = rospy.Subscriber("/robot_pose", PoseStamped, self._robot_pose_callback)  # Robot pose
        self._robot_vel_sub = rospy.Subscriber('/odom', Odometry, self._robot_vel_callback)  # Robot velocity
        self._final_goal_sub = rospy.Subscriber("/move_base/current_goal", PoseStamped, self._final_goal_callback)  # Global goal
        self._goal_status_sub = rospy.Subscriber("/move_base/status", GoalStatusArray, self._goal_status_callback)  # Goal status
        self._ped_sub = rospy.Subscriber("/track_ped", TrackedPersons, self._ped_callback)  # Pedestrian data

        # ROS publishers
        self._cmd_vel_pub = rospy.Publisher('/drl_cmd_vel', Twist, queue_size=5)  # Velocity commands
        self._initial_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1, latch=True)  # Initial goal
        self._set_robot_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)  # Set robot state
        self._initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)  # Initial pose

        self._check_all_systems_ready()  # Verify ROS connections
        self.gazebo.pauseSim()  # Pause simulation
        rospy.logdebug("Finished TurtleBot2Env INIT...")

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Execute one step: apply action, get observation and reward."""
        self.gazebo.unpauseSim()  # Resume simulation
        self._take_action(action)  # Apply velocity command
        self.gazebo.pauseSim()    # Pause simulation
        obs = self._get_observation()  # Get processed observation
        reward = self._compute_reward()  # Calculate reward
        done = self._is_done(reward)     # Check if episode ends
        info = self._post_information()  # Episode info
        return obs, reward, done, info

    def reset(self):
        """Reset environment for a new episode."""
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._reset_sim()  # Reset simulation state
        obs = self._get_observation()  # Initial observation
        info = self._post_information()  # Initial info
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs

    def _reset_sim(self):
        """Reset Gazebo simulation."""
        self.gazebo.unpauseSim()
        self._set_init()  # Set initial conditions
        self.gazebo.pauseSim()
        return True

    def _set_init(self):
        """Set initial robot pose and random goal."""
        self._cmd_vel_pub.publish(Twist())  # Reset velocity
        if self._reset:
            self._reset = False
            self._check_all_systems_ready()  # Ensure ROS readiness
            self.pos_valid_flag = False
            while not self.pos_valid_flag:  # Ensure valid starting position
                seed_initial_pose = random.randint(0, 18)
                self._set_initial_pose(seed_initial_pose)  # Random initial pose
                time.sleep(4)  # Wait for stabilization
                x, y = self.curr_pose.position.x, self.curr_pose.position.y
                self.pos_valid_flag = self._is_pos_valid(x, y, self.ROBOT_RADIUS, self.map)
        [goal_x, goal_y, goal_yaw] = self._publish_random_goal()  # Set random goal
        time.sleep(1)
        self._check_all_systems_ready()
        self.init_pose = self.curr_pose                 # Store initial pose
        self.goal_position.x = goal_x                   # Store global goal
        self.goal_position.y = goal_y
        self.bump_flag = False                          # Reset collision
        self.num_iterations = 0                         # Reset step counter
        self.dist_to_goal_reg = np.zeros(self.DIST_NUM) # Reset distance history
        self._episode_done = False                      # Reset done flag
        return self.init_pose, self.goal_position


    """ Chooses one of several predefined initial positions based on a seed_initial_pose value
        which determines where in the environment the robot will start from."""
    def _set_initial_pose(self, seed_initial_pose):
        if(seed_initial_pose == 0):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(1, 1, 0)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(1, 0, 0.1377)
        elif(seed_initial_pose == 1):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(14, 7, 1.5705)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())     
            ''' 
            # reset robot inital pose in rviz:
            self._pub_initial_position(12.61, 7.5, 1.70)
        elif(seed_initial_pose == 2):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(1, 16, 0)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())   
            '''     
            # reset robot inital pose in rviz:
            self._pub_initial_position(-1, 14.5, 0.13)
        elif(seed_initial_pose == 3):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(14, 22.5, -1.3113)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
            self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(10.7, 23, -1.16)
        elif(seed_initial_pose == 4):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(4, 4, 1.5705)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(3.6, 3.1, 1.70)
        elif(seed_initial_pose == 5):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(2, 9, 0)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(1, 8, 0.13)
        elif(seed_initial_pose == 6):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(30, 9, 3.14)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(28, 11.4, 3.25)
        elif(seed_initial_pose == 7):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(25, 17, 3.14)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(22.5, 18.8, 3.25)
        elif(seed_initial_pose == 8):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(5, 8, 0)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(3.96, 7.47, 0.137)
        elif(seed_initial_pose == 9):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(10, 12, 0)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(8.29, 12.07, 0.116)
        elif(seed_initial_pose == 10):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(14, 15, 1.576)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(11.75, 15.14, 1.729)
        elif(seed_initial_pose == 11):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(18.5, 15.7, 3.14)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(16.06, 16.7, -2.983)
        elif(seed_initial_pose == 12):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(18.5, 11.3, 3.14)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(16.686, 12.434, -2.983)
        elif(seed_initial_pose == 13):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(14, 11.3, 3.14)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(12.142, 12.1, -2.983)
        elif(seed_initial_pose == 14):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(12.5, 13.2, 0.78)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(10.538, 13.418, 0.9285)
        elif(seed_initial_pose == 15):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(12.07, 16.06, 0)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(9.554, 16.216, 0.143)
        elif(seed_initial_pose == 16):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(21, 14, -1.576)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(18.462, 15.352, -1.4276)
        elif(seed_initial_pose == 17):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(14, 22.5, 1.576)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(10.586, 22.836, 1.7089)
        elif(seed_initial_pose == 18):
            # set turtlebot initial pose in gazebo:
            self._pub_initial_model_state(18, 8.5, -1.576)
            time.sleep(1)
            '''
            # reset robot odometry:
            timer = time.time()
            while time.time() - timer < 0.5:
                self._reset_odom_pub.publish(Empty())
            '''
            # reset robot inital pose in rviz:
            self._pub_initial_position(16.551, 9.630, -1.4326)



    def _get_observation(self):
        """Process sensory data into observation for PPO."""
        self.ped_pos = self.cnn_data.ped_pos_map  # Pedestrian kinematics
        self.scan = self.cnn_data.scan           # Lidar scan history
        self.goal = self.cnn_data.goal_cart      # Sub-goal
        
        # Normalize pedestrian data
        v_min, v_max = -2, 2
        self.ped_pos = np.array(self.ped_pos, dtype=np.float32)
        self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1)

        # Process scan into 80x80 grid
        temp = np.array(self.scan, dtype=np.float32)
        scan_avg = np.zeros((20, 80))  # 10 timestamps, 80 bins
        for n in range(10):
            scan_tmp = temp[n*720:(n+1)*720]
            for i in range(80):
                scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])    # Min pooling
                scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])  # Avg pooling
        scan_avg = scan_avg.reshape(1600)
        scan_avg_map = np.matlib.repmat(scan_avg, 1, 4)
        self.scan = scan_avg_map.reshape(6400)
        s_min, s_max = 0, 30
        self.scan = 2 * (self.scan - s_min) / (s_max - s_min) + (-1)

        # Normalize goal
        g_min, g_max = -2, 2
        self.goal = np.array(self.goal, dtype=np.float32)
        self.goal = 2 * (self.goal - g_min) / (g_max - g_min) + (-1)

        # Combine into observation
        self.observation = np.concatenate((self.ped_pos, self.scan, self.goal), axis=None)  # 19202 elements
        return self.observation

    def _take_action(self, action):
        """Apply normalized action as velocity command."""
        cmd_vel = Twist()
        vx_min, vx_max = 0, 0.5
        vz_min, vz_max = -2, 2
        cmd_vel.linear.x = (action[0] + 1) * (vx_max - vx_min) / 2 + vx_min    # Denormalize to [0, 0.5] m/s
        cmd_vel.angular.z = (action[1] + 1) * (vz_max - vz_min) / 2 + vz_min   # Denormalize to [-2, 2] rad/s
        rate = rospy.Rate(20)
        for _ in range(1):
            self._cmd_vel_pub.publish(cmd_vel)  # Publish velocity
            rate.sleep()

    def _compute_reward(self):
        """Calculate reward based on navigation progress."""
        r_arrival = 20    # Reward for reaching goal
        r_waypoint = 3.2  # Reward for progress
        r_collision = -20 # Penalty for collision
        r_scan = -0.2     # Penalty for near obstacles
        r_angle = 0.6     # Reward for heading
        r_rotation = -0.1 # Penalty for rotation
        
        r_g = self._goal_reached_reward(r_arrival, r_waypoint)  # Goal progress
        r_c = self._obstacle_collision_punish(self.cnn_data.scan[-720:], r_scan, r_collision)  # Collision penalty
        r_w = self._angular_velocity_punish(self.curr_vel.angular.z, r_rotation, 1)  # Rotation penalty
        r_t = self._theta_reward(self.goal, self.mht_peds, self.curr_vel.linear.x, r_angle, np.pi/6)  # Heading reward
        return r_g + r_c + r_t + r_w  # Total reward

    def _is_done(self, reward):
        """Check if episode ends (goal reached, collision, timeout)."""
        self.num_iterations += 1  # Increment step counter
        dist_to_goal = np.linalg.norm([self.curr_pose.position.x - self.goal_position.x,
                                        self.curr_pose.position.y - self.goal_position.y,
                                        self.curr_pose.position.z - self.goal_position.z])
        if dist_to_goal <= self.GOAL_RADIUS:  # Goal reached
            self._cmd_vel_pub.publish(Twist())
            self._episode_done = True
            return True
        scan = self.cnn_data.scan[-720:]  # Latest scan
        min_scan_dist = np.amin(scan[scan!=0])
        if min_scan_dist <= self.ROBOT_RADIUS and min_scan_dist >= 0.02:  # Collision
            self.bump_num += 1
        if self.bump_num >= 3:  # Too many collisions
            self._cmd_vel_pub.publish(Twist())
            self.bump_num = 0
            self._episode_done = True
            self._reset = True
            return True
        if self.num_iterations > 512:  # Max steps reached
            self._cmd_vel_pub.publish(Twist())
            self._episode_done = True
            self._reset = True
            return True
        return False