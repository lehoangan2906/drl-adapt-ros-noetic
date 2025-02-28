#!usr/bin/python

# This script publishes velocity commands to the robot.

# It defines a ROS node 'mix_cmd_vel' that subscribes to velocity commands from the policy via '/drl_cmd_vel' that published by 'inference.py' or 'train.py' then republishes them to the robot's '/cmd_vel' topic for execution.


import rospy
import numpy as np
import geometry_msgs.msg
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose


# Class to manage velocity publishing
class VelSwitch:
    cmd_vel_vx = None   # Linear velocity
    cmd_Vel_wz = None   # Angular velocity


    # ROS objects
    drl_vel_sub = None  # Subscriber for DRL velocity commands
    cmd_vel_pub = None  # Publisher for robot velocity commands
    timer = None        # Timer for periodic publishing
    rate = None         # Publishing rate


    # Constructor
    def __init__(self):
        # Initialize velocities
        self.drl_vel = Twist()  # DRL command velocity
        self.cmd_vel_vx = 0.    # Initial linear velocity
        self.cmd_vel_wz = 0.    # Initial angular velocity

        
        # Setup ROS subscriber/publisher
        self.drl_vel_sub = rospy.Subscriber('/drl_cmd_vel', Twist, self.drl_callback)   # Subscribe to DRL policy velocity commands
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1, latch = False)   # Publish to robot
        self.rate = 80  # Set publish rate to 80hz


    # Callback to receive DRL velocity commands
    def drl_callback(self, drl_vel_msg):
        self.drl_vel = drl_vel_msg      # Store received message
        cmd_vel = Twist()               # Create new velocity message

        # Extract and store velocities
        self.cmd_vel_vx = self.drl_vel.linear.x
        self.cmd_vel_wz = self.drl_vel.angular.z


        # Publish velocity command
        cmd_vel.linear.x = self.cmd_vel_vx
        cmd_vel.angular.z = self.cmd_vel_wz
        self.cmd_vel_pub.publish(cmd_vel)       # Send to robot

if __name__ == '__main__':
    try:
        rospy.init_node('mix_cmd_vel')  # Initialize ROS node
        VelSwitch()                     # Create velocity switch instance
        rospy.spin()                    # Keep node running
    except rospy.ROSInterruptException:
        pass                            # Handle shutdown
