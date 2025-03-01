#!/usr/bin/python

"""
Publishes pedestrian kinematics from Pedsim in the robot's frame for DRL-VO.
Transforms ground truth pedestrian data from Gazebo into the base_footprint frame,
providing position and velocity for navigation in crowded scenes.
"""

import rospy
import tf  # Transform library
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose
from pedsim_msgs.msg import TrackedPersons, TrackedPerson  # Pedsim messages
from gazebo_msgs.srv import GetModelState  # Gazebo service
import numpy as np


"""Class to process and publish pedestrian kinematics."""
class TrackPed:
    def __init__(self):
        """Initialize subscribers and publishers."""

        # ROS subscribers and services
        self.ped_sub = rospy.Subscriber('/pedsim_visualizer/tracked_persons', TrackedPersons, self.ped_callback)    # Pedsim data
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)                       # Robot state from Gazebo
        
        # ROS publisher
        self.track_ped_pub = rospy.Publisher('/track_ped', TrackedPersons, queue_size=10)  # Publish transformed pedestrian data
    

    """Get robot's current pose from Gazebo."""
    def get_robot_states(self):
        robot = None
        rospy.wait_for_service("/gazebo/get_model_state")           # Wait for service
        try:
            robot = self.get_state_service('mobile_base', 'world')  # Get robot state
        except rospy.ServiceException:
            rospy.logwarn("/gazebo/get_model_state service call failed")
        return robot
    

    """Transform and publish pedestrian data in robot frame."""
    def ped_callback(self, peds_msg):
        peds = peds_msg                 # Pedsim pedestrian data
        robot = self.get_robot_states()  # Get robot pose
        
        if robot is not None:
            # Robot position and orientation
            robot_pos = np.zeros(3)
            robot_pos[:2] = np.array([robot.pose.position.x, robot.pose.position.y])  # x, y position
            robot_quat = (robot.pose.orientation.x, robot.pose.orientation.y, 
                        robot.pose.orientation.z, robot.pose.orientation.w)
            (_, _, robot_pos[2]) = tf.transformations.euler_from_quaternion(robot_quat)  # Yaw angle
            robot_vel = np.array([robot.twist.linear.x, robot.twist.linear.y])           # Robot velocity
            
            # Transformation matrices
            map_R_robot = np.array([[np.cos(robot_pos[2]), -np.sin(robot_pos[2])],
                                    [np.sin(robot_pos[2]), np.cos(robot_pos[2])]])  # Rotation matrix
            map_T_robot = np.array([[np.cos(robot_pos[2]), -np.sin(robot_pos[2]), robot_pos[0]],
                                    [np.sin(robot_pos[2]), np.cos(robot_pos[2]), robot_pos[1]],
                                    [0, 0, 1]])         # Homogeneous transform
            robot_R_map = np.linalg.inv(map_R_robot)    # Inverse rotation
            robot_T_map = np.linalg.inv(map_T_robot)    # Inverse transform
            
            # Transform pedestrian data
            tracked_peds = TrackedPersons()
            tracked_peds.header.frame_id = 'base_footprint'  # Robot frame
            tracked_peds.header.stamp = rospy.Time.now()     # Current timestamp

            for ped in peds.tracks:
                tracked_ped = TrackedPerson()
                # Pedestrian position and velocity in world frame
                ped_pos = np.array([ped.pose.pose.position.x, ped.pose.pose.position.y, 1])
                ped_vel = np.array([ped.twist.twist.linear.x, ped.twist.twist.linear.y])
                
                # Transform to robot frame
                ped_pos_in_robot = np.matmul(robot_T_map, ped_pos.T)  # Position
                ped_vel_in_robot = np.matmul(robot_R_map, ped_vel.T)  # Velocity

                # Update pedestrian message
                tracked_ped = ped
                tracked_ped.pose.pose.position.x = ped_pos_in_robot[0]
                tracked_ped.pose.pose.position.y = ped_pos_in_robot[1]
                tracked_ped.twist.twist.linear.x = ped_vel_in_robot[0]
                tracked_ped.twist.twist.linear.y = ped_vel_in_robot[1]
                tracked_peds.tracks.append(tracked_ped)

            # Publish transformed data
            self.track_ped_pub.publish(tracked_peds)

if __name__ == '__main__':
    try:
        rospy.init_node('track_ped')  # Initialize ROS node
        tp = TrackPed()               # Create pedestrian tracker instance
        rospy.spin()                  # Keep node running
    except rospy.ROSInterruptException:
        pass  # Handle shutdown