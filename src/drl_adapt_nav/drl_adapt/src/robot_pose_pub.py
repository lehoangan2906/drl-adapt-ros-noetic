#!/usr/bin/python

"""
Publishes the robot's current pose in the map frame.
Uses TF to get the robot's position and orientation, providing pose data for
navigation and tracking in the Gazebo simulation.
"""

import rospy
import geometry_msgs.msg
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose  # Pose message
import tf  # Transform library
from geometry_msgs.msg import Point
import numpy as np

def robot_pose_pub():
    """Publish the robot's pose at 30 Hz."""
    rospy.init_node('robot_pose', anonymous=True)  # Initialize ROS node
    tf_listener = tf.TransformListener()           # Listener for TF transforms
    robot_pose_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=1)  # Publisher for pose
    rate = rospy.Rate(30)  # Publish at 30 Hz
    
    while not rospy.is_shutdown():
        trans = rot = None
        # Get robot pose from TF (map to base_footprint)
        try:
            (trans, rot) = tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn('Could not get robot pose')  # Log warning on failure
            trans = [-1, -1, -1]  # Default invalid translation
            rot = [-1, -1, -1, -1]  # Default invalid rotation
        
        # Create and populate PoseStamped message
        rob_pos = PoseStamped()
        rob_pos.header.stamp = rospy.Time.now()  # Current timestamp
        rob_pos.header.frame_id = '/map'  # Reference frame
        rob_pos.pose.position.x = trans[0]  # X position
        rob_pos.pose.position.y = trans[1]  # Y position
        rob_pos.pose.position.z = trans[2]  # Z position
        rob_pos.pose.orientation.x = rot[0]  # Quaternion x
        rob_pos.pose.orientation.y = rot[1]  # Quaternion y
        rob_pos.pose.orientation.z = rot[2]  # Quaternion z
        rob_pos.pose.orientation.w = rot[3]  # Quaternion w
        
        # Publish robot pose
        robot_pose_pub.publish(rob_pos)
        rate.sleep()  # Maintain 30 Hz rate

if __name__ == '__main__':
    try:
        robot_pose_pub()  # Start publishing pose
    except rospy.ROSInterruptException:
        pass  # Handle shutdown