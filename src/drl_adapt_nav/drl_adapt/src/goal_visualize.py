#!/usr/bin/python

"""
Visualizes the current goal point in RViz.
Subscribes to the robot's goal topic and publishes a marker to display the goal position
in the simulation environment.
"""

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PoseStamped  # Goal pose
from visualization_msgs.msg import Marker  # Visualization marker

def goal_callback(goal_msg):
    """Callback to process and visualize the current goal."""
    # Set up header for marker
    h = Header()
    h.frame_id = "map"          # Reference frame
    h.stamp = rospy.Time.now()  # Current timestamp
    
    # Create goal marker
    goal_marker = Marker()
    goal_marker.header = h
    goal_marker.type = Marker.SPHERE  # Spherical marker
    goal_marker.action = Marker.ADD   # Add to visualization
    goal_marker.pose = goal_msg.pose  # Set position/orientation from goal
    goal_marker.scale.x = 1.8         # Size in x (meters)
    goal_marker.scale.y = 1.8         # Size in y
    goal_marker.scale.z = 1.8         # Size in z
    goal_marker.color.r = 1.0         # Red color
    goal_marker.color.g = 0.0         # No green
    goal_marker.color.b = 0.0         # No blue
    goal_marker.color.a = 0.5         # 50% transparency
    
    # Publish marker to RViz
    goal_vis_pub.publish(goal_marker)

if __name__ == '__main__':
    try:
        rospy.init_node('goal_vis')  # Initialize ROS node
        # Subscribe to current goal from move_base
        goal_sub = rospy.Subscriber("/move_base/current_goal", PoseStamped, goal_callback)
        # Publish marker to RViz
        goal_vis_pub = rospy.Publisher('goal_markers', Marker, queue_size=1, latch=True)
        rospy.spin()  # Keep node running
    except rospy.ROSInterruptException:
        pass  # Handle shutdown