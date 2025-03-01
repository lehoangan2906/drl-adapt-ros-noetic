#!/usr/bin/python

"""
Publishes a sequence of goal points for the robot to navigate in simulation.
Sends goals to move_base, tracks success and metrics (time, distance), aiding evaluation
of navigation performance in crowded scenes.
"""

import rospy
import math
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  # Goal action
from actionlib_msgs.msg import GoalStatus                    # Goal status
from geometry_msgs.msg import Pose, Point, Quaternion        # Pose data
from tf.transformations import quaternion_from_euler         # Quaternion conversion
from math import hypot
from kobuki_msgs.msg import BumperEvent                      # Bumper sensor
from nav_msgs.msg import Odometry                            # Odometry data

GOAL_NUM = 3  # Number of coordinates per goal (x, y, z)

class MoveBaseSeq:
    """Class to manage goal sequence publishing and navigation metrics."""
    def __init__(self):
        # Metrics initialization
        self.success_num = 0      # Count of successful goals
        self.total_time = 0       # Total navigation time
        self.start_time = 0       # Start timestamp
        self.end_time = 0         # End timestamp
        self.total_distance = 0.  # Total distance traveled
        self.previous_x = 0       # Previous x position
        self.previous_y = 0       # Previous y position
        self.odom_start = True    # Flag for odometry start
        self.bump_flag = False    # Collision flag

        # Initialize ROS node
        rospy.init_node('move_base_sequence')
        points_seq = rospy.get_param('~p_seq')            # Load goal points from launch file
        yaweulerangles_seq = rospy.get_param('~yea_seq')  # Load yaw angles

        # Subscribers
        self.bumper_sub = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bumper_callback)  # Collision detection
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)  # Odometry tracking

        # Prepare goal sequence
        self.pose_seq = list()  # List of goal poses
        self.goal_cnt = 0       # Current goal index
        quat_seq = list()       # List of quaternions
        points = [points_seq[i:i+GOAL_NUM] for i in range(0, len(points_seq), GOAL_NUM)]  # Split points into [x, y, z]
        rospy.loginfo(str(points))
        for i in range(len(points)):
            quat_seq.append(Quaternion(*quaternion_from_euler(0, 0, 0, axes='sxyz')))  # Zero yaw quaternion
        for point in points:
            self.pose_seq.append(Pose(Point(*point), quat_seq[n-3]))  # Create pose with point and quaternion
            n += 1

        # Set up action client for move_base
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        wait = self.client.wait_for_server()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
            return
        rospy.loginfo("Connected to move base server")
        self.movebase_client()  # Start sending goals

    def bumper_callback(self, bumper_msg):
        """Track collisions from bumper events."""
        if bumper_msg.state == BumperEvent.PRESSED:
            self.bump_flag = True  # Set collision flag
        rospy.loginfo("Bumper Event:" + str(bumper_msg.bumper))

    def odom_callback(self, odom_msg):
        """Calculate total distance traveled from odometry."""
        if self.odom_start:
            self.previous_x = odom_msg.pose.pose.position.x  # Initial x
            self.previous_y = odom_msg.pose.pose.position.y  # Initial y
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        d_increment = hypot((x - self.previous_x), (y - self.previous_y))  # Distance increment
        self.total_distance += d_increment  # Update total distance
        self.previous_x = x
        self.previous_y = y
        self.odom_start = False  # Reset flag after first update

    def active_cb(self):
        """Log when a goal is being processed."""
        rospy.loginfo(f"Goal pose {self.goal_cnt+1} is now being processed...")

    def feedback_cb(self, feedback):
        """Log feedback for current goal."""
        rospy.loginfo(f"Feedback for goal pose {self.goal_cnt+1} received")

    def done_cb(self, status, result):
        """Handle goal completion or failure."""
        self.goal_cnt += 1
        if status == 2:  # Goal canceled after starting
            rospy.loginfo(f"Goal pose {self.goal_cnt} received a cancel request, completed!")
            self.log_metrics()
        elif status == 3:  # Goal reached
            rospy.loginfo(f"Goal pose {self.goal_cnt} reached")
            if not self.bump_flag:  # No collision
                self.success_num += 1
            self.bump_flag = False  # Reset collision flag
            self.end_time = rospy.get_time()  # Record end time
            self.total_time = self.end_time - self.start_time  # Calculate time
            self.log_metrics()
            self.send_next_goal()  # Send next goal if available
        elif status == 4:  # Goal aborted
            rospy.loginfo(f"Goal pose {self.goal_cnt} aborted")
            self.log_metrics()
            rospy.signal_shutdown(f"Goal pose {self.goal_cnt} aborted, shutting down!")
        elif status == 5:  # Goal rejected
            rospy.loginfo(f"Goal pose {self.goal_cnt} rejected")
            self.log_metrics()
            rospy.signal_shutdown(f"Goal pose {self.goal_cnt} rejected, shutting down!")
        elif status == 8:  # Goal canceled before starting
            rospy.loginfo(f"Goal pose {self.goal_cnt} canceled before executing!")
            self.log_metrics()

    def log_metrics(self):
        """Log navigation metrics."""
        rospy.loginfo(f"Success Number: {self.success_num} in total number {self.goal_cnt}")
        rospy.loginfo(f"Total Running Time: {self.total_time} secs")
        rospy.loginfo(f"Total Trajectory Length: {self.total_distance} m")

    def send_next_goal(self):
        """Send the next goal in the sequence."""
        if self.goal_cnt < len(self.pose_seq):
            next_goal = MoveBaseGoal()
            next_goal.target_pose.header.frame_id = "map"
            next_goal.target_pose.header.stamp = rospy.Time.now()
            next_goal.target_pose.pose = self.pose_seq[self.goal_cnt]
            rospy.loginfo(f"Sending goal pose {self.goal_cnt+1} to Action Server")
            rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
            self.client.send_goal(next_goal, self.done_cb, self.active_cb, self.feedback_cb)

    def movebase_client(self):
        """Start sending the goal sequence."""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        rospy.loginfo(f"Sending goal pose {self.goal_cnt+1} to Action Server")
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
        while rospy.get_time() < 12.237:  # Delay for simulation sync (Lobby world timing)
            self.start_time = rospy.get_time()
        self.start_time = rospy.get_time()
        rospy.loginfo(f"Start Time: {self.start_time} secs")
        self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)  # Send first goal
        rospy.spin()

if __name__ == '__main__':
    try:
        MoveBaseSeq()  # Start goal sequence publishing
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation finished.")