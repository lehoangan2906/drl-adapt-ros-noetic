#!usr/bin/python

"""
Publishes pedestrian kinematic maps, lidar data, goals, and velocity as CNN_data messages for DRL-ADAPT policy.

- Create a ROS node (cnn_data) that aggregate:
    + Pedestrian kinematic data from Pedsim via (/track_ped)
    + Lidar scans via (/scan)
    + Sub-goals via (/cnn_goal)
    + Robot velocity via (/mobile_base/commands/velocity)

- Processes this data into the CNN_data message format and publishes it on /cnn_data at 20 Hz
"""

import rospy
import numpy as np
from cnn_msgs.msg import CNN_data
from sensor_msgs.msg import LaserScan
from scipy.interpolate import interp1d
from pedsim_msgs.msg import TrackedPerson, TrackedPersons
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped


# Constants
NUM_TP = 10             # Number of timestamps for lidar history
NUM_PEDS = 34+1         # Maximum number of pedestrians (plus the robot)
LIDAR_MIN_RANGE = 0.03  # minimum valid scan range for Hokuyo UST-30LX
LIDAR_MAX_RANGE = 30.0  # maximum valid scan range for lidar

class CNNData:
    def __init__(self):
        """ Initialize data structures and ROS subscribers/publishers. """
        
        # Data initialization
        self.ped_pos_map = []           # Pedestrian kinematic map
        self.scan = []                  # 180 degree front Lidar scan 
        self.scan_all = np.zeros(1080)  # Full 270 degree lidar scan 
        self.goal_cart = np.zeros(2)    # Current sub-goal
        self.vel = np.zeros(2)          # Robot velocity

        # Temporary data buffers to prevent race condition when mutiple function access and update the same variable in different rate
        self.ped_pos_map_tmp = np.zeros((2, 80, 80))    # Temp 80x80x2 pedestrian map
        self.scan_tmp = np.zeros(720)                   # Temp 720-point scan
        self.scan_all_tmp = np.zeros(1080)              # Temp full scan

        # ROS subscribers
        self.ped_sub = rospy.Subscriber("/track_ped", TrackedPersons, self.ped_callback)                # Pedestrian data
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)                        # Lidar data
        self.goal_sub = rospy.Subscriber("/cnn_goal", Point, self.goal_callback)                        # Sub-goal
        self.vel_sub = rospy.Subscriber("/mobile_base/commands/velocity", Twist, self.vel_callback)     # Velocity
        self.cnn_data_sub = rospy.Publisher("/cnn_data", CNN_data, queue_size=1, latch=False)           # CNN data publisher

        # Timer for periodic publishing
        self.rate = 20  
        self.ts_cnt = 0     # Timestamp counter
        self.timer = rospy.Timer(rospy.Duration(1./self.rate), self.timer_callback)


    def ped_callback(self, trackPed_msg):
        """Process pedestrian data into 80x80x2 velocity map."""

        self.ped_pos_map_tmp = np.zeros((2, 80, 80))       # Reset temp map
        
        if len(trackPed_msg.tracks) != 0:   # If pedestrians detected

            # Get each pedestrian's position and velocity 
            for ped in trackPed_msg.tracks:
                x = ped.pose.pose.position.x    # Pedestrian x position
                y = ped.pose.pose.position.y    # Pedestrian y position
                vx = ped.twist.twist.linear.x   # x velocity
                vy = ped.twist.twist.linear.y   # y velocity

                # Convert a pedestrian's position (x, y) in the robot's local frame to 20m x 20m grid indices (r, c) (0.25m cell)
                if x >= 0 and x <= 20 and np.abs(y) <= 10:
                    c = int(np.floor(-(y-10)/0.25))     # column index (y-axis)
                    r = int(np.floor(x/0.25))           # row index (x-axis)

                    # Adjust indices to stay within the 80x80 grid
                    c = min(c, 79)
                    r = min(x, 79)

                    # Store velocity components in grid
                    self.ped_pos_map_tmp[0, r, c] = vx  # Channel 0: x-velocity
                    self.ped_pos_map_tmp[1, r, c] = vy  # Channel 1: y-velocity


    def scan_callback(self, laserScan_msg):
        """Capture and process lidar scan data."""

        self.scan_tmp = np.zeros(720)           # Reset temp 720-point scan
        self.scan_all_tmp = np.zeros(1080)      # Reset full scan


        # Extract lidar ranges data from the ROS topic message as a numpy array
        scan_data = np.array(laserScan_msg.ranges, dtype=np.float32)        


        scan_data[np.isnan(scan_data)] = 0.     # Replace NaN with 0
        scan_data[np.isinf(scan_data)] = 0.     # Replace inf with 0


        # Create a mask to identify which points fall within a valid range
        # e.g., valid_mask ==[True, True, False, ...]
        valid_mask = (scan_data > LIDAR_MIN_RANGE) & (scan_data < LIDAR_MAX_RANGE)


        # Check if there are any invalid points (0.0 or out of range values)
        if np.any(~valid_mask):
            # Get the indices where the data is valid (within the lidar's range)
            valid_indices = np.where(valid_mask)[0]


            # Extract only the valid lidar scan data points
            valid_scan_data = scan_data[valid_mask]


            # Interpolate the invalid (missing) points using linear interpolation
            # `interp1d` creates a function that will interpolate values based on the valid points
            # valid_indices: indices of valid points
            # valid_scan_data: corresponding valid distance values at those indices
            # kind="linear": linear interpolation method
            # bounds_error=False: allows interpolation for out-of-bound indices without raising an error
            # fill_value="extrapolate": fills missing values by extrapolating where data points are missing
            interpolation = interp1d(
                valid_indices,
                valid_scan_data,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )


            # Use the interpolation function to fill in missing points for all indices
            # np.arange(len(scan_data)) creates an array with all indices from 0 to len(scan_data) - 1
            interpolated_scan = interpolation(np.arange(len(scan_data)))


            # Clip the interpolated scan data to ensure no values exceed the lidar's valid range
            # Anything below 0 or above LIDAR_MAX_RANGE will be limited within that range
            scan_data = np.clip(interpolated_scan, 0, LIDAR_MAX_RANGE)


        self.scan_tmp = scan_data[180:900]  # Extract 720 points (central 180 degree FOV)
        self.scan_all_tmp = scan_data       # Store the full interpolated scan data (including the front, back, and sides)


    def goal_callback(self, goal_msg):
        """Store sub-goal position from pure pursuit."""

        self.goal_cart[0] = goal_msg.x      # x-coordinate
        self.goal_cart[1] = goal_msg.y      # y-coordinate

    
    def vel_callback(self, vel_msg):
        """Store current robot velocity"""

        self.vel[0] = vel_msg.linear.x      # Linear velocity (vx)
        self.vel[1] = vel_msg.angular.z     # Angular velocity (wz)


    def timer_callback(self, event):
        """Periodically publish CNN data every 1/20s."""

        self.ped_pos_map = self.ped_pos_map_tmp     # Update pedestrian map
        self.scan.append(self.scan_tmp.tolist())    # Add latest scan to history
        self.scan_all = self.scan_all_tmp           # Update full scan

        self.ts_cnt += 1    # Increment timestamp counter

        # After 10 timesteps (0.5s history)
        if self.ts_cnt == NUM_TP:   
            cnn_data = CNN_data()       # Create CNN_data message

            cnn_data.ped_pos_map = [float(val) for sublist in self.ped_pos_map for subb in sublist for val in subb]     # Flatten 80x80x2 grid to 12800 array for ped_pos_map
            cnn_data.scan = [float(val) for sublist in self.scan for val in sublist]  # Flatten scan history
            cnn_data.scan_all = self.scan_all       # Full scan
            cnn_data.depth = []                     # Unused (no camera)
            cnn_data.image_gray = []                # Unused (no camera)
            cnn_data.goal_cart = self.goal_cart     # Sub-goal
            cnn_data.goal_final_polar = []          # Unused
            cnn_data.vel = self.vel                 # Robot velocity

            self.cnn_data_pub.publish(cnn_data)     # Publish to /cnn_data


if __name__ == '__main__':
    try:
        rospy.init_node('cnn_data')     # Initialize ROS node
        CNNData()                       # Create class instance
        rospy.spin()                    # Keep node running
    except rospy.ROSInterruptException:
        pass