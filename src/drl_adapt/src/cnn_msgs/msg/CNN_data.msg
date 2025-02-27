# purpose: Defines a structured format for passing preprocessed observation data from sensor nodes to the DRL policy nodes. This data is published on a ROS topic(by cnn_data_pub.py) and subscribed by nodes like drl_adapt_train.py

float32[] ped_pos_map     # pedestrian's position costmap in Cartesian coordinate, 2 channels (pedestrians' positions and velocities), covering 20m x 20m area.
float32[] scan            # 720 range lidar scan data over 270 degree FOV
float32[] scan_all        # 1080 range lidar scan data 
float32[] image_gray      # image data from the zed camera
float32[] depth           # depth image data from the zed camera
float32[] goal_cart       # current sub-goal point in the robot's cartesian frame ([x, y] relative to the robot)
float32[] goal_final_polar  # the final goal point in the robot's polar frame [distance, angle]
float32[] vel               # robot's current velocity in its local frame.

