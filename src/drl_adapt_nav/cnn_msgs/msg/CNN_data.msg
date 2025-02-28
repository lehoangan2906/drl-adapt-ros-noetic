# Define a custom ROS message type named CNN_data (like a blueprint), which serve as a container for the input data required by the other modules.

float32[] ped_pos_map 			# pedestrian kinematics map in robot's frame, size 12800
float32[] scan				# front-facing 180 degree laser scan data (720 points)
float32[] scan_all			# full 270 degree laser scan data (1080 points)
float32[] image_gray			# Grayscale image data from depth camera
float32[] depth				# Depth image data from depth camera
float32[] goal_cart			# Sub-goal position in cartesian coordinates (robot frame)
float32[] goal_final_polar		# Final goal in polar coordinates (robot frame)
float32[] vel				# Current robot velocity in robot frame
