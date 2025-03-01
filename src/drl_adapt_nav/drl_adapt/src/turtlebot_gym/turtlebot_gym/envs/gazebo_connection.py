#!/usr/bin/python
"""
Controls Gazebo simulation for the project's Gym environment.
Provides methods to pause, unpause, reset, and adjust physics properties,
supporting training and simulation reset in drl_nav_env.py.
"""

import rospy
from std_srvs.srv import Empty  # Standard ROS service
from gazebo_msgs.msg import ODEPhysics  # Physics message
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest  # Physics service
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

class GazeboConnection:
    """Class to manage Gazebo simulation state."""
    def __init__(self, start_init_physics_parameters, reset_world_or_sim):
        """Initialize service proxies and physics settings."""
        # Service proxies for Gazebo control
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)  # Unpause simulation
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)      # Pause simulation
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)  # Reset full simulation
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)  # Reset world state
        
        # Physics service
        rospy.wait_for_service('/gazebo/set_physics_properties')  # Wait for physics service
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        
        # Initialization flags
        self.start_init_physics_parameters = start_init_physics_parameters  # Flag to init physics
        self.reset_world_or_sim = reset_world_or_sim  # Reset mode (WORLD or SIMULATION)
        
        self.init_values()  # Set initial physics and reset
        self.pauseSim()     # Start paused

    def pauseSim(self):
        """Pause Gazebo simulation."""
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()  # Call pause service
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

    def unpauseSim(self):
        """Unpause Gazebo simulation."""
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()  # Call unpause service
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

    def resetSim(self):
        """Reset simulation based on mode (WORLD or SIMULATION)."""
        if self.reset_world_or_sim == "SIMULATION":
            rospy.logwarn("SIMULATION RESET")
            self.resetSimulation()  # Full reset
        elif self.reset_world_or_sim == "WORLD":
            rospy.logwarn("WORLD RESET")
            self.resetWorld()  # Reset objects only
        else:
            rospy.logwarn("WRONG Reset Option:" + str(self.reset_world_or_sim))

    def resetSimulation(self):
        """Reset entire Gazebo simulation."""
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()  # Call reset simulation service
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        """Reset Gazebo world (object positions only)."""
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()  # Call reset world service
        except rospy.ServiceException as e:
            print("/gazebo/reset_world service call failed")

    def init_values(self):
        """Initialize simulation with physics settings."""
        self.resetSim()  # Perform initial reset
        if self.start_init_physics_parameters:
            rospy.logwarn("Initialising Simulation Physics Parameters")
            self.init_physics_parameters()  # Set physics properties
        else:
            rospy.logerr("NOT Initialising Simulation Physics Parameters")

    def init_physics_parameters(self):
        """Set Gazebo physics properties (gravity, timestep)."""
        self._time_step = Float64(0.001)  # Simulation timestep
        self._max_update_rate = Float64(1000.0)  # Max update rate
        self._gravity = Vector3(x=0.0, y=0.0, z=-9.81)  # Gravity vector
        self._ode_config = ODEPhysics()  # ODE physics settings
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20
        self.update_gravity_call()  # Apply physics settings

    def update_gravity_call(self):
        """Update Gazebo physics with configured settings."""
        self.pauseSim()  # Pause before update
        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data  # Set timestep
        set_physics_request.max_update_rate = self._max_update_rate.data  # Set update rate
        set_physics_request.gravity = self._gravity  # Set gravity
        set_physics_request.ode_config = self._ode_config  # Set ODE config
        result = self.set_physics(set_physics_request)  # Apply settings
        rospy.logdebug("Gravity Update Result==" + str(result.success))
        self.unpauseSim()  # Resume simulation

    def change_gravity(self, x, y, z):
        """Adjust gravity vector (not used in DRL-VO)."""
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z
        self.update_gravity_call()  # Update with new gravity