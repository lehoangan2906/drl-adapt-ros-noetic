"""
Sets up the turtlebot_gym package as a Python module.
Installs the custom Gym environment (drl-nav-v0) for training the navigation policy
in a Gazebo simulation with Pedsim pedestrians.
"""

from setuptools import setup  

setup(
    name='turtlebot_gym',       
    version='0.0.1',            
    install_requires=['gym']    # Dependency: OpenAI Gym library
)