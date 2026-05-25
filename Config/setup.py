from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pioneer_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Pioneer navigation package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This copies your launch files to the install directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # This copies your config files (like YAMLs) to the install directory
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    entry_points={
        'console_scripts': [
            'pioneer_controller = pioneer_nav.pioneer_controller:main',
            'astar_planner = pioneer_nav.astar_planner:main',
            'dwa_controller = pioneer_nav.dwa_controller:main',
            'ariaNode = pioneer_nav.ariaNode:main',
            'cone_detector = pioneer_nav.cone_detector:main',
            'waypoint_nav = pioneer_nav.waypoint_nav:main',
            'estop = pioneer_nav.estop:main',
            'ui_node = pioneer_nav.ui_node:main',
            'letter_detector = pioneer_nav.greek:main'
        ],
    },
)
