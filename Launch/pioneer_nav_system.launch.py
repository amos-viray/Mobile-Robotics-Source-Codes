import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    package_name = 'pioneer_nav'
    pkg_share    = get_package_share_directory(package_name)
    nav2_params  = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    urdf_path    = '/home/ws/robots/pioneer.urdf'

    joystick_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'joystick.launch.py')
        ),
        launch_arguments={'use_sim_time': 'false'}.items()
    )

    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    slam_launch_path = os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch_path),
        launch_arguments={
            'use_sim_time': 'false',
            'slam_params_file': '/home/ws/ws/src/pioneer_nav/config/slam_params.yaml',
        }.items()
    )

    return LaunchDescription([

        # --- Joystick
        joystick_launch,

        # --- Hardware
        Node(
            package='ariaNode',
            executable='ariaNode',
            name='aria_node',
            arguments=['-rp', '/dev/ttyUSB0'],
            output='screen'
        ),

        # --- Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': open(urdf_path).read(),
                'use_sim_time': False
            }]
        ),

        # --- OAK-D Camera
        Node(
            package='depthai_ros_driver_v3',
            executable='driver_node',
            name='oak',
            parameters=[{
                'use_sim_time': False,
                'i_nn_type': 'none',
                'i_enable_imu': False,
            }],
            remappings=[
                ('~/rgb/image_raw',    '/oak/rgb/image_raw'),
                ('~/stereo/image_raw', '/oak/stereo/image_raw'),
            ]
        ),

        # --- E-Stop + UI + Detectors (t=3s, after hardware is up)
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package=package_name,
                    executable='estop',
                    name='estop',
                    output='screen',
                    parameters=[{'use_sim_time': False}]
                ),
                Node(
                    package=package_name,
                    executable='ui_node',
                    name='ui_node',
                    output='screen',
                    parameters=[{'use_sim_time': False}]
                ),
                Node(
                    package=package_name,
                    executable='letter_detector',
                    name='letter_detector',
                    output='screen',
                    parameters=[{'use_sim_time': False}]
                ),
                Node(
                    package=package_name,
                    executable='cone_detector',
                    name='cone_detector',
                    output='screen',
                    parameters=[{'use_sim_time': False}]
                ),
            ]
        ),

        # --- EKF (t=5s)
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='robot_localization',
                    executable='ekf_node',
                    name='ekf_filter_node',
                    output='screen',
                    parameters=[
                        '/home/ws/ws/src/pioneer_nav/config/ekf.yaml',
                        {'use_sim_time': False}
                    ],
                ),
            ]
        ),

        # --- SLAM Toolbox (t=7s, after EKF establishes odom->base_link)
        TimerAction(
            period=7.0,
            actions=[
                slam_toolbox_launch,
            ]
        ),

        # --- Nav2 + waypoint_follower + Roadmap Explorer (t=15s)
        TimerAction(
            period=15.0,
            actions=[
                Node(package='nav2_controller', executable='controller_server', name='controller_server',
                     parameters=[nav2_params],
                     remappings=[('/cmd_vel', '/cmd_vel_in')]),
                Node(package='nav2_planner', executable='planner_server', name='planner_server',
                     parameters=[nav2_params]),
                Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server',
                     parameters=[nav2_params],
                     remappings=[('/cmd_vel', '/cmd_vel_in')]),
                Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
                     parameters=[nav2_params]),
                Node(package='nav2_waypoint_follower', executable='waypoint_follower',
                     name='waypoint_follower', parameters=[nav2_params],
                     remappings=[('/cmd_vel', '/cmd_vel_in')]),
                Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                     name='lifecycle_manager_navigation',
                     parameters=[{'autostart': True, 'use_sim_time': False,
                                  'node_names': ['controller_server', 'planner_server',
                                                 'behavior_server', 'bt_navigator',
                                                 'waypoint_follower']}]),
                Node(
                    package='roadmap_explorer',
                    executable='roadmap_exploration_server',
                    name='roadmap_exploration_server',
                    output='screen',
                    parameters=[{'use_sim_time': False}]
                ),
            ]
        ),
    ])
