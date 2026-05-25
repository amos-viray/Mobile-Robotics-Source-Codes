import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    package_name = 'pioneer_nav'
    pkg_share    = get_package_share_directory(package_name)
    nav2_params  = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    joystick_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'joystick.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('slam_toolbox'),
                         'launch', 'online_async_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file':
                '/home/ws/ws/src/pioneer_nav/config/slam_params.yaml',
        }.items()
    )

    return LaunchDescription([

        # ── Gazebo ────────────────────────────────────────────────────
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', '/home/ws/worlds/basic_urdf.sdf'],
            output='screen'
        ),

        # ── ROS-Gazebo bridges ────────────────────────────────────────
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/depth_image/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            ],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        # ── Static TF: chassis → laser_frame ─────────────────────────
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0.1', '0', '0', '0',
                       'chassis', 'laser_frame'],
            parameters=[{'use_sim_time': True}]
        ),

        # ── Robot State Publisher ─────────────────────────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description':
                    open('/home/ws/robots/pioneer.urdf').read(),
                'use_sim_time': True
            }]
        ),

        # ── t=5s: E-Stop, UI, Letter Detector, Object Detector ───────
        TimerAction(period=5.0, actions=[
            Node(package=package_name, executable='estop',
                 name='estop', output='screen',
                 parameters=[{'use_sim_time': True}]),
            Node(package=package_name, executable='ui_node',
                 name='ui_node', output='screen',
                 parameters=[{'use_sim_time': True}]),
            Node(package=package_name, executable='letter_detector',
                 name='letter_detector', output='screen',
                 parameters=[{'use_sim_time': True}]),
            Node(package=package_name, executable='object_detector',
                 name='object_detector', output='screen',
                 parameters=[{'use_sim_time': True}]),
        ]),

        # ── t=8s: EKF ────────────────────────────────────────────────
        TimerAction(period=8.0, actions=[
            Node(package='robot_localization',
                 executable='ekf_node',
                 name='ekf_filter_node', output='screen',
                 parameters=[
                     '/home/ws/ws/src/pioneer_nav/config/ekf.yaml',
                     {'use_sim_time': True}
                 ]),
        ]),

        # ── t=10s: SLAM Toolbox ───────────────────────────────────────
        TimerAction(period=10.0, actions=[slam_toolbox_launch]),

        # ── t=20s: Nav2 stack + Roadmap Explorer ─────────────────────
        # roadmap_exploration_server is NOT in lifecycle_manager node_names
        # — mission_manager handles its lifecycle manually so it can be
        # started/stopped per phase without affecting Nav2.
        TimerAction(period=20.0, actions=[
            Node(package='nav2_controller',
                 executable='controller_server',
                 name='controller_server',
                 parameters=[nav2_params, {'use_sim_time': True}],
                 remappings=[('/cmd_vel', '/cmd_vel_in')]),
            Node(package='nav2_planner',
                 executable='planner_server',
                 name='planner_server',
                 parameters=[nav2_params, {'use_sim_time': True}]),
            Node(package='nav2_behaviors',
                 executable='behavior_server',
                 name='behavior_server',
                 parameters=[nav2_params, {'use_sim_time': True}],
                 remappings=[('/cmd_vel', '/cmd_vel_in')]),
            Node(package='nav2_bt_navigator',
                 executable='bt_navigator',
                 name='bt_navigator',
                 parameters=[nav2_params, {'use_sim_time': True}]),
            Node(package='nav2_waypoint_follower',
                 executable='waypoint_follower',
                 name='waypoint_follower',
                 parameters=[nav2_params, {'use_sim_time': True}],
                 remappings=[('/cmd_vel', '/cmd_vel_in')]),
            Node(package='nav2_lifecycle_manager',
                 executable='lifecycle_manager',
                 name='lifecycle_manager_navigation',
                 parameters=[{
                     'autostart': True,
                     'use_sim_time': True,
                     'node_names': [
                         'controller_server', 'planner_server',
                         'behavior_server', 'bt_navigator',
                         'waypoint_follower'
                     ]
                 }]),
            # Roadmap explorer started separately — mission_manager
            # manages its lifecycle
            Node(package='roadmap_explorer',
                 executable='roadmap_exploration_server',
                 name='roadmap_exploration_server',
                 output='screen',
                 parameters=[{'use_sim_time': True}]),
        ]),

        # ── t=25s: Mission Manager ────────────────────────────────────
        # Starts after Nav2 is up. Manages phase transitions.
        TimerAction(period=25.0, actions=[
            Node(package=package_name,
                 executable='mission_manager',
                 name='mission_manager',
                 output='screen',
                 parameters=[{'use_sim_time': True}]),
        ]),

        # ── RViz ─────────────────────────────────────────────────────
        Node(package='rviz2', executable='rviz2',
             parameters=[{'use_sim_time': True}]),

        joystick_launch
    ])
