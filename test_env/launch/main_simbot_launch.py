import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration, IncludeLaunchDescription, SetEnvironmentVariable, OpaqueFunction, GroupAction
from launch.launch_description_sources import FrontendLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
from launch.frontend.parse_substitution import parse_substitution
import xacro

# ===========================


def launch_arguments():
    return [
        DeclareLaunchArgument("scenario", default_value="Exp_C"),
        DeclareLaunchArgument("configuration", default_value="config1"),
        DeclareLaunchArgument("simulation", default_value="sim1"),
        DeclareLaunchArgument("namespace", default_value="PioneerP3DX"),
        DeclareLaunchArgument("num_sensors", default_value="4",
                              description="Number of PID gas sensors to launch (1-4). "
                                          "Exp_C=4, 10x6_empty_room=3, MAPIRlab/10x6_maze=2"),
    ]
# ==========================


def launch_setup(context, *args, **kwargs):
    share_dir = get_package_share_directory("test_env")
    namespace = LaunchConfiguration("namespace").perform(context)
    scenario = LaunchConfiguration("scenario").perform(context)
    simulation = LaunchConfiguration("simulation").perform(context)
    configuration = LaunchConfiguration("configuration").perform(context)
    num_sensors = int(LaunchConfiguration("num_sensors").perform(context))

    # robot description for state_publisher
    robot_desc = xacro.process_file(
        os.path.join(share_dir, "navigation_config", "resources", "giraff.xacro"),
        mappings={"frame_ns": namespace},
    )
    robot_desc = robot_desc.toprettyxml(indent="  ")

    visualization_nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"use_sim_time": True, "robot_description": robot_desc}],
        ),
    ]

    robot_simulator = [
        Node(
            package="basic_sim",
            executable="basic_sim",
            prefix="xterm -hold -e",
            parameters=[
                {"deltaTime": 0.03},
                {"speed": 1.0},
                {"worldFile": os.path.join(share_dir, "scenarios", scenario, "environment_configurations", configuration, "BasicSimScene.yaml")}
            ],
        )
    ]

    gaden_player = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("test_env"),
                    "launch",
                    "gaden_player_launch.py",
                )
            ]
        ),
        launch_arguments={
            "use_rviz": "True",
            "scenario": scenario,
            "configuration": configuration,
            "simulation": simulation
        }.items(),
    )

    nav2_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("test_env"),
                    "navigation_config",
                    "nav2_launch.py",
                )
            ]
        ),
        launch_arguments={
            "namespace": LaunchConfiguration("namespace").perform(context),
            "scenario": LaunchConfiguration("scenario").perform(context),
        }.items(),
    )

    anemometer = [
        Node(
            package="simulated_anemometer",
            executable="simulated_anemometer",
            name="wind_sensor",
            parameters=[
                {"sensor_frame": parse_substitution("$(var namespace)_anemometer_frame")},
                {"fixed_frame": "map"},
                {"noise_std": 0.3},
                {"use_map_ref_system": False},
                {'use_sim_time': True},
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='anemometer_tf_pub',
            arguments=['0', '0', '0.5', '1.0', '0.0', '0', '0', parse_substitution('$(var namespace)_base_link'), parse_substitution('$(var namespace)_anemometer_frame')],
            parameters=[{'use_sim_time': True}]
        ),
    ]

    # GAS SENSORS configuration (gas1, gas2, gas3, gas4)
    # =================================================

    gas1 = [  # PID — ethanol only (sim1)
        Node(
            package="simulated_gas_sensor",
            executable="simulated_gas_sensor",
            name="gas1",
            parameters=[
                {"sensor_model": 30},
                {"target_gas": "ethanol"},
                {"sensor_frame": parse_substitution("$(var namespace)_gas1_frame")},
                {"fixed_frame": "map"},
                {"noise_std": 0.1},
                {'use_sim_time': True},
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gas1_tf_pub',
            arguments=['0.1', '0', '0.5', '1.0', '0.0', '0', '0', parse_substitution('$(var namespace)_base_link'), parse_substitution('$(var namespace)_gas1_frame')],
            parameters=[{'use_sim_time': True}]
        ),
    ]

    gas2 = [  # PID — methane only (sim2)
        Node(
            package="simulated_gas_sensor",
            executable="simulated_gas_sensor",
            name="gas2",
            parameters=[
                {"sensor_model": 30},
                {"target_gas": "methane"},
                {"sensor_frame": parse_substitution("$(var namespace)_gas2_frame")},
                {"fixed_frame": "map"},
                {"noise_std": 0.1},
                {'use_sim_time': True},
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gas2_tf_pub',
            arguments=['-0.1', '0', '0.5', '1.0', '0.0', '0', '0', parse_substitution('$(var namespace)_base_link'), parse_substitution('$(var namespace)_gas2_frame')],
            parameters=[{'use_sim_time': True}]
        ),
    ]

    gas3 = [  # PID — hydrogen only (sim3)
        Node(
            package="simulated_gas_sensor",
            executable="simulated_gas_sensor",
            name="gas3",
            parameters=[
                {"sensor_model": 30},
                {"target_gas": "hydrogen"},
                {"sensor_frame": parse_substitution("$(var namespace)_gas3_frame")},
                {"fixed_frame": "map"},
                {"noise_std": 0.1},
                {'use_sim_time': True},
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gas3_tf_pub',
            arguments=['0', '0.1', '0.5', '1.0', '0.0', '0', '0',
                       parse_substitution('$(var namespace)_base_link'),
                       parse_substitution('$(var namespace)_gas3_frame')],
            parameters=[{'use_sim_time': True}]
        ),
    ]

    gas4 = [  # PID — propanol only (sim4)
        Node(
            package="simulated_gas_sensor",
            executable="simulated_gas_sensor",
            name="gas4",
            parameters=[
                {"sensor_model": 30},
                {"target_gas": "propanol"},
                {"sensor_frame": parse_substitution("$(var namespace)_gas4_frame")},
                {"fixed_frame": "map"},
                {"noise_std": 0.1},
                {'use_sim_time': True},
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gas4_tf_pub',
            arguments=['0', '-0.1', '0.5', '1.0', '0.0', '0', '0', parse_substitution('$(var namespace)_base_link'), parse_substitution('$(var namespace)_gas4_frame')],
            parameters=[{'use_sim_time': True}]
        ),
    ]

    namespaced_actions = [PushRosNamespace(namespace)]
    namespaced_actions.extend(visualization_nodes)

    # Gas-type → sensor config mapping (sensor index 1-based)
    all_gas_sensors = [gas1, gas2, gas3, gas4]

    other_actions = [gaden_player]
    other_actions.extend(robot_simulator)
    other_actions.extend(anemometer)
    for sensor_nodes in all_gas_sensors[:num_sensors]:
        other_actions.extend(sensor_nodes)
    other_actions.append(nav2_nodes)
    
    # Critical TF dummy link to solve Nav2 BehaviorTree hardcoded frame_id bugs
    other_actions.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='dummy_base_link_tf',
        arguments=['0', '0', '0', '0', '0', '0', parse_substitution('$(var namespace)_base_link'), 'base_link'],
        parameters=[{'use_sim_time': True}]
    ))

    return [GroupAction(actions=namespaced_actions),
            GroupAction(actions=other_actions)
            ]


def generate_launch_description():

    launch_description = [
        # Set env var to print messages to stdout immediately
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
        SetLaunchConfiguration(
            name="pkg_dir",
            value=[get_package_share_directory("test_env")],
        ),
        SetLaunchConfiguration(
            name="nav_params_yaml",
            value=[PathJoinSubstitution(
                [LaunchConfiguration("pkg_dir"), "navigation_config", "nav2_params.yaml"]
            )],
        ),
    ]

    launch_description.extend(launch_arguments())
    launch_description.append(OpaqueFunction(function=launch_setup))

    return LaunchDescription(launch_description)
