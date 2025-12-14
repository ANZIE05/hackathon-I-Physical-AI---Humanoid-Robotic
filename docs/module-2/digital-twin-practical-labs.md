---
sidebar_position: 6
---

# Digital Twin Practical Labs

## Overview
This section provides hands-on practical labs that reinforce the concepts of digital twin simulation, Gazebo fundamentals, physics simulation, sensor simulation, and Unity visualization. Each lab builds upon the previous knowledge and provides real-world application of simulation principles in Physical AI and Humanoid Robotics contexts.

## Lab 1: Basic Robot Simulation in Gazebo

### Objective
Create a simple differential drive robot model in Gazebo with basic sensors and ROS 2 integration to understand the fundamentals of robotic simulation.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Gazebo Garden installed
- Basic understanding of URDF/SDF
- Understanding of ROS 2 communication patterns

### Lab Setup
```bash
# Create workspace
mkdir -p ~/simulation_labs/ws_basic_robot/src
cd ~/simulation_labs/ws_basic_robot

# Create simulation package
ros2 pkg create --build-type ament_python basic_robot_simulation --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs tf2_ros gazebo_ros_pkgs
```

### Implementation Steps

#### Step 1: Create Robot URDF Model
Create `basic_robot_simulation/urdf/differential_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="differential_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="0.15" />
  <xacro:property name="wheel_pos_x" value="0.2" />
  <xacro:property name="wheel_pos_y" value="0.15" />

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="${wheel_pos_x} ${wheel_pos_y} 0" rpy="${-M_PI/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="${wheel_pos_x} ${-wheel_pos_y} 0" rpy="${-M_PI/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="${base_length/2 - 0.05} 0 ${base_height/2}" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

</robot>
```

#### Step 2: Create Gazebo World
Create `basic_robot_simulation/worlds/simple_room.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
        </constraints>
      </ode>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.2 0.3 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Room walls -->
    <model name="wall_1">
      <pose>-3 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.1 6 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.1 6 2</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>10</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>10</iyy>
            <iyz>0</iyz>
            <izz>10</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_2">
      <pose>3 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.1 6 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.1 6 2</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>10</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>10</iyy>
            <iyz>0</iyz>
            <izz>10</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_3">
      <pose>0 -3 1 0 0 1.5707</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.1 6 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.1 6 2</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>10</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>10</iyy>
            <iyz>0</iyz>
            <izz>10</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_4">
      <pose>0 3 1 0 0 1.5707</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.1 6 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.1 6 2</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>10</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>10</iyy>
            <iyz>0</iyz>
            <izz>10</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Obstacles -->
    <model name="obstacle_1">
      <pose>1 1 0.2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder><radius>0.3</radius><length>0.4</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.3</radius><length>0.4</length></cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>0.5</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.5</iyy>
            <iyz>0</iyz>
            <izz>0.8</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Robot spawn point -->
    <include>
      <uri>model://differential_robot</uri>
      <pose>0 0 0.2 0 0 0</pose>
    </include>
  </world>
</sdf>
```

#### Step 3: Create Robot Control Node
Create `basic_robot_simulation/basic_robot_simulation/robot_controller.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import math
import numpy as np

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_publisher = self.create_publisher(
            rclpy.qos.QoSProfile(depth=10), 'robot_status', 10)

        # Subscribers
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.odom_subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.image_subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.latest_scan = None
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.cv_bridge = CvBridge()

        # Control parameters
        self.obstacle_threshold = 0.5  # meters
        self.target_distance = 2.0     # meters
        self.current_mode = 'exploring'  # exploring, avoiding, following

        self.get_logger().info('Robot Controller Node Started')

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.latest_scan = msg

        # Check for obstacles in front of robot
        if len(msg.ranges) > 0:
            # Get front-facing ranges (±30 degrees)
            front_start = len(msg.ranges) // 2 - 15
            front_end = len(msg.ranges) // 2 + 15

            if front_start >= 0 and front_end < len(msg.ranges):
                front_ranges = msg.ranges[front_start:front_end]
                valid_ranges = [r for r in front_ranges if 0 < r < float('inf')]

                if valid_ranges:
                    min_range = min(valid_ranges)
                    if min_range < self.obstacle_threshold:
                        self.current_mode = 'avoiding'
                    else:
                        self.current_mode = 'exploring'

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.robot_pose['x'] = msg.pose.pose.position.x
        self.robot_pose['y'] = msg.pose.pose.position.y

        # Convert quaternion to euler
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        self.robot_pose['theta'] = theta

    def image_callback(self, msg):
        """Process camera image data"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # In a real application, you would process the image here
            # For this lab, we just log that we received an image
            self.get_logger().debug(f'Received image: {cv_image.shape}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def control_loop(self):
        """Main control logic"""
        cmd = Twist()

        if self.current_mode == 'avoiding':
            # Emergency obstacle avoidance
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn right
        elif self.current_mode == 'exploring':
            # Simple exploration pattern
            cmd.linear.x = 0.3  # Move forward slowly
            cmd.angular.z = 0.0

        # Publish command
        self.cmd_vel_publisher.publish(cmd)

        # Publish status
        status_msg = rclpy.qos.QoSProfile(depth=10)
        status_msg.data = f'Mode: {self.current_mode}, Pos: ({self.robot_pose["x"]:.2f}, {self.robot_pose["y"]:.2f})'
        self.status_publisher.publish(status_msg)

        self.get_logger().info(f'Control: {self.current_mode}, Vel: ({cmd.linear.x:.2f}, {cmd.angular.z:.2f})',
                              throttle_duration_sec=2)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 4: Create Launch File
Create `basic_robot_simulation/launch/basic_simulation.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='simple_room.sdf',
        description='Choose one of the world files from `/basic_robot_simulation/worlds`'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('basic_robot_simulation'),
                'worlds',
                LaunchConfiguration('world')
            ]),
            'verbose': 'false'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': open(
                PathJoinSubstitution([
                    FindPackageShare('basic_robot_simulation'),
                    'urdf',
                    'differential_robot.urdf'
                ])
            ).read()
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': True}]
    )

    # Gazebo ROS diff drive plugin configuration
    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_cont'],
        parameters=[{'use_sim_time': True}]
    )

    # Robot controller node
    robot_controller = Node(
        package='basic_robot_simulation',
        executable='robot_controller',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        robot_controller
    ])
```

#### Step 5: Update Package Configuration
Update `basic_robot_simulation/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>basic_robot_simulation</name>
  <version>0.0.0</version>
  <description>Basic robot simulation package for lab exercises</description>
  <maintainer email="student@university.edu">Student</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>cv_bridge</depend>
  <depend>gazebo_ros_pkgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Update `basic_robot_simulation/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'basic_robot_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/differential_robot.urdf']),
        ('share/' + package_name + '/worlds', ['worlds/simple_room.sdf']),
        ('share/' + package_name + '/launch', ['launch/basic_simulation.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@university.edu',
    description='Basic robot simulation package for lab exercises',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = basic_robot_simulation.robot_controller:main',
        ],
    },
)
```

#### Step 6: Build and Test
```bash
# Build the package
cd ~/simulation_labs/ws_basic_robot
colcon build --packages-select basic_robot_simulation

# Source the workspace
source install/setup.bash

# Launch the simulation
ros2 launch basic_robot_simulation basic_simulation.launch.py

# In another terminal, send commands to test
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'
```

### Lab Exercises
1. Modify the robot's behavior to follow walls instead of random exploration
2. Add more complex obstacles to the environment
3. Implement a simple navigation algorithm that moves toward a target
4. Add more sensors to the robot (IMU, additional cameras)

### Expected Results
- Robot model successfully loads in Gazebo
- Robot responds to velocity commands
- Sensor data is published and received correctly
- Basic control logic functions as expected

## Lab 2: Advanced Sensor Simulation

### Objective
Create a comprehensive sensor simulation system with multiple sensor types (LiDAR, camera, IMU) and implement sensor fusion for enhanced perception.

### Implementation Steps

#### Step 1: Enhanced Robot Model with Multiple Sensors
Create `basic_robot_simulation/urdf/sensor_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="sensor_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="0.15" />
  <xacro:property name="wheel_pos_x" value="0.2" />
  <xacro:property name="wheel_pos_y" value="0.15" />

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="${wheel_pos_x} ${wheel_pos_y} 0" rpy="${-M_PI/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="${wheel_pos_x} ${-wheel_pos_y} 0" rpy="${-M_PI/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="${base_length/2 - 0.05} 0 ${base_height/2}" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- LiDAR -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 ${base_height + 0.05}" rpy="0 0 0"/>
  </joint>

  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0002"/>
    </inertial>
  </link>

  <!-- IMU -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>

  <!-- Gazebo plugins for sensors -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>30</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <topic_name>camera/image_raw</topic_name>
        <hack_baseline>0.07</hack_baseline>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="lidar_link">
    <sensor name="3d_lidar" type="ray">
      <always_on>1</always_on>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
        <topic_name>scan</topic_name>
        <frame_name>lidar_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>1</always_on>
      <update_rate>100</update_rate>
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <topic_name>imu/data</topic_name>
        <frame_name>imu_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
    </plugin>
  </gazebo>

</robot>
```

#### Step 2: Sensor Fusion Node
Create `basic_robot_simulation/basic_robot_simulation/sensor_fusion.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import math

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Publishers
        self.fused_data_publisher = self.create_publisher(
            Float32MultiArray, 'fused_sensor_data', 10)
        self.environment_map_publisher = self.create_publisher(
            Float32MultiArray, 'environment_map', 10)

        # Subscribers
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.odom_subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        # Timer for fusion processing
        self.fusion_timer = self.create_timer(0.1, self.fusion_loop)

        # Sensor data storage
        self.latest_scan = None
        self.latest_imu = None
        self.latest_odom = None
        self.cv_bridge = CvBridge()

        # Environment mapping
        self.map_resolution = 0.1  # meters per cell
        self.map_size = 20  # meters (10m in each direction)
        self.map_cells = int(self.map_size / self.map_resolution)
        self.environment_map = np.zeros((self.map_cells, self.map_cells))

        self.get_logger().info('Sensor Fusion Node Started')

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        self.latest_scan = msg

        # Update environment map based on scan
        if self.latest_odom:
            self.update_map_from_scan(msg)

    def imu_callback(self, msg):
        """Process IMU data"""
        self.latest_imu = msg

    def odom_callback(self, msg):
        """Process odometry data"""
        self.latest_odom = msg

    def update_map_from_scan(self, scan_msg):
        """Update occupancy grid map from LiDAR scan"""
        if not self.latest_odom:
            return

        robot_x = self.latest_odom.pose.pose.position.x
        robot_y = self.latest_odom.pose.pose.position.y

        # Convert quaternion to yaw
        quat = self.latest_odom.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Process each ray in the scan
        for i, range_val in enumerate(scan_msg.ranges):
            if 0 < range_val < scan_msg.range_max:
                # Calculate angle of this ray
                angle = scan_msg.angle_min + i * scan_msg.angle_increment + robot_yaw

                # Calculate world coordinates of detected point
                world_x = robot_x + range_val * math.cos(angle)
                world_y = robot_y + range_val * math.sin(angle)

                # Convert to map coordinates
                map_x = int((world_x + self.map_size/2) / self.map_resolution)
                map_y = int((world_y + self.map_size/2) / self.map_resolution)

                # Update map if coordinates are valid
                if 0 <= map_x < self.map_cells and 0 <= map_y < self.map_cells:
                    self.environment_map[map_y, map_x] = 1.0  # Mark as occupied

    def fusion_loop(self):
        """Main sensor fusion processing"""
        if not all([self.latest_scan, self.latest_imu, self.latest_odom]):
            return

        # Create fused data array
        fused_data = Float32MultiArray()

        # Pack sensor data into array:
        # [scan_min_range, scan_avg_range, imu_angular_velocity_z, robot_linear_x, robot_angular_z]
        scan_ranges = [r for r in self.latest_scan.ranges if 0 < r < float('inf')]
        if scan_ranges:
            scan_min = min(scan_ranges)
            scan_avg = sum(scan_ranges) / len(scan_ranges)
        else:
            scan_min = 0.0
            scan_avg = 0.0

        imu_angular_z = self.latest_imu.angular_velocity.z
        robot_linear_x = self.latest_odom.twist.twist.linear.x
        robot_angular_z = self.latest_odom.twist.twist.angular.z

        fused_data.data = [scan_min, scan_avg, imu_angular_z, robot_linear_x, robot_angular_z]

        # Publish fused data
        self.fused_data_publisher.publish(fused_data)

        # Publish environment map
        map_msg = Float32MultiArray()
        map_msg.data = self.environment_map.flatten().tolist()
        self.environment_map_publisher.publish(map_msg)

        # Log fusion results
        self.get_logger().info(f'Fusion: Min Range={scan_min:.2f}, '
                              f'IMU Angular Z={imu_angular_z:.2f}, '
                              f'Robot Vel=({robot_linear_x:.2f}, {robot_angular_z:.2f})',
                              throttle_duration_sec=2)

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()

    try:
        rclpy.spin(sensor_fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Enhanced Launch File
Update the launch file to include sensor fusion:

```python
# Enhanced launch file with sensor fusion
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='simple_room.sdf',
        description='Choose one of the world files from `/basic_robot_simulation/worlds`'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('basic_robot_simulation'),
                'worlds',
                LaunchConfiguration('world')
            ]),
            'verbose': 'false'
        }.items()
    )

    # Robot state publisher with enhanced model
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': open(
                PathJoinSubstitution([
                    FindPackageShare('basic_robot_simulation'),
                    'urdf',
                    'sensor_robot.urdf'
                ])
            ).read()
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': True}]
    )

    # Robot controller node
    robot_controller = Node(
        package='basic_robot_simulation',
        executable='robot_controller',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Sensor fusion node
    sensor_fusion = Node(
        package='basic_robot_simulation',
        executable='sensor_fusion',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        robot_controller,
        sensor_fusion
    ])
```

### Lab Exercises
1. Implement a more sophisticated sensor fusion algorithm that combines multiple sensor inputs
2. Create an occupancy grid map from LiDAR data
3. Implement obstacle detection and avoidance using fused sensor data
4. Add visualizations for the fused sensor data

### Expected Results
- Multiple sensors working simultaneously
- Sensor fusion node processing and combining data
- Environment mapping from sensor data
- Enhanced robot behavior based on fused information

## Lab 3: Unity Visualization Integration

### Objective
Create a Unity visualization system that connects to the ROS 2 simulation and provides enhanced 3D visualization of the robot and environment.

### Implementation Steps

#### Step 1: Unity Setup for Robotics
Create a Unity scene with the robot model and environment visualization:

```csharp
// UnityRobotVisualizer.cs - Main visualization script
using UnityEngine;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;

public class UnityRobotVisualizer : MonoBehaviour
{
    [Header("Robot Configuration")]
    public GameObject robotModel;
    public Transform robotTransform;

    [Header("Sensor Visualization")]
    public GameObject lidarPointCloud;
    public GameObject cameraFeedDisplay;
    public GameObject occupancyGrid;

    [Header("ROS Connection")]
    public string rosBridgeUrl = "ws://localhost:9090";
    public string robotPoseTopic = "/odom";
    public string lidarTopic = "/scan";
    public string cameraTopic = "/camera/image_raw";

    private RosConnection rosConnection;
    private UnitySensorSimulation sensorSim;

    void Start()
    {
        ConnectToROS();
        SetupSubscribers();
        InitializeVisualization();
    }

    void ConnectToROS()
    {
        rosConnection = GetComponent<RosConnection>();
        rosConnection.rosBridgeServerUrl = rosBridgeUrl;
    }

    void SetupSubscribers()
    {
        rosConnection.Subscribe<OdometryMsg>(robotPoseTopic, UpdateRobotPose);
        rosConnection.Subscribe<LaserScanMsg>(lidarTopic, ProcessLidarData);
        rosConnection.Subscribe<ImageMsg>(cameraTopic, ProcessCameraData);
    }

    void InitializeVisualization()
    {
        if (robotModel == null)
        {
            robotModel = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            robotModel.name = "RobotModel";
        }

        robotTransform = robotModel.transform;
    }

    void UpdateRobotPose(OdometryMsg odom)
    {
        // Update robot position and orientation
        Vector3 position = new Vector3(
            (float)odom.pose.pose.position.x,
            (float)odom.pose.pose.position.z, // Map Z to Unity Y for up direction
            (float)odom.pose.pose.position.y  // Map Y to Unity Z
        );

        Quaternion rotation = new Quaternion(
            (float)odom.pose.pose.orientation.x,
            (float)odom.pose.pose.orientation.z,
            (float)odom.pose.pose.orientation.y,
            (float)odom.pose.pose.orientation.w
        );

        robotTransform.position = position;
        robotTransform.rotation = rotation;
    }

    void ProcessLidarData(LaserScanMsg scan)
    {
        // Convert LiDAR data to Unity visualization
        if (lidarPointCloud != null)
        {
            UpdateLidarVisualization(scan);
        }
    }

    void ProcessCameraData(ImageMsg image)
    {
        // Process camera image for display
        if (cameraFeedDisplay != null)
        {
            UpdateCameraVisualization(image);
        }
    }

    void UpdateLidarVisualization(LaserScanMsg scan)
    {
        // This would typically update a point cloud visualization
        // For this example, we'll just log the data
        Debug.Log($"Received LiDAR scan with {scan.ranges.Length} points");
    }

    void UpdateCameraVisualization(ImageMsg image)
    {
        // This would update a texture with camera data
        Debug.Log($"Received camera image: {image.width}x{image.height}");
    }

    void Update()
    {
        // Update visualization elements
        UpdateEnvironment();
    }

    void UpdateEnvironment()
    {
        // Update environment visualization based on current state
    }
}
```

#### Step 2: Environment Visualization
```csharp
// EnvironmentVisualizer.cs - Visualize the simulated environment
using UnityEngine;

public class EnvironmentVisualizer : MonoBehaviour
{
    [Header("Environment Configuration")]
    public float mapSize = 20f;
    public int resolution = 100;
    public Material occupiedMaterial;
    public Material freeSpaceMaterial;
    public Material unknownMaterial;

    private GameObject[,] gridCells;
    private float cellSize;

    void Start()
    {
        InitializeGrid();
        GenerateEnvironment();
    }

    void InitializeGrid()
    {
        cellSize = mapSize / resolution;
        gridCells = new GameObject[resolution, resolution];

        // Create grid visualization
        for (int x = 0; x < resolution; x++)
        {
            for (int z = 0; z < resolution; z++)
            {
                Vector3 position = new Vector3(
                    (x - resolution / 2) * cellSize,
                    0,
                    (z - resolution / 2) * cellSize
                );

                GameObject cell = GameObject.CreatePrimitive(PrimitiveType.Quad);
                cell.transform.position = position;
                cell.transform.localScale = new Vector3(cellSize * 0.9f, 0.1f, cellSize * 0.9f);
                cell.transform.rotation = Quaternion.Euler(90, 0, 0); // Face upward

                cell.GetComponent<Renderer>().material = unknownMaterial;
                cell.name = $"GridCell_{x}_{z}";

                gridCells[x, z] = cell;
            }
        }
    }

    public void UpdateGridCell(int x, int z, float occupancy)
    {
        if (x >= 0 && x < resolution && z >= 0 && z < resolution)
        {
            GameObject cell = gridCells[x, z];
            if (cell != null)
            {
                if (occupancy > 0.7f) // Occupied
                {
                    cell.GetComponent<Renderer>().material = occupiedMaterial;
                }
                else if (occupancy < 0.3f) // Free space
                {
                    cell.GetComponent<Renderer>().material = freeSpaceMaterial;
                }
                else // Unknown
                {
                    cell.GetComponent<Renderer>().material = unknownMaterial;
                }
            }
        }
    }

    void GenerateEnvironment()
    {
        // Generate static environment elements
        CreateWalls();
        CreateObstacles();
    }

    void CreateWalls()
    {
        // Create boundary walls
        float wallHeight = 2f;
        float wallThickness = 0.1f;
        float halfSize = mapSize / 2;

        // Create 4 walls around the boundary
        CreateWall(new Vector3(0, wallHeight/2, halfSize), new Vector3(mapSize, wallThickness, wallHeight), Color.gray);
        CreateWall(new Vector3(0, wallHeight/2, -halfSize), new Vector3(mapSize, wallThickness, wallHeight), Color.gray);
        CreateWall(new Vector3(halfSize, wallHeight/2, 0), new Vector3(wallThickness, wallThickness, mapSize), Color.gray);
        CreateWall(new Vector3(-halfSize, wallHeight/2, 0), new Vector3(wallThickness, wallThickness, mapSize), Color.gray);
    }

    void CreateObstacles()
    {
        // Create some static obstacles in the environment
        CreateObstacle(new Vector3(2, 0.5f, 2), new Vector3(0.5f, 1f, 0.5f), Color.red);
        CreateObstacle(new Vector3(-2, 0.5f, -2), new Vector3(1f, 1f, 0.3f), Color.blue);
    }

    GameObject CreateWall(Vector3 position, Vector3 size, Color color)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.transform.position = position;
        wall.transform.localScale = size;

        Renderer renderer = wall.GetComponent<Renderer>();
        Material material = new Material(Shader.Find("Standard"));
        material.color = color;
        renderer.material = material;

        wall.AddComponent<Rigidbody>();
        wall.GetComponent<Rigidbody>().isKinematic = true;

        return wall;
    }

    GameObject CreateObstacle(Vector3 position, Vector3 size, Color color)
    {
        GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
        obstacle.transform.position = position;
        obstacle.transform.localScale = size;

        Renderer renderer = obstacle.GetComponent<Renderer>();
        Material material = new Material(Shader.Find("Standard"));
        material.color = color;
        renderer.material = material;

        obstacle.AddComponent<Rigidbody>();
        obstacle.GetComponent<Rigidbody>().isKinematic = true;

        return obstacle;
    }
}
```

### Lab Exercises
1. Implement real-time visualization of LiDAR point clouds in Unity
2. Create a 3D occupancy grid visualization that updates from ROS 2 data
3. Implement camera feed display in the Unity scene
4. Add interactive controls to the Unity visualization

### Expected Results
- Unity visualization connected to ROS 2 simulation
- Real-time robot position and orientation display
- Environment visualization with obstacles
- Sensor data visualization (LiDAR, camera)

## Troubleshooting Common Issues

### Gazebo Simulation Issues
- **Model Not Loading**: Check URDF/SDF syntax, file paths, and dependencies
- **Physics Instability**: Adjust time steps, solver parameters, and mass properties
- **Sensor Data Issues**: Verify plugin configuration and topic names
- **Performance Problems**: Optimize collision meshes and reduce update rates

### Unity Integration Issues
- **Connection Problems**: Check ROS bridge connection and topic names
- **Coordinate System Mismatch**: Verify coordinate system conversions
- **Performance Issues**: Optimize Unity rendering and reduce update frequency
- **Data Synchronization**: Ensure proper timing between systems

### Sensor Fusion Issues
- **Data Alignment**: Ensure all sensors are properly calibrated and synchronized
- **Noise Filtering**: Implement appropriate filtering for sensor data
- **Algorithm Performance**: Optimize fusion algorithms for real-time operation

## Lab Report Template

### Lab Documentation Requirements
Each lab should include:

1. **Objective**: Clear statement of what was implemented
2. **Implementation**: Code snippets and explanations
3. **Results**: What worked, what didn't, and why
4. **Analysis**: Discussion of challenges and solutions
5. **Extensions**: Ideas for improvement or additional features

### Evaluation Criteria
- **Functionality**: Does the implementation work as expected?
- **Code Quality**: Is the code well-structured and documented?
- **Understanding**: Does the student demonstrate understanding of concepts?
- **Creativity**: Are there innovative solutions or extensions?
- **Problem-Solving**: How effectively were challenges addressed?

## Review Questions

1. How do you configure multiple sensors on a robot model in Gazebo?
2. What are the key considerations for sensor fusion in robotics applications?
3. How do you optimize Gazebo simulation performance for real-time applications?
4. What are the challenges of integrating Unity with ROS 2 for visualization?
5. How do you validate the accuracy of simulated sensors against real sensors?

## Next Steps
After completing these practical labs, students should be able to:
- Design and implement comprehensive simulation environments
- Integrate multiple sensor systems for enhanced perception
- Create advanced visualization systems for robotics
- Apply learned concepts to real-world robotics challenges

These hands-on labs provide essential practical experience that bridges the gap between theoretical knowledge and real-world robotics simulation and visualization systems.