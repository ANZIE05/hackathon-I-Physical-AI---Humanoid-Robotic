---
sidebar_position: 5
---

# ROS 2 Practical Labs

## Overview
This section provides hands-on practical labs that reinforce the ROS 2 concepts learned in previous sections. Each lab builds upon the previous knowledge and provides real-world application of ROS 2 principles in Physical AI and Humanoid Robotics contexts.

## Lab 1: Basic ROS 2 Communication

### Objective
Create a simple robot communication system with sensor publishing and command subscription to understand basic ROS 2 concepts.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Basic Python knowledge
- Understanding of ROS 2 fundamentals

### Lab Setup
```bash
# Create workspace
mkdir -p ~/ros2_labs/ws_basic_communication/src
cd ~/ros2_labs/ws_basic_communication

# Create package
ros2 pkg create --build-type ament_python basic_communication_pkg --dependencies rclpy std_msgs sensor_msgs geometry_msgs
```

### Implementation Steps

#### Step 1: Create the Sensor Publisher
Create `basic_communication_pkg/basic_communication_pkg/sensor_publisher.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import math
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publisher for laser scan
        self.laser_publisher = self.create_publisher(LaserScan, 'laser_scan', 10)

        # Create publisher for status
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)

        # Create timer for publishing
        self.timer = self.create_timer(0.5, self.publish_sensor_data)

        self.scan_count = 0
        self.get_logger().info('Sensor Publisher Node Started')

    def publish_sensor_data(self):
        """Publish simulated sensor data"""
        # Create and populate laser scan message
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'

        # Set laser scan parameters
        scan_msg.angle_min = -math.pi / 2  # -90 degrees
        scan_msg.angle_max = math.pi / 2   # 90 degrees
        scan_msg.angle_increment = math.pi / 180  # 1 degree
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0

        # Generate simulated ranges (add some randomness to simulate real sensors)
        num_ranges = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment) + 1
        ranges = []

        for i in range(num_ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            # Simulate a wall at 2 meters in front with some noise
            distance = 2.0 + random.uniform(-0.1, 0.1)
            ranges.append(distance)

        scan_msg.ranges = ranges
        scan_msg.intensities = []  # No intensity data for this example

        # Publish laser scan
        self.laser_publisher.publish(scan_msg)

        # Publish status message
        status_msg = String()
        status_msg.data = f'Scanning environment - count: {self.scan_count}'
        self.status_publisher.publish(status_msg)

        self.scan_count += 1

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Create the Command Subscriber
Create `basic_communication_pkg/basic_communication_pkg/command_subscriber.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

class CommandSubscriber(Node):
    def __init__(self):
        super().__init__('command_subscriber')

        # Create subscriber for velocity commands
        self.cmd_vel_subscription = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        # Create subscriber for status messages
        self.status_subscription = self.create_subscription(
            String, 'robot_status', self.status_callback, 10)

        # Create publisher for processed commands
        self.processed_cmd_publisher = self.create_publisher(Twist, 'processed_cmd_vel', 10)

        # Create publisher for robot state
        self.state_publisher = self.create_publisher(String, 'robot_state', 10)

        # Robot state variables
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.robot_state = 'idle'

        self.get_logger().info('Command Subscriber Node Started')

    def cmd_vel_callback(self, msg):
        """Handle incoming velocity commands"""
        self.linear_velocity = msg.linear.x
        self.angular_velocity = msg.angular.z

        # Process and validate commands
        processed_cmd = self.process_command(msg)

        # Publish processed command
        self.processed_cmd_publisher.publish(processed_cmd)

        # Update robot state
        if abs(self.linear_velocity) > 0.01 or abs(self.angular_velocity) > 0.01:
            self.robot_state = 'moving'
        else:
            self.robot_state = 'stopped'

        # Publish robot state
        state_msg = String()
        state_msg.data = f'{self.robot_state} - linear: {self.linear_velocity:.2f}, angular: {self.angular_velocity:.2f}'
        self.state_publisher.publish(state_msg)

        self.get_logger().info(f'Received command: linear={self.linear_velocity:.2f}, angular={self.angular_velocity:.2f}')

    def status_callback(self, msg):
        """Handle status messages from other nodes"""
        self.get_logger().info(f'Robot status: {msg.data}')

    def process_command(self, cmd_msg):
        """Process and validate incoming commands"""
        processed_cmd = Twist()

        # Apply safety limits
        MAX_LINEAR = 1.0  # m/s
        MAX_ANGULAR = 1.0  # rad/s

        processed_cmd.linear.x = max(-MAX_LINEAR, min(MAX_LINEAR, cmd_msg.linear.x))
        processed_cmd.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, cmd_msg.angular.z))

        # Apply smoothing (simple low-pass filter)
        alpha = 0.1  # Smoothing factor
        processed_cmd.linear.x = alpha * processed_cmd.linear.x + (1 - alpha) * self.linear_velocity
        processed_cmd.angular.z = alpha * processed_cmd.angular.z + (1 - alpha) * self.angular_velocity

        return processed_cmd

def main(args=None):
    rclpy.init(args=args)
    command_subscriber = CommandSubscriber()

    try:
        rclpy.spin(command_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        command_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Update Package Configuration
Update `basic_communication_pkg/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>basic_communication_pkg</name>
  <version>0.0.0</version>
  <description>Basic ROS 2 communication package for lab exercises</description>
  <maintainer email="student@university.edu">Student</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Update `basic_communication_pkg/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'basic_communication_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@university.edu',
    description='Basic ROS 2 communication package for lab exercises',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_publisher = basic_communication_pkg.sensor_publisher:main',
            'command_subscriber = basic_communication_pkg.command_subscriber:main',
        ],
    },
)
```

#### Step 4: Build and Test
```bash
# Build the package
cd ~/ros2_labs/ws_basic_communication
colcon build --packages-select basic_communication_pkg

# Source the workspace
source install/setup.bash

# Run the publisher in one terminal
ros2 run basic_communication_pkg sensor_publisher

# In another terminal, run the subscriber
ros2 run basic_communication_pkg command_subscriber

# In a third terminal, send commands
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'
```

### Lab Exercises
1. Modify the sensor publisher to simulate different environments (open space, narrow corridor)
2. Add more sophisticated command processing in the subscriber
3. Implement a simple state machine to track robot behavior
4. Add parameter configuration for the nodes

### Expected Results
- Two nodes communicating via ROS 2 topics
- Proper logging and error handling
- Working command processing with safety limits
- Demonstrated understanding of basic ROS 2 concepts

## Lab 2: Advanced ROS 2 with URDF and Simulation

### Objective
Create a simulated robot model with URDF, integrate it with ROS 2, and implement basic navigation behaviors.

### Prerequisites
- Lab 1 completed
- Gazebo installed
- Understanding of URDF basics

### Lab Setup
```bash
mkdir -p ~/ros2_labs/ws_robot_model/src
cd ~/ros2_labs/ws_robot_model

# Create package
ros2 pkg create --build-type ament_python robot_model_pkg --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs tf2_ros
```

### Implementation Steps

#### Step 1: Create Robot URDF Model
Create `robot_model_pkg/robot_model_pkg/urdf/simple_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="0.15" />
  <xacro:property name="wheel_pos_x" value="0.2" />
  <xacro:property name="wheel_pos_y" value="0.2" />

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
    <origin xyz="${base_length/2} 0 ${base_height/2}" rpy="0 0 0"/>
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

#### Step 2: Create Navigation Node
Create `robot_model_pkg/robot_model_pkg/navigation_node.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.odom_subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        # Timers
        self.navigation_timer = self.create_timer(0.1, self.navigation_loop)

        # Robot state
        self.latest_scan = None
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.target_pose = {'x': 5.0, 'y': 5.0}  # Target location

        # Navigation state
        self.navigation_state = 'exploring'  # exploring, navigating, stopped
        self.obstacle_detected = False
        self.safety_distance = 0.5  # meters

        self.get_logger().info('Navigation Node Started')

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
                    self.obstacle_detected = min_range < self.safety_distance
                else:
                    self.obstacle_detected = False
            else:
                self.obstacle_detected = False

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

    def navigation_loop(self):
        """Main navigation logic"""
        if self.latest_scan is None:
            return

        cmd = Twist()

        if self.obstacle_detected:
            # Emergency stop or obstacle avoidance
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn right to avoid obstacle
            self.navigation_state = 'avoiding'
        else:
            # Navigate towards target
            dx = self.target_pose['x'] - self.robot_pose['x']
            dy = self.target_pose['y'] - self.robot_pose['y']
            distance_to_target = math.sqrt(dx*dx + dy*dy)

            if distance_to_target < 0.5:  # Close enough to target
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.navigation_state = 'reached_target'
            else:
                # Calculate desired heading
                desired_theta = math.atan2(dy, dx)
                angle_diff = desired_theta - self.robot_pose['theta']

                # Normalize angle difference
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi

                # PID-like control for angular velocity
                angular_kp = 1.0
                cmd.angular.z = angular_kp * angle_diff

                # Limit angular velocity
                cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))

                # Set linear velocity based on angular error
                if abs(angle_diff) < 0.2:  # Only move forward if roughly aligned
                    cmd.linear.x = 0.5
                else:
                    cmd.linear.x = 0.1  # Slow down when turning

                self.navigation_state = 'navigating'

        # Publish command
        self.cmd_vel_publisher.publish(cmd)

        # Log navigation state
        self.get_logger().info(f'Navigation: {self.navigation_state}, '
                              f'Pos: ({self.robot_pose["x"]:.2f}, {self.robot_pose["y"]:.2f}), '
                              f'Obstacle: {self.obstacle_detected}')

def main(args=None):
    rclpy.init(args=args)
    navigation_node = NavigationNode()

    try:
        rclpy.spin(navigation_node)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Create Launch File
Create `robot_model_pkg/launch/robot_navigation.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('robot_model_pkg')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': open(os.path.join(pkg_share, 'urdf', 'simple_robot.urdf')).read()
            }]),

        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{'use_sim_time': use_sim_time}]),

        # Navigation node
        Node(
            package='robot_model_pkg',
            executable='navigation_node',
            name='navigation_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'),

        # RViz2 for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_share, 'config', 'robot_navigation.rviz')],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=launch.conditions.IfCondition(LaunchConfiguration('rviz', default='true'))),
    ])
```

#### Step 4: Build and Test
```bash
# Build the package
cd ~/ros2_labs/ws_robot_model
colcon build --packages-select robot_model_pkg

# Source the workspace
source install/setup.bash

# Launch the robot model
ros2 launch robot_model_pkg robot_navigation.launch.py
```

### Lab Exercises
1. Add more sophisticated navigation algorithms (Dijkstra, A*)
2. Implement obstacle mapping using laser scan data
3. Add camera processing for visual navigation
4. Create a simple SLAM system using the sensor data

### Expected Results
- Working URDF robot model
- Navigation system that can avoid obstacles
- Proper integration of sensor data
- Demonstrated understanding of robot state and control

## Lab 3: AI Integration with ROS 2

### Objective
Integrate a simple AI model (computer vision) with ROS 2 for object detection and navigation.

### Prerequisites
- Labs 1 and 2 completed
- Python OpenCV and NumPy installed
- Understanding of rclpy integration

### Lab Setup
```bash
mkdir -p ~/ros2_labs/ws_ai_integration/src
cd ~/ros2_labs/ws_ai_integration

# Create package
ros2 pkg create --build-type ament_python ai_integration_pkg --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge
```

### Implementation Steps

#### Step 1: Create AI Processing Node
Create `ai_integration_pkg/ai_integration_pkg/object_detection_node.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.detection_publisher = self.create_publisher(String, 'object_detections', 10)
        self.target_publisher = self.create_publisher(Point, 'target_location', 10)

        # Subscribers
        self.image_subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        # Timers
        self.ai_processing_timer = self.create_timer(0.2, self.ai_processing_loop)

        # AI state
        self.latest_image = None
        self.latest_scan = None
        self.detections = []
        self.target_location = None
        self.avoid_obstacles = True

        # AI model parameters
        self.hsv_lower_red1 = np.array([0, 50, 50])
        self.hsv_upper_red1 = np.array([10, 255, 255])
        self.hsv_lower_red2 = np.array([170, 50, 50])
        self.hsv_upper_red2 = np.array([180, 255, 255])

        self.get_logger().info('Object Detection Node Started')

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def scan_callback(self, msg):
        """Process incoming laser scan"""
        self.latest_scan = msg

    def ai_processing_loop(self):
        """Main AI processing loop"""
        if self.latest_image is not None:
            # Run object detection
            self.detections = self.detect_objects(self.latest_image)

            # Publish detection results
            if self.detections:
                detection_msg = String()
                detection_msg.data = f'Detected {len(self.detections)} objects'
                self.detection_publisher.publish(detection_msg)

                # Select target (closest red object)
                self.target_location = self.select_target(self.detections)

                if self.target_location:
                    # Publish target location
                    target_msg = Point()
                    target_msg.x = self.target_location['x']
                    target_msg.y = self.target_location['y']
                    target_msg.z = 0.0
                    self.target_publisher.publish(target_msg)

                    # Generate navigation command
                    cmd = self.generate_navigation_command()
                    self.cmd_vel_publisher.publish(cmd)
            else:
                # No objects detected, continue exploration
                cmd = self.explore_behavior()
                self.cmd_vel_publisher.publish(cmd)

    def detect_objects(self, image):
        """Detect objects using color-based segmentation"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create masks for red color (red wraps around in HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower_red1, self.hsv_upper_red1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_red2, self.hsv_upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            # Filter by area (avoid tiny detections)
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center and relative position
                center_x = x + w // 2
                center_y = y + h // 2

                # Convert to relative coordinates (0-1)
                img_height, img_width = image.shape[:2]
                rel_x = center_x / img_width
                rel_y = center_y / img_height

                detection = {
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'relative_pos': (rel_x, rel_y),
                    'area': area
                }

                detections.append(detection)

        # Sort detections by area (largest first)
        detections.sort(key=lambda x: x['area'], reverse=True)

        # Visualize detections (optional)
        vis_image = image.copy()
        for detection in detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(vis_image, detection['center'], 5, (255, 0, 0), -1)

        # Display visualization (for debugging)
        cv2.imshow('Object Detection', vis_image)
        cv2.waitKey(1)

        return detections

    def select_target(self, detections):
        """Select the best target from detections"""
        if not detections:
            return None

        # For now, select the largest detection
        largest_detection = detections[0]

        # Convert relative position to navigation target
        rel_x, rel_y = largest_detection['relative_pos']

        # Map relative position to navigation commands
        # This is a simplified mapping - in reality, you'd use more sophisticated logic
        target = {
            'x': rel_x * 2 - 1,  # Convert 0-1 to -1 to 1
            'y': (1 - rel_y) * 2 - 1,  # Convert 0-1 to 1 to -1 (invert Y)
            'area': largest_detection['area']
        }

        return target

    def generate_navigation_command(self):
        """Generate navigation command based on target"""
        cmd = Twist()

        if self.target_location:
            # Simple proportional controller
            kp_linear = 0.5
            kp_angular = 1.0

            # Target relative to center of image (-1 to 1)
            target_x = self.target_location['x']  # Horizontal position
            target_y = self.target_location['y']  # Vertical position (inverted)

            # Move toward target
            cmd.linear.x = kp_linear * min(1.0, max(-1.0, 1 - abs(target_y)))  # Move forward based on vertical position
            cmd.angular.z = -kp_angular * target_x  # Turn toward horizontal position

            # Adjust for object size (move closer to larger objects)
            size_factor = min(1.0, self.target_location['area'] / 10000)
            cmd.linear.x *= size_factor

        return cmd

    def explore_behavior(self):
        """Behavior when no objects are detected"""
        cmd = Twist()

        # Simple exploration pattern: move forward unless obstacle detected
        if self.latest_scan and len(self.latest_scan.ranges) > 0:
            # Check for obstacles in front
            front_ranges = self.latest_scan.ranges[len(self.latest_scan.ranges)//2-10:len(self.latest_scan.ranges)//2+10]
            min_range = min([r for r in front_ranges if 0 < r < float('inf')], default=float('inf'))

            if min_range < 0.8:  # Obstacle too close
                cmd.linear.x = 0.0
                cmd.angular.z = 0.3  # Turn right
            else:
                cmd.linear.x = 0.3  # Move forward
                cmd.angular.z = 0.0
        else:
            # Default exploration
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0

        return cmd

def main(args=None):
    rclpy.init(args=args)
    object_detection_node = ObjectDetectionNode()

    try:
        rclpy.spin(object_detection_node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        pass
    finally:
        object_detection_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Update Package Configuration
Update `ai_integration_pkg/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>ai_integration_pkg</name>
  <version>0.0.0</version>
  <description>AI integration package for ROS 2 lab exercises</description>
  <maintainer email="student@university.edu">Student</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>cv_bridge</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Update `ai_integration_pkg/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'ai_integration_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@university.edu',
    description='AI integration package for ROS 2 lab exercises',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = ai_integration_pkg.object_detection_node:main',
        ],
    },
)
```

#### Step 3: Build and Test
```bash
# Build the package
cd ~/ros2_labs/ws_ai_integration
colcon build --packages-select ai_integration_pkg

# Source the workspace
source install/setup.bash

# Run the AI integration node
ros2 run ai_integration_pkg object_detection_node
```

### Lab Exercises
1. Implement more sophisticated object detection (using deep learning models)
2. Add multiple object tracking capabilities
3. Integrate with a real camera feed
4. Create a more complex navigation behavior based on object properties

### Expected Results
- Working object detection system
- AI-driven navigation based on visual input
- Proper integration of computer vision with ROS 2
- Demonstrated understanding of AI-ROS integration

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

## Troubleshooting Common Issues

### ROS 2 Communication Issues
- **Node Discovery**: Ensure proper network configuration and domain IDs
- **Topic Names**: Verify topic names match between publishers and subscribers
- **Message Types**: Ensure correct message types are used
- **QoS Settings**: Check QoS policies match between nodes

### Performance Issues
- **CPU Usage**: Monitor CPU usage and optimize processing loops
- **Memory Leaks**: Use proper cleanup and avoid circular references
- **Timing**: Ensure proper timer intervals for real-time performance

### Simulation Issues
- **Model Loading**: Verify URDF files are properly formatted
- **Physics Parameters**: Check mass, inertia, and friction values
- **Sensor Data**: Validate sensor ranges and data quality

## Review Questions

1. How do you debug communication issues between ROS 2 nodes?
2. What are common performance bottlenecks in Python ROS 2 nodes?
3. How do you handle real-time constraints in Python-based robotic systems?
4. What are the advantages of using launch files for system deployment?
5. How would you extend the basic navigation system to handle dynamic obstacles?

## Next Steps
After completing these practical labs, students should be able to:
- Design and implement complex ROS 2 systems
- Integrate AI/ML models with robotic platforms
- Create robust and efficient robotic applications
- Apply learned concepts to real-world robotics challenges

These hands-on labs provide essential practical experience that bridges the gap between theoretical knowledge and real-world robotic system implementation.