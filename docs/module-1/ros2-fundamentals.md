---
sidebar_position: 1
---

# ROS 2 Fundamentals

## Overview
Robot Operating System 2 (ROS 2) is the foundational middleware for modern robotics applications. This module introduces the core concepts, architecture, and practical implementation of ROS 2 for Physical AI and Humanoid Robotics applications.

## Learning Objectives
By the end of this section, students will be able to:
- Explain the core concepts and architecture of ROS 2
- Implement basic ROS 2 nodes and communication patterns
- Understand the differences between ROS 1 and ROS 2
- Set up and configure a ROS 2 development environment
- Use ROS 2 command-line tools for system introspection

## Key Concepts

### What is ROS 2?
ROS 2 is the next-generation Robot Operating System that provides:
- **Middleware**: Communication infrastructure between robot components
- **Development Tools**: Debugging, visualization, and testing utilities
- **Package Management**: Standardized way to organize and distribute robotic software
- **Hardware Abstraction**: Common interfaces for sensors, actuators, and other hardware

### ROS 2 vs. ROS 1
| Feature | ROS 1 | ROS 2 |
|---------|--------|--------|
| Communication | Custom TCP/UDP | DDS-based |
| Real-time Support | Limited | Enhanced |
| Multi-robot Support | Complex | Built-in |
| Security | None | Built-in security |
| Deployment | Single machine | Distributed |

### Core Architecture Components

#### Nodes
- **Definition**: Processes that perform computation
- **Purpose**: Encapsulate robot functionality
- **Implementation**: Can be written in C++ or Python
- **Communication**: Nodes communicate via topics, services, and actions

#### Topics
- **Definition**: Named buses for data transmission
- **Pattern**: Publish/Subscribe communication
- **Use Case**: Continuous data streams (sensor data, commands)
- **Characteristics**: Many-to-many, asynchronous

#### Services
- **Definition**: Request/Response communication pattern
- **Pattern**: Synchronous client-server interaction
- **Use Case**: One-time requests (calibration, configuration)
- **Characteristics**: One-to-one, blocking calls

#### Actions
- **Definition**: Goal-oriented communication pattern
- **Pattern**: Asynchronous request with feedback and status
- **Use Case**: Long-running tasks (navigation, manipulation)
- **Characteristics**: Goal, feedback, result structure

## Practical Implementation

### Creating Your First ROS 2 Package
```bash
# Create a new workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Create a new package
ros2 pkg create --build-type ament_python my_robot_package --dependencies rclpy std_msgs

# Navigate to package directory
cd src/my_robot_package
```

### Basic Publisher Node
```python
# my_robot_package/my_robot_package/talker.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = TalkerNode()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Basic Subscriber Node
```python
# my_robot_package/my_robot_package/listener.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = ListenerNode()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Package Configuration
```xml
<!-- my_robot_package/package.xml -->
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

```python
# my_robot_package/setup.py
from setuptools import find_packages, setup

package_name = 'my_robot_package'

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
    maintainer='user',
    maintainer_email='user@example.com',
    description='Example ROS 2 package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_package.talker:main',
            'listener = my_robot_package.listener:main',
        ],
    },
)
```

## ROS 2 Communication Patterns

### Publisher-Subscriber Pattern
The most common communication pattern in ROS 2:

```python
# Publisher implementation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(LaserScan, 'laser_scan', 10)

    def publish_scan(self, scan_data):
        msg = LaserScan()
        msg.ranges = scan_data
        self.publisher.publish(msg)
```

### Service Client-Server Pattern
For request-response communication:

```python
# Service server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddService(Node):
    def __init__(self):
        super().__init__('add_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_callback)

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response

# Service client
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
import sys

class AddClient(Node):
    def __init__(self):
        super().__init__('add_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

### Action Client-Server Pattern
For goal-oriented tasks:

```python
# Action server
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Quality of Service (QoS) Settings

### Understanding QoS
Quality of Service settings control communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Configure QoS for reliable communication
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)

# For sensor data (fast, may lose some messages)
sensor_qos = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST
)

# Publisher with custom QoS
publisher = self.create_publisher(String, 'topic_name', qos_profile)
```

## Launch Files

### Creating Launch Files
Launch files allow you to start multiple nodes at once:

```python
# my_robot_package/launch/my_launch_file.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='talker',
            name='talker_node',
            parameters=[
                {'param_name': 'param_value'}
            ],
            remappings=[
                ('original_topic', 'new_topic')
            ]
        ),
        Node(
            package='my_robot_package',
            executable='listener',
            name='listener_node'
        )
    ])
```

## ROS 2 Command-Line Tools

### Essential Commands
```bash
# List all topics
ros2 topic list

# Echo messages from a topic
ros2 topic echo /chatter

# Publish to a topic
ros2 topic pub /chatter std_msgs/String "data: 'Hello'"

# List all services
ros2 service list

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"

# List all actions
ros2 action list

# Send an action goal
ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci "{order: 5}"

# List all nodes
ros2 node list

# Show node information
ros2 node info /talker
```

## Best Practices

### Node Design
- **Single Responsibility**: Each node should have one clear purpose
- **Error Handling**: Implement proper exception handling
- **Logging**: Use appropriate logging levels (info, warn, error)
- **Parameters**: Use parameters for configurable behavior
- **Shutdown**: Implement proper cleanup during shutdown

### Performance Considerations
- **Message Rates**: Optimize message publishing rates
- **Data Size**: Minimize message payload sizes
- **Threading**: Use appropriate executor types
- **Memory**: Manage memory efficiently in long-running nodes

## Practical Lab: Basic ROS 2 Communication

### Lab Objective
Create a simple robot communication system with sensor publishing and command subscription.

### Implementation Steps
1. Create a sensor publisher that publishes mock sensor data
2. Create a command subscriber that processes commands
3. Create a launch file to start both nodes
4. Test the communication using ROS 2 command-line tools

### Expected Outcome
- Two nodes communicating via ROS 2 topics
- Proper logging and error handling
- Working launch file
- Demonstrated understanding of basic ROS 2 concepts

## Review Questions

1. What are the main differences between ROS 1 and ROS 2?
2. Explain the publisher-subscriber communication pattern in ROS 2.
3. What is Quality of Service (QoS) and why is it important?
4. How do you create and run a ROS 2 launch file?
5. What are the advantages of using actions over services for long-running tasks?

## Next Steps
After mastering ROS 2 fundamentals, students should proceed to:
- Advanced ROS 2 concepts (services, actions, parameters)
- Integration with Python agents using rclpy
- URDF for robot modeling
- Practical labs with simulation environments

This comprehensive introduction to ROS 2 fundamentals provides the foundation for all subsequent modules in Physical AI and Humanoid Robotics.