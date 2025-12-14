---
sidebar_position: 4
---

# ROS 2 Python Integration (rclpy)

## Overview
The ROS 2 Python Client Library (rclpy) provides Python bindings for ROS 2, enabling integration of Python-based AI agents and applications with ROS 2 systems. This section covers advanced Python integration techniques for Physical AI applications.

## Learning Objectives
By the end of this section, students will be able to:
- Implement complex ROS 2 nodes using rclpy
- Integrate Python AI libraries with ROS 2
- Design efficient Python-ROS communication patterns
- Create custom message types and services
- Implement real-time Python-ROS integration

## Key Concepts

### rclpy Architecture
- **rcl**: ROS Client Library (C-based)
- **rclpy**: Python bindings for rcl
- **Middleware**: DDS-based communication layer
- **Executor**: Manages node callbacks and events

### Python in Robotics
- **AI Integration**: Easy integration with PyTorch, TensorFlow, OpenCV
- **Rapid Prototyping**: Fast development and testing
- **Data Processing**: Excellent for data analysis and visualization
- **Scripting**: Automated testing and deployment

### Performance Considerations
- **Latency**: Python's GIL affects real-time performance
- **Memory**: Higher memory usage than C++
- **CPU**: Interpreter overhead for computation-intensive tasks
- **Threading**: Careful consideration for real-time systems

## Practical Implementation

### Advanced Node Structure
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import threading
import time

class AdvancedPythonNode(Node):
    def __init__(self):
        super().__init__('advanced_python_node')

        # Quality of service profiles
        self.sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.control_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', self.control_qos)
        self.status_publisher = self.create_publisher(String, 'status', 10)

        # Subscribers
        self.image_subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, self.sensor_qos)
        self.laser_subscription = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, self.sensor_qos)

        # Timers
        self.processing_timer = self.create_timer(0.1, self.processing_loop)
        self.status_timer = self.create_timer(1.0, self.status_callback)

        # Parameters
        self.declare_parameter('robot_name', 'python_robot')
        self.declare_parameter('processing_rate', 10)
        self.robot_name = self.get_parameter('robot_name').value

        # Internal state
        self.latest_image = None
        self.latest_laser = None
        self.processing_rate = self.get_parameter('processing_rate').value
        self.state_lock = threading.Lock()

        self.get_logger().info(f'Advanced Python Node initialized: {self.robot_name}')

    def image_callback(self, msg):
        """Handle incoming image messages"""
        with self.state_lock:
            self.latest_image = msg
            self.get_logger().debug(f'Received image: {msg.width}x{msg.height}')

    def laser_callback(self, msg):
        """Handle incoming laser scan messages"""
        with self.state_lock:
            self.latest_laser = msg
            self.get_logger().debug(f'Received laser scan with {len(msg.ranges)} points')

    def processing_loop(self):
        """Main processing loop"""
        with self.state_lock:
            if self.latest_image is not None and self.latest_laser is not None:
                # Process sensor data
                cmd = self.process_sensors(self.latest_image, self.latest_laser)

                # Publish command
                self.cmd_publisher.publish(cmd)

                # Update status
                status_msg = String()
                status_msg.data = f'Processing at {self.processing_rate}Hz'
                self.status_publisher.publish(status_msg)

    def process_sensors(self, image, laser):
        """Process sensor data and return command"""
        # This is where AI/ML processing would happen
        cmd = Twist()

        # Simple obstacle avoidance based on laser
        if laser.ranges:
            min_distance = min([r for r in laser.ranges if r > 0], default=float('inf'))
            if min_distance < 1.0:  # Too close to obstacle
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn away
            else:
                cmd.linear.x = 0.5
                cmd.angular.z = 0.0

        return cmd

    def status_callback(self):
        """Periodic status update"""
        self.get_logger().info(f'Node status: Active - {self.robot_name}')

def main(args=None):
    rclpy.init(args=args)

    node = AdvancedPythonNode()

    # Use multi-threaded executor for better performance
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Custom Message and Service Integration
```python
# Custom message integration example
from rclpy.node import Node
from my_robot_msgs.msg import RobotState, SensorData  # Custom messages
from my_robot_msgs.srv import SetMode, GetRobotInfo  # Custom services

class CustomMessageNode(Node):
    def __init__(self):
        super().__init__('custom_message_node')

        # Publishers with custom messages
        self.state_publisher = self.create_publisher(RobotState, 'robot_state', 10)
        self.sensor_publisher = self.create_publisher(SensorData, 'sensor_data', 10)

        # Service server
        self.set_mode_service = self.create_service(
            SetMode, 'set_robot_mode', self.set_mode_callback)
        self.get_info_service = self.create_service(
            GetRobotInfo, 'get_robot_info', self.get_info_callback)

        # Service client
        self.info_client = self.create_client(GetRobotInfo, 'get_robot_info')

        self.current_mode = 'idle'
        self.robot_info = {'battery': 100.0, 'temperature': 25.0}

    def set_mode_callback(self, request, response):
        """Handle mode change requests"""
        old_mode = self.current_mode
        self.current_mode = request.mode

        response.success = True
        response.message = f'Mode changed from {old_mode} to {self.current_mode}'

        self.get_logger().info(response.message)
        return response

    def get_info_callback(self, request, response):
        """Handle info requests"""
        response.info = str(self.robot_info)
        response.mode = self.current_mode
        response.success = True
        return response

    def call_get_info(self):
        """Call service to get robot info"""
        while not self.info_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = GetRobotInfo.Request()
        future = self.info_client.call_async(request)
        return future
```

### AI Integration with rclpy
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from cv_bridge import CvBridge
import torch
import torch.nn as nn

class AIPoweredNode(Node):
    def __init__(self):
        super().__init__('ai_powered_node')

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # AI model (example: simple CNN for obstacle detection)
        self.ai_model = self.load_model()

        # Publishers and subscribers
        self.image_subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.ai_timer = self.create_timer(0.2, self.ai_processing_loop)

        # AI state
        self.latest_processed_image = None
        self.ai_results = None
        self.ai_processing_active = True

    def load_model(self):
        """Load or create AI model"""
        # Example: Simple model for demonstration
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.fc1 = nn.Linear(32 * 120 * 160, 120)  # Adjusted for 480x640/4
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 3)  # 3 classes: clear, obstacle_left, obstacle_right

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 32 * 120 * 160)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return torch.softmax(x, dim=1)

        model = SimpleCNN()
        # In practice, you would load a pre-trained model
        # model.load_state_dict(torch.load('model_weights.pth'))
        model.eval()
        return model

    def image_callback(self, msg):
        """Process incoming image and prepare for AI processing"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess for AI model (resize, normalize)
            processed_image = self.preprocess_image(cv_image)

            # Store for AI processing
            self.latest_processed_image = processed_image

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def preprocess_image(self, cv_image):
        """Preprocess image for AI model"""
        # Resize image to expected input size (example: 480x640)
        resized = cv2.resize(cv_image, (640, 480))

        # Convert to tensor format (CHW, normalized)
        tensor_image = resized.astype(np.float32) / 255.0
        tensor_image = np.transpose(tensor_image, (2, 0, 1))  # HWC to CHW
        tensor_image = np.expand_dims(tensor_image, axis=0)   # Add batch dimension

        return torch.from_numpy(tensor_image)

    def ai_processing_loop(self):
        """Main AI processing loop"""
        if self.latest_processed_image is not None and self.ai_processing_active:
            try:
                # Run AI inference
                with torch.no_grad():
                    output = self.ai_model(self.latest_processed_image)
                    probabilities = torch.softmax(output[0], dim=0)

                # Interpret results
                class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[class_idx].item()

                self.ai_results = {
                    'class_idx': class_idx,
                    'confidence': confidence,
                    'probabilities': probabilities.tolist()
                }

                # Generate command based on AI results
                cmd = self.generate_command_from_ai()
                self.cmd_publisher.publish(cmd)

            except Exception as e:
                self.get_logger().error(f'AI processing error: {e}')

    def generate_command_from_ai(self):
        """Generate robot command based on AI results"""
        cmd = Twist()

        if self.ai_results:
            class_idx = self.ai_results['class_idx']
            confidence = self.ai_results['confidence']

            if confidence < 0.7:  # Low confidence
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                return cmd

            # Class 0: Clear path
            # Class 1: Obstacle on the left
            # Class 2: Obstacle on the right
            if class_idx == 0:  # Clear
                cmd.linear.x = 0.5
                cmd.angular.z = 0.0
            elif class_idx == 1:  # Left obstacle
                cmd.linear.x = 0.3
                cmd.angular.z = 0.3  # Turn right
            elif class_idx == 2:  # Right obstacle
                cmd.linear.x = 0.3
                cmd.angular.z = -0.3  # Turn left

        return cmd

def main(args=None):
    rclpy.init(args=args)
    ai_node = AIPoweredNode()
    rclpy.spin(ai_node)
    ai_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Parameter Management and Configuration
```python
from rclpy.node import Node
from rclpy.parameter import Parameter
import json
import yaml

class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with descriptions and constraints
        self.declare_parameter('control.linear_velocity', 0.5)
        self.declare_parameter('control.angular_velocity', 0.3)
        self.declare_parameter('safety.min_distance', 0.5)
        self.declare_parameter('ai.model_path', '/default/model/path')
        self.declare_parameter('debug.enabled', False)

        # Load parameters with default values
        self.linear_velocity = self.get_parameter('control.linear_velocity').value
        self.angular_velocity = self.get_parameter('control.angular_velocity').value
        self.min_distance = self.get_parameter('safety.min_distance').value
        self.model_path = self.get_parameter('ai.model_path').value
        self.debug_enabled = self.get_parameter('debug.enabled').value

        # Set parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Load configuration from file
        self.load_config_from_file('/path/to/config.yaml')

    def parameter_callback(self, params):
        """Handle parameter changes"""
        changes = {}

        for param in params:
            if param.name == 'control.linear_velocity':
                if 0.0 <= param.value <= 2.0:  # Reasonable limits
                    self.linear_velocity = param.value
                    changes[param.name] = param.value
                else:
                    return SetParametersResult(successful=False, reason='Invalid velocity range')
            elif param.name == 'control.angular_velocity':
                if -1.0 <= param.value <= 1.0:
                    self.angular_velocity = param.value
                    changes[param.name] = param.value
                else:
                    return SetParametersResult(successful=False, reason='Invalid angular velocity range')

        if changes:
            self.get_logger().info(f'Parameters updated: {changes}')

        return SetParametersResult(successful=True)

    def load_config_from_file(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Update parameters from config file
            for param_name, param_value in config.get('parameters', {}).items():
                self.set_parameters([Parameter(param_name, Parameter.Type.NOT_SET, param_value)])

        except FileNotFoundError:
            self.get_logger().warn(f'Config file not found: {config_path}')
        except yaml.YAMLError as e:
            self.get_logger().error(f'Error parsing config file: {e}')

    def save_config_to_file(self, config_path):
        """Save current parameters to configuration file"""
        config = {
            'parameters': {
                'control.linear_velocity': self.linear_velocity,
                'control.angular_velocity': self.angular_velocity,
                'safety.min_distance': self.min_distance,
                'ai.model_path': self.model_path,
                'debug.enabled': self.debug_enabled
            }
        }

        with open(config_path, 'w') as file:
            yaml.dump(config, file)
```

## Advanced Topics

### Asynchronous Processing
```python
import asyncio
import concurrent.futures
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class AsyncNode(Node):
    def __init__(self):
        super().__init__('async_node')

        # Use a mutually exclusive callback group for async operations
        self.async_group = MutuallyExclusiveCallbackGroup()

        # Subscription with async callback group
        self.subscription = self.create_subscription(
            String, 'input_topic', self.async_callback, 10,
            callback_group=self.async_group)

        # Create thread pool for CPU-intensive tasks
        self.executor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def async_callback(self, msg):
        """Non-blocking callback that schedules async work"""
        # Schedule heavy computation in thread pool
        future = self.executor_pool.submit(self.heavy_computation, msg.data)

        # Add done callback for result processing
        future.add_done_callback(self.computation_done)

    def heavy_computation(self, input_data):
        """CPU-intensive computation that runs in thread pool"""
        # Simulate heavy computation
        import time
        time.sleep(0.1)  # Simulate processing time
        return f"Processed: {input_data}"

    def computation_done(self, future):
        """Handle completion of async computation"""
        try:
            result = future.result()
            self.get_logger().info(f'Computation result: {result}')
        except Exception as e:
            self.get_logger().error(f'Computation failed: {e}')
```

### Memory Management
```python
import gc
import weakref
from collections import deque

class MemoryEfficientNode(Node):
    def __init__(self):
        super().__init__('memory_efficient_node')

        # Use deque for efficient append/pop operations
        self.image_buffer = deque(maxlen=10)  # Keep only last 10 images

        # Use weak references to avoid circular references
        self.nodes_ref = weakref.WeakSet()

        # Memory monitoring
        self.memory_timer = self.create_timer(5.0, self.memory_monitor)

    def memory_monitor(self):
        """Monitor and report memory usage"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        self.get_logger().info(f'Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB')

        # Force garbage collection periodically
        gc.collect()
```

## Performance Optimization

### Efficient Message Handling
```python
class EfficientMessageNode(Node):
    def __init__(self):
        super().__init__('efficient_message_node')

        # Use efficient data structures
        self.message_cache = {}
        self.message_counter = 0

        # Throttle message processing
        self.process_every_n = 3  # Process every 3rd message
        self.subscription = self.create_subscription(
            LaserScan, 'scan', self.throttled_callback, 10)

    def throttled_callback(self, msg):
        """Process messages with throttling"""
        self.message_counter += 1

        if self.message_counter % self.process_every_n == 0:
            self.process_message(msg)

    def process_message(self, msg):
        """Efficient message processing"""
        # Use list comprehension instead of loops where possible
        valid_ranges = [r for r in msg.ranges if 0 < r < 10.0]

        if valid_ranges:
            min_range = min(valid_ranges)
            # Process min_range...
```

## Best Practices

### Python-ROS Integration Patterns
- **Threading**: Use MultiThreadedExecutor for I/O bound operations
- **Memory**: Use deque for buffers, avoid memory leaks with proper cleanup
- **Error Handling**: Implement comprehensive exception handling
- **Logging**: Use appropriate log levels for debugging and monitoring
- **Configuration**: Use parameters for runtime configuration

### Performance Considerations
- **Real-time**: Python may not be suitable for hard real-time requirements
- **AI Integration**: Python excellent for ML/AI integration
- **Prototyping**: Fast development and testing capabilities
- **Deployment**: Consider PyInstaller or similar for deployment

## Practical Lab: AI-ROS Integration

### Lab Objective
Create a Python node that integrates a simple AI model (object detection) with ROS 2 for navigation.

### Implementation Steps
1. Create a node that subscribes to camera images
2. Implement a simple AI model for object detection
3. Generate navigation commands based on AI results
4. Test the integration in simulation

### Expected Outcome
- Working Python-AI-ROS integration
- Proper error handling and performance optimization
- Demonstrated understanding of rclpy concepts

## Review Questions

1. What are the advantages and disadvantages of using Python vs C++ for ROS 2 nodes?
2. How do you handle real-time requirements in Python-ROS integration?
3. Explain the parameter management system in rclpy.
4. What are callback groups and why are they important?
5. How do you optimize Python nodes for better performance?

## Next Steps
After mastering rclpy integration, students should proceed to:
- Advanced simulation with Gazebo
- Integration with NVIDIA Isaac tools
- Vision-Language-Action system development
- Real-world deployment considerations

This comprehensive guide to rclpy integration enables students to create sophisticated Python-based robotic applications that leverage AI/ML capabilities while maintaining compatibility with ROS 2 systems.