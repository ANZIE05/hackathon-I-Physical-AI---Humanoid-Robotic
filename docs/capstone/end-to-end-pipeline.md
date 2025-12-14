---
sidebar_position: 2
---

# End-to-End Pipeline

## Overview
This section details the complete end-to-end pipeline for the autonomous humanoid system with conversational AI. The pipeline integrates all modules learned throughout the course into a cohesive, functional system.

## Pipeline Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    User Interaction                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Voice Input    │  │  Text Input     │  │  Gesture    │ │
│  │  Recognition    │  │  Processing     │  │  Recognition│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ Natural Language Input
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Natural Language Processing                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Speech-to-    │  │  Intent         │  │  Context    │ │
│  │  Text          │  │  Classification  │  │  Modeling   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ Processed Intent
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Task Planning & Reasoning                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  High-level     │  │  Action         │  │  Safety      │ │
│  │  Planning       │  │  Sequencing     │  │  Validation  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ Action Commands
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   ROS 2 Control Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Navigation     │  │  Manipulation   │  │  Humanoid   │ │
│  │  (Nav2)         │  │  (MoveIt)       │  │  Control    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ Low-level Commands
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Robot Execution Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Perception     │  │  Actuation      │  │  Feedback   │ │
│  │  (Sensors)      │  │  (Motors)       │  │  Loop       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Voice-to-Action Pipeline

### Speech Recognition Module
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import speech_recognition as sr

class VoiceToActionNode(Node):
    def __init__(self):
        super().__init__('voice_to_action_node')
        self.subscription = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10)
        self.command_publisher = self.create_publisher(
            String,
            'voice_command',
            10)

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 3000

    def audio_callback(self, msg):
        # Convert audio data to text
        audio_data = sr.AudioData(
            msg.data,
            sample_rate=msg.sample_rate,
            sample_width=msg.sample_width
        )

        try:
            text = self.recognizer.recognize_google(audio_data)
            self.get_logger().info(f'Recognized: {text}')

            # Publish recognized text
            cmd_msg = String()
            cmd_msg.data = text
            self.command_publisher.publish(cmd_msg)

        except sr.UnknownValueError:
            self.get_logger().warn('Could not understand audio')
        except sr.RequestError as e:
            self.get_logger().error(f'Error: {e}')
```

### Natural Language Understanding
```python
import openai
from rclpy.node import Node
from std_msgs.msg import String

class NLUProcessor(Node):
    def __init__(self):
        super().__init__('nlu_processor')
        self.subscription = self.create_subscription(
            String,
            'voice_command',
            self.command_callback,
            10)
        self.intent_publisher = self.create_publisher(
            String,
            'intent',
            10)

        # OpenAI API configuration
        self.client = openai.OpenAI(api_key='your-api-key')

    def command_callback(self, msg):
        # Process natural language command
        user_command = msg.data

        # Define the system prompt for intent classification
        system_prompt = """
        You are a robot command interpreter. Your role is to:
        1. Classify the intent of the user command
        2. Extract relevant parameters
        3. Return a structured JSON response

        Intents include:
        - navigation: Move to a location
        - manipulation: Pick up, place, or manipulate an object
        - information: Ask for information about the environment
        - interaction: Social interaction commands
        - system: System control commands
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Command: {user_command}"}
                ],
                response_format={"type": "json_object"}
            )

            intent_data = response.choices[0].message.content
            # Publish intent to next stage
            intent_msg = String()
            intent_msg.data = intent_data
            self.intent_publisher.publish(intent_msg)

        except Exception as e:
            self.get_logger().error(f'NLU processing error: {e}')
```

## Task Planning and Execution

### High-Level Planner
```python
import json
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

class TaskPlanner(Node):
    def __init__(self):
        super().__init__('task_planner')
        self.subscription = self.create_subscription(
            String,
            'intent',
            self.intent_callback,
            10)

        # Publishers for different action types
        self.nav_publisher = self.create_publisher(
            PoseStamped,
            'navigation_goal',
            10)
        self.manipulation_publisher = self.create_publisher(
            String,
            'manipulation_command',
            10)

    def intent_callback(self, msg):
        try:
            intent_data = json.loads(msg.data)
            intent_type = intent_data.get('intent', 'unknown')

            if intent_type == 'navigation':
                self.handle_navigation(intent_data)
            elif intent_type == 'manipulation':
                self.handle_manipulation(intent_data)
            elif intent_type == 'information':
                self.handle_information(intent_data)
            else:
                self.get_logger().warn(f'Unknown intent: {intent_type}')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in intent data')

    def handle_navigation(self, intent_data):
        # Extract navigation parameters
        location = intent_data.get('parameters', {}).get('location', 'unknown')

        # Convert location to coordinates (simplified)
        pose = self.get_location_pose(location)

        # Publish navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose = pose
        self.nav_publisher.publish(goal_msg)

    def handle_manipulation(self, intent_data):
        # Extract manipulation parameters
        action = intent_data.get('parameters', {}).get('action', 'unknown')
        object_name = intent_data.get('parameters', {}).get('object', 'unknown')

        # Create manipulation command
        cmd_msg = String()
        cmd_msg.data = f"{action} {object_name}"
        self.manipulation_publisher.publish(cmd_msg)

    def get_location_pose(self, location):
        # Simplified location mapping
        locations = {
            'kitchen': [1.0, 2.0, 0.0],
            'living room': [3.0, 1.0, 0.0],
            'bedroom': [0.0, 4.0, 0.0],
            'office': [2.0, 0.0, 0.0]
        }

        if location in locations:
            x, y, theta = locations[location]
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            pose.orientation = self.euler_to_quaternion(0, 0, theta)
            return pose
        else:
            return Pose()  # Default pose

    def euler_to_quaternion(self, roll, pitch, yaw):
        # Convert Euler angles to quaternion
        import math
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion(x=x, y=y, z=z, w=w)
```

## Perception Integration

### Multi-Sensor Fusion Node
```python
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np

class PerceptionFusion(Node):
    def __init__(self):
        super().__init__('perception_fusion')

        # Multiple sensor subscriptions
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)
        self.lidar_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            'pointcloud',
            self.pointcloud_callback,
            10)

        # Perception result publisher
        self.perception_pub = self.create_publisher(
            String,
            'perception_results',
            10)

    def image_callback(self, msg):
        # Process image data
        image = self.ros_image_to_cv2(msg)

        # Object detection using OpenCV or custom model
        detections = self.detect_objects(image)

        # Publish detection results
        self.publish_perception_results(detections)

    def lidar_callback(self, msg):
        # Process LIDAR data for obstacle detection
        ranges = np.array(msg.ranges)
        obstacles = self.detect_obstacles(ranges)

        # Publish obstacle information
        self.publish_perception_results(obstacles)

    def pointcloud_callback(self, msg):
        # Process point cloud data for 3D understanding
        pointcloud_data = self.process_pointcloud(msg)

        # Extract 3D features
        features = self.extract_3d_features(pointcloud_data)

        # Publish 3D features
        self.publish_perception_results(features)

    def ros_image_to_cv2(self, ros_image):
        # Convert ROS image message to OpenCV format
        dtype = np.uint8
        img = np.frombuffer(ros_image.data, dtype=dtype)
        img = img.reshape(ros_image.height, ros_image.width, 3)
        return img

    def detect_objects(self, image):
        # Simplified object detection
        # In practice, use YOLO, SSD, or custom model
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect simple shapes or use pre-trained model
        # Return detected objects with confidence scores
        return {"objects": [], "confidence": 0.0}

    def detect_obstacles(self, ranges):
        # Detect obstacles from LIDAR ranges
        min_distance = 0.5  # meters
        obstacle_angles = []

        for i, range_val in enumerate(ranges):
            if range_val < min_distance and not np.isnan(range_val):
                obstacle_angles.append(i)

        return {"obstacles": obstacle_angles, "count": len(obstacle_angles)}

    def process_pointcloud(self, pointcloud_msg):
        # Process point cloud data
        # Extract 3D information
        return {"points": [], "features": []}

    def extract_3d_features(self, pointcloud_data):
        # Extract 3D features from point cloud
        return {"surfaces": [], "objects": [], "boundaries": []}

    def publish_perception_results(self, results):
        # Publish combined perception results
        results_msg = String()
        results_msg.data = str(results)
        self.perception_pub.publish(results_msg)
```

## Safety and Validation Layer

### Safety Monitor Node
```python
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Subscribe to commands and sensor data
        self.cmd_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.command_callback,
            10)
        self.lidar_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)

        # Override command publisher for safety
        self.safety_cmd_pub = self.create_publisher(
            Twist,
            'safety_cmd_vel',
            10)

        # Safety status publisher
        self.safety_status_pub = self.create_publisher(
            String,
            'safety_status',
            10)

        self.safety_enabled = True
        self.min_distance = 0.5  # meters
        self.emergency_stop = False

    def command_callback(self, msg):
        if not self.safety_enabled:
            self.safety_cmd_pub.publish(msg)
            return

        # Check if command is safe based on sensor data
        if self.emergency_stop:
            # Publish stop command
            stop_cmd = Twist()
            self.safety_cmd_pub.publish(stop_cmd)
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            return

        # Validate command based on obstacle proximity
        safe_cmd = self.validate_command(msg)
        self.safety_cmd_pub.publish(safe_cmd)

    def lidar_callback(self, msg):
        # Check for obstacles in path
        ranges = np.array(msg.ranges)

        # Check forward direction (simplified)
        forward_ranges = ranges[len(ranges)//2-30:len(ranges)//2+30]
        min_range = np.min(forward_ranges)

        if min_range < self.min_distance:
            self.emergency_stop = True
            self.get_logger().warn(f'OBSTACLE DETECTED: {min_range:.2f}m')
        else:
            self.emergency_stop = False

    def validate_command(self, cmd):
        # Validate and potentially modify command for safety
        if self.emergency_stop:
            # Override with stop command
            safe_cmd = Twist()
        elif cmd.linear.x > 0 and self.is_path_blocked():
            # Reduce speed or stop if path is blocked
            safe_cmd = Twist()
            safe_cmd.linear.x = min(cmd.linear.x, 0.1)  # Reduced speed
            safe_cmd.angular.z = cmd.angular.z
        else:
            # Command is safe, pass through
            safe_cmd = cmd

        return safe_cmd

    def is_path_blocked(self):
        # Check if robot's path is blocked
        # This would typically use more sophisticated path planning
        return self.emergency_stop

    def enable_safety(self):
        self.safety_enabled = True
        self.get_logger().info('Safety system enabled')

    def disable_safety(self):
        self.safety_enabled = False
        self.get_logger().warn('Safety system disabled - PROCEED WITH CAUTION')
```

## System Integration and Launch

### Main Launch File
```xml
<launch>
  <!-- Voice recognition node -->
  <node pkg="voice_to_action" exec="voice_to_action_node" name="voice_to_action">
    <param name="model" value="vosk-model-small-en-us-0.15"/>
  </node>

  <!-- Natural Language Understanding -->
  <node pkg="nlu_processor" exec="nlu_processor" name="nlu_processor">
    <param name="openai_api_key" value="$(var openai_api_key)"/>
  </node>

  <!-- Task Planner -->
  <node pkg="task_planner" exec="task_planner" name="task_planner"/>

  <!-- Navigation Stack -->
  <include file="$(find-pkg-share nav2_bringup)/launch/navigation_launch.py">
    <arg name="use_sim_time" value="true"/>
  </include>

  <!-- Manipulation Stack -->
  <include file="$(find-pkg-share moveit_bringup)/launch/moveit.launch.py"/>

  <!-- Perception Fusion -->
  <node pkg="perception_fusion" exec="perception_fusion" name="perception_fusion"/>

  <!-- Safety Monitor -->
  <node pkg="safety_monitor" exec="safety_monitor" name="safety_monitor">
    <param name="min_distance" value="0.5"/>
  </node>

  <!-- Humanoid Control -->
  <node pkg="humanoid_control" exec="humanoid_controller" name="humanoid_controller"/>

  <!-- Main coordinator -->
  <node pkg="capstone_coordinator" exec="coordinator" name="coordinator"/>
</launch>
```

## Performance Optimization

### Real-time Considerations
- **Threading**: Use multi-threaded executors for concurrent processing
- **Message Queues**: Optimize queue sizes for real-time performance
- **Computational Efficiency**: Optimize algorithms for real-time execution
- **Memory Management**: Use efficient data structures and memory pools

### Resource Management
- **CPU Affinity**: Assign critical nodes to specific CPU cores
- **Priority Scheduling**: Set real-time priorities for safety-critical nodes
- **Memory Allocation**: Pre-allocate memory for performance-critical operations
- **I/O Optimization**: Optimize sensor data processing pipelines

## Testing and Validation

### Unit Testing
```python
import unittest
from task_planner import TaskPlanner

class TestTaskPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = TaskPlanner()

    def test_navigation_intent(self):
        # Test navigation intent processing
        intent_data = '{"intent": "navigation", "parameters": {"location": "kitchen"}}'
        result = self.planner.process_intent(intent_data)
        self.assertEqual(result['action'], 'navigate')

    def test_manipulation_intent(self):
        # Test manipulation intent processing
        intent_data = '{"intent": "manipulation", "parameters": {"action": "pick", "object": "cup"}}'
        result = self.planner.process_intent(intent_data)
        self.assertEqual(result['action'], 'manipulate')

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing
- **Simulation Testing**: Test complete pipeline in Gazebo simulation
- **Hardware-in-the-loop**: Test with real sensors and actuators
- **Performance Testing**: Validate real-time performance requirements
- **Safety Testing**: Verify safety system functionality

## Deployment Considerations

### Hardware Requirements
- **Compute**: NVIDIA Jetson Orin AGX or equivalent
- **Sensors**: RGB-D camera, LIDAR, IMU, microphones
- **Actuators**: Humanoid robot platform with manipulators
- **Connectivity**: Robust network connectivity for AI services

### Configuration Management
- **Parameter Files**: Organize parameters in YAML files
- **Environment Variables**: Use environment variables for API keys
- **Launch Files**: Create different launch configurations
- **Docker**: Containerize components for deployment

This end-to-end pipeline demonstrates the integration of all course modules into a comprehensive autonomous humanoid system with conversational AI capabilities. Each component builds upon the previous modules to create a sophisticated, safe, and functional robotic system.