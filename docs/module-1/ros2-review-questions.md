---
sidebar_position: 6
---

# ROS 2 Review Questions

## Overview
This section contains comprehensive review questions covering all aspects of ROS 2 fundamentals, architecture, and practical implementation. These questions are designed to test understanding and prepare students for advanced robotics applications.

## Module 1: The Robotic Nervous System (ROS 2)

### Fundamentals and Core Concepts

#### Question 1: ROS 2 vs ROS 1
**Difficulty**: Basic

Explain the key differences between ROS 1 and ROS 2, focusing on:
1. Communication architecture
2. Real-time capabilities
3. Multi-robot support
4. Security features

**Answer Guide**:
- ROS 1 uses a centralized master architecture, while ROS 2 uses DDS (Data Distribution Service) for decentralized communication
- ROS 2 has enhanced real-time support through DDS quality of service settings
- ROS 2 provides better multi-robot support with domain IDs
- ROS 2 includes built-in security features like authentication and encryption

#### Question 2: Core Communication Patterns
**Difficulty**: Basic

Describe the three main communication patterns in ROS 2 and provide an example use case for each:
1. Topics (Publish/Subscribe)
2. Services (Request/Response)
3. Actions (Goal-Based Communication)

**Answer Guide**:
- Topics: Continuous data streams like sensor data, robot state (e.g., laser scans, camera images)
- Services: One-time requests with immediate responses (e.g., calibration, configuration queries)
- Actions: Long-running tasks with feedback (e.g., navigation goals, manipulation tasks)

#### Question 3: Node Architecture
**Difficulty**: Intermediate

Explain the role of nodes in ROS 2 and describe how multiple nodes can coordinate to perform complex robotic tasks.

**Answer Guide**:
- Nodes are processes that perform computation and communicate via topics, services, and actions
- Nodes encapsulate specific functionality (navigation, perception, control)
- Multiple nodes coordinate through shared topics and services
- Example: Navigation stack with localization, path planning, and motion control nodes

### Architecture and Advanced Concepts

#### Question 4: Quality of Service (QoS)
**Difficulty**: Intermediate

Explain Quality of Service (QoS) policies in ROS 2 and describe when you would use different policies for:
1. Critical control commands
2. Sensor data streams
3. Logging and debugging information

**Answer Guide**:
- QoS controls how data is delivered between nodes
- Critical commands: RELIABLE + TRANSIENT_LOCAL + KEEP_LAST (depth=10)
- Sensor data: BEST_EFFORT + VOLATILE + KEEP_LAST (depth=5)
- Logging: BEST_EFFORT + TRANSIENT_LOCAL + KEEP_ALL

#### Question 5: DDS and Middleware
**Difficulty**: Advanced

Describe the role of DDS (Data Distribution Service) in ROS 2 architecture and explain how it enables distributed robotic systems.

**Answer Guide**:
- DDS provides the underlying communication middleware
- Enables discovery of nodes across networks
- Handles reliability, durability, and performance guarantees
- Supports multiple vendors (Fast DDS, Cyclone DDS, RTI Connext)
- Enables multi-robot and cloud robotics applications

#### Question 6: Domain IDs and Network Configuration
**Difficulty**: Intermediate

How do ROS_DOMAIN_ID and ROS_LOCALHOST_ONLY environment variables affect ROS 2 communication? Provide practical examples of when to use each.

**Answer Guide**:
- ROS_DOMAIN_ID (0-232): Isolates different ROS networks (e.g., Domain 0 for simulation, Domain 1 for real robot)
- ROS_LOCALHOST_ONLY: Restricts communication to local machine (0=allow network, 1=local only)
- Use cases: Multi-robot systems, separating dev/test/production, security

### URDF and Robot Modeling

#### Question 7: URDF Components
**Difficulty**: Basic

List and explain the three main components that must be defined for each link in a URDF file.

**Answer Guide**:
- Visual: How the link appears in visualization (geometry, material, color)
- Collision: How the link interacts with the environment (geometry for physics simulation)
- Inertial: Mass properties for physics simulation (mass, center of mass, moments of inertia)

#### Question 8: Joint Types and Applications
**Difficulty**: Intermediate

Describe the different joint types available in URDF and provide a specific use case for each:
1. Revolute
2. Continuous
3. Prismatic
4. Fixed

**Answer Guide**:
- Revolute: Rotational joint with limits (robot elbow with angle constraints)
- Continuous: Rotational joint without limits (wheel rotation)
- Prismatic: Linear sliding joint with limits (linear actuator)
- Fixed: No movement, rigid connection (sensor mount)

#### Question 9: Xacro Advantages
**Difficulty**: Intermediate

Explain the advantages of using Xacro over plain URDF and provide an example of how Xacro can simplify a complex robot model.

**Answer Guide**:
- Advantages: Macros, properties, mathematical expressions, reusability
- Example: Using macros to define multiple similar joints/wheels
- Simplification: Reduces code duplication, enables parameterization

### Python Integration (rclpy)

#### Question 10: rclpy vs rospy
**Difficulty**: Intermediate

Compare rclpy (ROS 2 Python client library) with rospy (ROS 1 Python client library) in terms of:
1. Underlying architecture
2. Performance characteristics
3. Multi-threading capabilities

**Answer Guide**:
- Architecture: rclpy uses rcl (ROS Client Library) as abstraction over DDS
- Performance: rclpy may have different performance due to DDS overhead
- Threading: rclpy provides better multi-threading support with executors

#### Question 11: Parameter Management
**Difficulty**: Advanced

Describe how to implement dynamic parameter reconfiguration in rclpy and explain the benefits for robotic applications.

**Answer Guide**:
- Use declare_parameter() and get_parameter() for parameter access
- Implement add_on_set_parameters_callback() for dynamic reconfiguration
- Benefits: Runtime tuning, adaptation to different environments, debugging

#### Question 12: Executor Types
**Difficulty**: Intermediate

Explain the different executor types available in rclpy and when to use each one.

**Answer Guide**:
- SingleThreadedExecutor: Simple, predictable execution (default)
- MultiThreadedExecutor: Better performance for I/O bound tasks
- Custom executors: Specialized behavior for specific requirements

### Practical Implementation

#### Question 13: Launch Files
**Difficulty**: Basic

Create a launch file that starts three nodes: a sensor publisher, a data processor, and a command executor.

**Answer Guide**:
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='sensor_publisher',
            name='sensor_publisher_node',
            parameters=[{'param_name': 'param_value'}]
        ),
        Node(
            package='my_package',
            executable='data_processor',
            name='data_processor_node'
        ),
        Node(
            package='my_package',
            executable='command_executor',
            name='command_executor_node'
        )
    ])
```

#### Question 14: Error Handling
**Difficulty**: Intermediate

Describe best practices for error handling in ROS 2 nodes and provide code examples for handling:
1. Sensor data exceptions
2. Communication failures
3. Resource management

**Answer Guide**:
- Use try-catch blocks around sensor processing
- Implement connection monitoring for publishers/subscribers
- Use context managers for resource cleanup
- Implement graceful degradation strategies

#### Question 15: Performance Optimization
**Difficulty**: Advanced

Identify and explain five techniques for optimizing ROS 2 node performance, particularly for real-time robotic applications.

**Answer Guide**:
1. Appropriate QoS settings for different data types
2. Efficient message sizes and frequencies
3. Proper threading models and executors
4. Memory management and garbage collection
5. Profiling and bottleneck identification

### Safety and Reliability

#### Question 16: Safety Systems
**Difficulty**: Advanced

Design a safety system architecture for a mobile robot using ROS 2 concepts. Include:
1. Emergency stop mechanism
2. Collision avoidance
3. System monitoring

**Answer Guide**:
- Emergency stop: Global topic with high-priority QoS, multiple activation methods
- Collision avoidance: Sensor fusion, safety state machine, velocity limiting
- System monitoring: Node health checks, resource monitoring, fault detection

#### Question 17: Fault Tolerance
**Difficulty**: Advanced

Explain how to implement fault tolerance in a ROS 2 system and describe strategies for handling node failures.

**Answer Guide**:
- Node monitoring and restart policies
- Graceful degradation when nodes fail
- Redundant sensors and systems
- Error recovery procedures
- State preservation and restoration

### Integration Scenarios

#### Question 18: Multi-Robot Systems
**Difficulty**: Advanced

Describe how to configure ROS 2 for a multi-robot system with 3 robots, including network configuration, topic organization, and coordination strategies.

**Answer Guide**:
- Use different ROS_DOMAIN_ID for each robot team
- Namespace topics with robot identifiers
- Implement coordination through shared topics or services
- Consider bandwidth and communication constraints

#### Question 19: Simulation Integration
**Difficulty**: Intermediate

Explain how ROS 2 integrates with simulation environments like Gazebo and describe the benefits of simulation-first development.

**Answer Guide**:
- ROS 2 plugins for Gazebo communication
- Sensor simulation and physics modeling
- Testing in safe, repeatable environment
- Rapid prototyping and algorithm validation

#### Question 20: AI Integration
**Difficulty**: Advanced

Describe how to integrate a Python-based AI model (e.g., PyTorch/TensorFlow) with ROS 2 for real-time robotic applications, including performance considerations.

**Answer Guide**:
- Use rclpy nodes for ROS communication
- Implement efficient data conversion between ROS and AI formats
- Consider inference timing and real-time constraints
- Use appropriate threading models for AI processing

## Self-Assessment Rubric

### Beginner Level (0-40% correct)
- Basic ROS 2 concepts understood
- Need significant improvement in practical implementation
- Requires additional study on core architecture

### Intermediate Level (41-70% correct)
- Good understanding of ROS 2 fundamentals
- Can implement basic ROS 2 applications
- Need improvement in advanced concepts

### Advanced Level (71-90% correct)
- Strong understanding of ROS 2 architecture
- Can implement complex ROS 2 systems
- Ready for advanced robotics applications

### Expert Level (91-100% correct)
- Comprehensive understanding of ROS 2
- Can design and implement sophisticated robotic systems
- Ready for research and professional development

## Practical Application Questions

### Question 21: System Design Challenge
**Difficulty**: Expert

Design a complete ROS 2 system for a warehouse robot that must:
- Navigate autonomously through aisles
- Detect and pick up packages
- Avoid dynamic obstacles (people, other robots)
- Communicate with warehouse management system

Provide:
1. System architecture diagram
2. List of required nodes and their functions
3. Topic/service/action definitions
4. QoS policy recommendations
5. Safety system design

**Expected Answer Elements**:
- Perception node (camera, LiDAR processing)
- Navigation node (path planning, obstacle avoidance)
- Manipulation node (arm control)
- Communication node (warehouse interface)
- Appropriate QoS for safety-critical vs. best-effort data
- Emergency stop and collision avoidance systems

### Question 22: Performance Optimization Challenge
**Difficulty**: Expert

Your robot application experiences high CPU usage and occasional message delays. Diagnose potential issues and provide optimization strategies for:
1. High-frequency sensor data processing
2. Multiple AI model inferences
3. Network communication overhead
4. Memory management

**Expected Answer Elements**:
- Threading models and executor optimization
- Message rate limiting and filtering
- Efficient data structures and algorithms
- Resource monitoring and profiling
- DDS configuration tuning

## Review and Preparation Tips

### Study Recommendations
1. **Practice Implementation**: Implement each concept in code
2. **Use Documentation**: Refer to official ROS 2 documentation
3. **Community Resources**: Engage with ROS community forums
4. **Hands-on Labs**: Complete all practical lab exercises
5. **Real Projects**: Apply concepts to real robotic systems

### Common Mistakes to Avoid
- Not understanding QoS policies and their impact
- Poor error handling in production systems
- Inefficient message handling and processing
- Ignoring security considerations
- Lack of proper testing and validation

### Advanced Topics for Further Study
- ROS 2 security implementation
- Real-time systems and determinism
- Multi-robot coordination algorithms
- Advanced simulation techniques
- Hardware integration and embedded systems

These review questions provide a comprehensive assessment of ROS 2 knowledge and skills, preparing students for advanced robotics applications and real-world system development.