---
sidebar_position: 2
---

# ROS 2 Architecture

## Overview
ROS 2's architecture is fundamentally different from ROS 1, built on the Data Distribution Service (DDS) standard. This section explores the middleware layer, Quality of Service (QoS) settings, and advanced architectural concepts that enable robust, distributed robotic systems.

## Learning Objectives
By the end of this section, students will be able to:
- Explain the DDS-based architecture of ROS 2
- Configure Quality of Service settings for different communication needs
- Implement secure and distributed ROS 2 systems
- Design advanced node architectures for complex robotic applications

## Key Concepts

### DDS (Data Distribution Service) Foundation

#### What is DDS?
- **Standard**: OMG (Object Management Group) specification for real-time data distribution
- **Implementation**: Provides discovery, reliability, and performance guarantees
- **Flexibility**: Supports multiple vendors (Fast DDS, Cyclone DDS, RTI Connext)
- **Scalability**: Handles distributed systems from single robots to multi-robot teams

#### DDS Architecture Components
- **Domain**: Logical separation of DDS systems (equivalent to ROS_DOMAIN_ID)
- **Participant**: Individual nodes in the DDS network
- **Topic**: Named data streams with type definitions
- **Publisher/Subscriber**: Communication entities for data exchange
- **DataWriter/DataReader**: Lower-level DDS entities managing data flow

### Middleware Implementations

#### Available DDS Implementations
- **Fast DDS (eProsima)**: Default in ROS 2 Humble, high performance
- **Cyclone DDS (Eclipse)**: Lightweight, efficient implementation
- **RTI Connext DDS**: Commercial solution with enterprise features
- **OpenSplice DDS**: Open-source implementation (discontinued)

#### Choosing the Right Implementation
- **Performance Requirements**: Fast DDS for high-throughput applications
- **Resource Constraints**: Cyclone DDS for embedded systems
- **Commercial Support**: RTI for enterprise deployments
- **Open Source**: Cyclone or Fast DDS for academic/research use

### Quality of Service (QoS) Policies

#### Reliability Policy
- **RELIABLE**: All messages are guaranteed to be delivered (with retries)
- **BEST_EFFORT**: Messages may be lost, but faster delivery
- **Use Cases**: RELIABLE for critical commands, BEST_EFFORT for sensor data

#### Durability Policy
- **TRANSIENT_LOCAL**: Late-joining subscribers receive previous messages
- **VOLATILE**: No historical data preservation
- **Use Cases**: TRANSIENT_LOCAL for configuration, VOLATILE for streaming data

#### History Policy
- **KEEP_LAST**: Maintain N most recent messages
- **KEEP_ALL**: Maintain all messages (use with care)
- **Depth Parameter**: Number of messages to keep in history

#### Deadline and Lifespan
- **Deadline**: Maximum time between samples
- **Lifespan**: Maximum lifetime of a sample
- **Liveliness**: How to detect if a publisher is alive

### Domain and Network Configuration

#### Domain IDs
```bash
# Set domain ID (0-232 range)
export ROS_DOMAIN_ID=42

# Multiple domains allow isolation
# Domain 0: Simulation
# Domain 1: Real robot
# Domain 2: Debug tools
```

#### Network Discovery
- **Multicast**: Automatic discovery of ROS 2 nodes
- **Static Discovery**: Pre-configured node lists for security
- **Security**: TLS encryption and authentication

## Practical Implementation

### Advanced QoS Configuration
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Configuration for critical control commands
critical_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Configuration for sensor data
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5
)

# Configuration for logging
log_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_ALL
)
```

### Domain Participant Configuration
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

class AdvancedNode(Node):
    def __init__(self):
        super().__init__('advanced_node')

        # Different publishers with different QoS for different data types
        self.critical_cmd_pub = self.create_publisher(
            String, 'critical_cmd', critical_qos)

        self.sensor_data_pub = self.create_publisher(
            LaserScan, 'sensor_data', sensor_qos)

        self.log_pub = self.create_publisher(
            String, 'log_data', log_qos)

        # Subscriber with matching QoS
        self.cmd_sub = self.create_subscription(
            String, 'cmd', self.cmd_callback, critical_qos)
```

### Multi-Domain Configuration
```python
# Example of configuring for multiple robots
# Robot 1: Domain ID 10
# Robot 2: Domain ID 11
# Control station: Domain ID 20

# For robot 1
# export ROS_DOMAIN_ID=10

# For robot 2
# export ROS_DOMAIN_ID=11

# For control station (if needs to communicate with both)
# export ROS_LOCALHOST_ONLY=0  # Allow network communication
# Can join multiple domains with careful configuration
```

## Security Architecture

### ROS 2 Security Features
- **Authentication**: Verify node identity
- **Authorization**: Control node permissions
- **Encryption**: Protect data in transit
- **Signing**: Verify message integrity

### Security Configuration Example
```python
# Security files directory structure
# ~/.ros/sros2_keys/
# ├── keystore/
# │   ├── config/
# │   │   └── governance.xml
# │   ├── ca/
# │   │   ├── cert/
# │   │   │   └── ca.cert.pem
# │   │   └── private/
# │   │       └── ca.key.pem
# │   └── identities/
# │       └── my_node/
# │           ├── cert/
# │           │   └── identity.cert.pem
# │           ├── private/
# │           │   └── identity.key.pem
# │           └── permissions/
# │               └── permissions.xml
```

## Performance Optimization

### Memory Management
- **Message Pools**: Reuse message objects to reduce allocation
- **Zero-copy**: Share memory between processes when possible
- **Serialization**: Optimize data structures for efficient serialization

### Network Optimization
- **Transport**: UDP multicast for discovery, TCP/UDP for data
- **Compression**: Compress large messages when bandwidth is limited
- **Threading**: Use appropriate executors for different workloads

### Executor Types
```python
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
import threading

# Single-threaded executor
single_executor = SingleThreadedExecutor()

# Multi-threaded executor
multi_executor = MultiThreadedExecutor(num_threads=4)

# Custom executor for specific needs
class RobotControlExecutor(MultiThreadedExecutor):
    def __init__(self):
        super().__init__(num_threads=2)  # Separate threads for safety-critical tasks
        self.safety_lock = threading.Lock()
```

## Advanced Node Architecture

### Component-Based Design
```python
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn

class RobotComponent(LifecycleNode):
    def __init__(self, name):
        super().__init__(name)
        self.declare_parameter('component_enabled', True)
        self.component_enabled = self.get_parameter('component_enabled').value

    def on_configure(self, state: LifecycleState):
        """Configure component resources"""
        self.get_logger().info(f'Configuring {self.get_name()}')
        # Initialize hardware, sensors, etc.
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        """Activate component"""
        self.get_logger().info(f'Activating {self.get_name()}')
        # Start timers, subscriptions, etc.
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        """Deactivate component"""
        self.get_logger().info(f'Deactivating {self.get_name()}')
        # Stop timers, reset state, etc.
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        """Clean up resources"""
        self.get_logger().info(f'Cleaning up {self.get_name()}')
        # Release hardware, close connections, etc.
        return TransitionCallbackReturn.SUCCESS
```

### State Machine Integration
```python
from enum import Enum
from rclpy.node import Node
from std_msgs.msg import String

class RobotState(Enum):
    IDLE = 1
    NAVIGATING = 2
    MANIPULATING = 3
    EMERGENCY_STOP = 4

class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine_node')
        self.current_state = RobotState.IDLE
        self.state_publisher = self.create_publisher(String, 'robot_state', 10)
        self.state_subscriber = self.create_subscription(
            String, 'state_commands', self.state_command_callback, 10)

        self.state_timer = self.create_timer(0.1, self.state_machine_loop)

    def state_machine_loop(self):
        """Main state machine execution"""
        state_msg = String()
        state_msg.data = self.current_state.name
        self.state_publisher.publish(state_msg)

        # Execute behavior based on current state
        if self.current_state == RobotState.NAVIGATING:
            self.execute_navigation()
        elif self.current_state == RobotState.MANIPULATING:
            self.execute_manipulation()
        # ... other states

    def state_command_callback(self, msg):
        """Handle state change commands"""
        if msg.data == 'NAVIGATE' and self.current_state == RobotState.IDLE:
            self.current_state = RobotState.NAVIGATING
        elif msg.data == 'MANIPULATE' and self.current_state == RobotState.IDLE:
            self.current_state = RobotState.MANIPULATING
        elif msg.data == 'STOP':
            self.current_state = RobotState.IDLE
```

## Distributed System Design

### Multi-Robot Communication
- **Domain Separation**: Use different domain IDs for different robot teams
- **Master Discovery**: Implement robot discovery protocols
- **Load Balancing**: Distribute computational tasks across robots
- **Fault Tolerance**: Handle robot failures gracefully

### Cloud Integration Patterns
- **Edge Processing**: Local processing for real-time tasks
- **Cloud Processing**: Heavy computation offloaded to cloud
- **Data Synchronization**: Keep local and cloud data consistent
- **Bandwidth Management**: Optimize data transfer rates

## Best Practices

### Architecture Design
- **Modularity**: Keep nodes focused on single responsibilities
- **Decoupling**: Minimize dependencies between components
- **Scalability**: Design for increasing complexity
- **Maintainability**: Use clear interfaces and documentation

### Performance Considerations
- **Real-time Requirements**: Meet timing constraints for safety-critical systems
- **Resource Management**: Efficiently use CPU, memory, and network
- **Communication Efficiency**: Optimize message sizes and frequencies
- **Error Handling**: Graceful degradation under adverse conditions

## Practical Lab: Advanced ROS 2 Architecture

### Lab Objective
Implement a distributed robotic system with multiple nodes, different QoS settings, and proper error handling.

### Implementation Steps
1. Create multiple nodes with different responsibilities
2. Configure appropriate QoS settings for each communication channel
3. Implement lifecycle nodes for proper resource management
4. Test system behavior under different network conditions

### Expected Outcome
- Working distributed system with proper architecture
- Correct use of QoS policies for different data types
- Lifecycle management for resource control
- Demonstrated understanding of advanced ROS 2 concepts

## Review Questions

1. Explain the role of DDS in ROS 2 architecture and how it differs from ROS 1's communication model.
2. When would you use RELIABLE vs BEST_EFFORT QoS policies? Provide specific examples.
3. Describe the security features available in ROS 2 and when to use them.
4. What are lifecycle nodes and why are they important for complex robotic systems?
5. How can you optimize performance in a distributed ROS 2 system?

## Next Steps
After mastering ROS 2 architecture, students should proceed to:
- URDF for robot modeling and representation
- Advanced simulation with Gazebo
- Integration with Python agents using rclpy
- Practical labs with real hardware platforms

This advanced understanding of ROS 2 architecture provides the foundation for building complex, distributed, and robust robotic systems in Physical AI applications.