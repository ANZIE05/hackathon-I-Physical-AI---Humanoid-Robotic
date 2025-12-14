---
sidebar_position: 6
---

# ROS 2 Review Questions

{: .review-questions}
## Module 1: The Robotic Nervous System (ROS 2)

Test your understanding of ROS 2 concepts with these review questions.

### Basic Concepts

1. What is the primary purpose of ROS 2 in robotic systems?
   - a) To provide a real-time operating system for robots
   - b) To provide middleware for communication between robotic software components
   - c) To serve as a programming language for robotics
   - d) To act as a simulation environment for robots

2. Which of the following is NOT a communication pattern in ROS 2?
   - a) Publisher-Subscriber
   - b) Service-Client
   - c) Action-Client
   - d) Database-Query

3. What is the difference between a service and an action in ROS 2?
   - a) There is no difference; they are the same thing
   - b) Services are for long-running tasks with feedback, actions are for quick requests
   - c) Actions are for long-running tasks with feedback and goals, services are for synchronous request-response
   - d) Services use XML, actions use JSON

### Architecture and Components

4. What does DDS stand for in the context of ROS 2?
   - a) Distributed Data System
   - b) Data Distribution Service
   - c) Dynamic Discovery Service
   - d) Distributed Development System

5. Which Python library is used for creating ROS 2 nodes in Python?
   - a) rospy
   - b) rclpy
   - c) ros2py
   - d) pyros

6. What is a Quality of Service (QoS) profile in ROS 2?
   - a) A measure of network performance
   - b) A set of policies that describe how messages are delivered
   - c) A configuration for robot hardware
   - d) A security protocol for robot communication

### URDF and Robot Description

7. What does URDF stand for?
   - a) Unified Robot Design Format
   - b) Universal Robot Description Format
   - c) Unified Robot Description File
   - d) Universal Robot Development Framework

8. Which joint type allows for unlimited rotation around a single axis?
   - a) Revolute
   - b) Prismatic
   - c) Fixed
   - d) Continuous

9. What are the three main elements that make up a robot in URDF?
   - a) Sensors, controllers, and actuators
   - b) Links, joints, and transmissions
   - c) Visual, collision, and inertial properties
   - d) Base, arms, and wheels

### Practical Application

10. Which command is used to build ROS 2 packages?
    - a) catkin_make
    - b) colcon build
    - c) make build
    - d) ros2 build

11. What is the purpose of the robot_state_publisher in ROS 2?
    - a) To publish sensor data from the robot
    - b) To publish the state of the robot's joints
    - c) To control the robot's movement
    - d) To save the robot's configuration

12. Which tool is commonly used to visualize ROS 2 robot models and sensor data?
    - a) Gazebo
    - b) RViz2
    - c) RQt
    - d) All of the above

### Advanced Concepts

13. What is the purpose of callback groups in rclpy?
    - a) To group related callback functions for better organization
    - b) To control the threading model of callbacks
    - c) To share data between different callback functions
    - d) To measure the performance of callback functions

14. Which of the following is true about ROS 2 domains?
    - a) All nodes must be on the same domain to communicate
    - b) Nodes on different domains can communicate without configuration
    - c) Domain IDs allow multiple isolated ROS 2 networks on the same infrastructure
    - d) Domain IDs are only used in simulation environments

15. What is the main advantage of using launch files in ROS 2?
    - a) They make the code run faster
    - b) They allow multiple nodes to be started with a single command
    - c) They reduce the memory usage of nodes
    - d) They provide better security for robot systems

## Answers

1. b) To provide middleware for communication between robotic software components
2. d) Database-Query
3. c) Actions are for long-running tasks with feedback and goals, services are for synchronous request-response
4. b) Data Distribution Service
5. b) rclpy
6. b) A set of policies that describe how messages are delivered
7. b) Universal Robot Description Format
8. d) Continuous
9. b) Links, joints, and transmissions
10. b) colcon build
11. b) To publish the state of the robot's joints
12. d) All of the above
13. b) To control the threading model of callbacks
14. c) Domain IDs allow multiple isolated ROS 2 networks on the same infrastructure
15. b) They allow multiple nodes to be started with a single command

## Self-Assessment

Rate your understanding of each topic:
- ROS 2 basic concepts: ___/10
- Communication patterns: ___/10
- URDF and robot description: ___/10
- Practical application: ___/10
- Advanced concepts: ___/10

If you scored below 7/10 on any section, consider reviewing the corresponding material before proceeding to the next module.