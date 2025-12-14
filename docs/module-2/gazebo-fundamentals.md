---
sidebar_position: 1
---

# Gazebo Fundamentals

## Overview
Gazebo is a physics-based simulation environment that plays a crucial role in Physical AI development. It provides realistic simulation of robots, environments, and sensors, enabling safe and cost-effective development and testing of robotic systems before deployment on real hardware.

## Learning Objectives
By the end of this section, students will be able to:
- Understand the physics simulation principles in Gazebo
- Create and customize simulation environments using SDF
- Implement robot models in Gazebo with proper physics properties
- Connect simulated robots to ROS 2 control systems
- Configure and use various sensor simulations
- Validate simulation accuracy against real-world behavior

## Key Concepts

### What is Gazebo?
- **Physics Simulation**: Accurate modeling of real-world physics including collision detection, contact forces, and rigid body dynamics
- **3D Visualization**: Realistic rendering of environments and robots
- **Sensor Simulation**: Accurate simulation of cameras, LiDAR, IMU, GPS, and other sensors
- **ROS Integration**: Seamless integration with ROS and ROS 2 for control and communication
- **Extensibility**: Plugin system for custom sensors, controllers, and world elements

### Physics Engines
Gazebo supports multiple physics engines, each with different strengths:
- **ODE (Open Dynamics Engine)**: Stable and widely used, good for most applications
- **Bullet**: Fast and accurate, good for complex interactions
- **DART (Dynamic Animation and Robotics Toolkit)**: Advanced kinematic and dynamic analysis
- **Simbody**: High-fidelity simulation for biomechanics and complex systems

### Simulation Pipeline
```
World Definition (SDF) → Physics Engine → Sensor Simulation → Visualization → ROS Interface
```

### SDF (Simulation Description Format)
- **XML-based**: Human-readable format for describing simulation elements
- **Hierarchical**: Organized structure for worlds, models, links, and joints
- **Extensible**: Custom elements and plugins can be added
- **Standard**: Industry standard for robot and environment description

## Practical Implementation

### Basic Gazebo World File
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sky -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Simple box object -->
    <model name="simple_box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
            <specular>0.8 0.2 0.2 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Robot model -->
    <include>
      <uri>model://my_robot</uri>
      <pose>2 0 0 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Robot Model with Sensors
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="sensor_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.2</iyy>
          <iyz>0</iyz>
          <izz>0.2</izz>
        </inertia>
      </inertial>

      <!-- Collision geometry -->
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </collision>

      <!-- Visual geometry -->
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.7 1</ambient>
          <diffuse>0.5 0.5 0.7 1</diffuse>
        </material>
      </visual>

      <!-- Camera sensor -->
      <sensor name="camera" type="camera">
        <pose>0.2 0 0 0 0 0</pose>
        <camera name="head">
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>10</far>
          </clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>
      </sensor>

      <!-- LiDAR sensor -->
      <sensor name="laser" type="ray">
        <pose>0.2 0 0.1 0 0 0</pose>
        <ray>
          <scan>
            <horizontal>
              <samples>640</samples>
              <resolution>1</resolution>
              <min_angle>-1.570796</min_angle>
              <max_angle>1.570796</max_angle>
            </horizontal>
          </scan>
          <range>
            <min>0.1</min>
            <max>10</max>
            <resolution>0.01</resolution>
          </range>
        </ray>
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
      </sensor>
    </link>

    <!-- Left wheel -->
    <joint name="left_wheel_hinge" type="revolute">
      <parent>chassis</parent>
      <child>left_wheel</child>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
      <limit>
        <effort>100</effort>
        <velocity>100</velocity>
      </limit>
    </joint>

    <link name="left_wheel">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.02</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
```

## Physics Simulation Configuration

### Physics Parameters
```xml
<physics type="ode">
  <!-- Time step settings -->
  <max_step_size>0.001</max_step_size>  <!-- Simulation time step (seconds) -->
  <real_time_factor>1</real_time_factor>  <!-- Real-time vs simulation time ratio -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- Hz -->

  <!-- Solver settings -->
  <ode>
    <solver>
      <type>quick</type>  <!-- Type of solver -->
      <iters>10</iters>   <!-- Number of iterations -->
      <sor>1.3</sor>      <!-- Successive over-relaxation parameter -->
    </solver>

    <!-- Constraints settings -->
    <constraints>
      <cfm>0.0</cfm>      <!-- Constraint force mixing -->
      <erp>0.2</erp>      <!-- Error reduction parameter -->
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Material Properties
```xml
<material>
  <ambient>0.3 0.3 0.3 1</ambient>    <!-- Ambient light reflection -->
  <diffuse>0.7 0.7 0.7 1</diffuse>    <!-- Diffuse light reflection -->
  <specular>0.9 0.9 0.9 1</specular>  <!-- Specular light reflection -->
  <emissive>0 0 0 1</emissive>        <!-- Emissive color -->
</material>
```

## Sensor Simulation

### Camera Configuration
```xml
<sensor name="camera" type="camera">
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>  <!-- Field of view in radians -->
    <image>
      <width>640</width>                    <!-- Image width -->
      <height>480</height>                  <!-- Image height -->
      <format>R8G8B8</format>               <!-- Pixel format -->
    </image>
    <clip>
      <near>0.1</near>                      <!-- Near clipping plane -->
      <far>10</far>                         <!-- Far clipping plane -->
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### LiDAR Configuration
```xml
<sensor name="laser" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>640</samples>              <!-- Number of samples -->
        <resolution>1</resolution>           <!-- Resolution -->
        <min_angle>-1.570796</min_angle>    <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>     <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>                       <!-- Minimum range -->
      <max>10</max>                        <!-- Maximum range -->
      <resolution>0.01</resolution>         <!-- Range resolution -->
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

## ROS 2 Integration

### Gazebo ROS Packages
```xml
<!-- In your robot model SDF -->
<plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
  <ros>
    <namespace>robot</namespace>
    <argument>odom_frame_id:odom</argument>
    <argument>base_frame_id:base_link</argument>
  </ros>
  <left_joint>left_wheel_joint</left_joint>
  <right_joint>right_wheel_joint</right_joint>
  <wheel_separation>0.3</wheel_separation>
  <wheel_diameter>0.15</wheel_diameter>
  <max_wheel_torque>20</max_wheel_torque>
  <max_wheel_acceleration>1.0</max_wheel_acceleration>
  <command_topic>cmd_vel</command_topic>
  <odometry_topic>odom</odometry_topic>
  <odometry_frame>odom</odometry_frame>
  <robot_base_frame>base_link</robot_base_frame>
</plugin>
```

### Launching Gazebo with ROS 2
```python
# launch/gazebo_simulation.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    world_path = PathJoinSubstitution([
        FindPackageShare('my_robot_gazebo'),
        'worlds',
        'my_world.sdf'
    ])

    # Launch Gazebo with world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_path,
            'verbose': 'true'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': open('/path/to/robot.urdf').read()
        }]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher
    ])
```

## World Building and Environment Design

### Creating Custom Worlds
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="indoor_world">
    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
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

    <!-- Building structure -->
    <model name="wall_1">
      <pose>-5 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 2</size>
            </box>
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

    <!-- Furniture -->
    <model name="table">
      <pose>2 2 0.4 0 0 0</pose>
      <link name="base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>20</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Objects to interact with -->
    <model name="cylinder_object">
      <pose>3 3 0.1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.1</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.1</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.002</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.002</iyy>
            <iyz>0</iyz>
            <izz>0.004</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Best Practices

### Performance Optimization
- **Collision Meshes**: Use simplified meshes for collision detection
- **Update Rates**: Set appropriate update rates for different sensors
- **Physics Parameters**: Tune parameters for stability and performance
- **Visual Elements**: Disable visualization for headless simulations

### Accuracy Considerations
- **Inertial Properties**: Use realistic mass and inertia values
- **Friction Coefficients**: Set appropriate surface properties
- **Sensor Noise**: Add realistic noise models to sensor data
- **Time Synchronization**: Ensure proper timing between simulation and real-time

### Model Validation
- **Physical Plausibility**: Verify models behave physically correctly
- **Sensor Accuracy**: Validate sensor output against real sensors
- **Control Response**: Test control systems in simulation vs. reality
- **Edge Cases**: Test in challenging scenarios

## Troubleshooting Common Issues

### Physics Instability
- **Symptoms**: Objects jittering, unrealistic movements
- **Solutions**: Adjust time step, increase solver iterations, tune parameters

### Sensor Issues
- **Symptoms**: No sensor data, unrealistic readings
- **Solutions**: Check sensor configuration, verify plugin loading, adjust parameters

### Performance Problems
- **Symptoms**: Slow simulation, high CPU usage
- **Solutions**: Optimize meshes, reduce update rates, adjust physics parameters

## Practical Lab: Basic Gazebo Simulation

### Lab Objective
Create a simple differential drive robot model in Gazebo with basic sensors and ROS 2 integration.

### Implementation Steps
1. Create a URDF model of a differential drive robot
2. Create an SDF world file with simple environment
3. Integrate with ROS 2 for control and sensing
4. Test basic navigation in the simulated environment

### Expected Outcome
- Working robot model in Gazebo
- Proper ROS 2 integration
- Basic control and sensing capabilities
- Demonstrated understanding of simulation concepts

## Review Questions

1. What are the key differences between ODE, Bullet, and DART physics engines in Gazebo?
2. Explain the purpose of SDF and how it differs from URDF.
3. How do you configure a camera sensor in Gazebo and what parameters are important?
4. What are the key physics parameters that affect simulation stability?
5. How do you integrate a Gazebo simulation with ROS 2?

## Next Steps
After mastering Gazebo fundamentals, students should proceed to:
- Advanced physics simulation concepts
- Sensor simulation and calibration
- Gazebo workflows and best practices
- Integration with NVIDIA Isaac tools

This comprehensive introduction to Gazebo provides the foundation for creating realistic simulation environments essential for Physical AI and Humanoid Robotics development.