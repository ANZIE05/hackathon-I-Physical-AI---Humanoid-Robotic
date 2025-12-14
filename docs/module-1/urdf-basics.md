---
sidebar_position: 3
---

# URDF Basics

## Overview
Unified Robot Description Format (URDF) is the standard XML-based format for representing robot models in ROS. This section covers the fundamentals of creating and working with URDF files for humanoid and other robotic platforms.

## Learning Objectives
By the end of this section, students will be able to:
- Create basic robot models using URDF
- Define robot kinematic chains and joint constraints
- Include visual and collision properties
- Use Xacro for complex robot descriptions
- Validate URDF models for correctness

## Key Concepts

### What is URDF?
- **Definition**: XML-based format for describing robot models
- **Purpose**: Define robot structure, kinematics, and dynamics
- **Components**: Links, joints, visual/collision properties
- **Integration**: Works with RViz, Gazebo, MoveIt, and other ROS tools

### URDF Structure
- **Links**: Rigid bodies of the robot (base, arms, etc.)
- **Joints**: Connections between links (revolute, prismatic, etc.)
- **Visual**: How the robot appears in visualization
- **Collision**: How the robot interacts with the environment
- **Inertial**: Mass properties for physics simulation

### Joint Types
- **Revolute**: Rotational joint with limited range
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint with limits
- **Fixed**: No movement (rigid connection)
- **Floating**: 6DOF movement (rarely used)

## Practical Implementation

### Basic URDF Structure
```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Arm link -->
  <link name="arm_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.5"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joint connecting base to arm -->
  <joint name="base_to_arm" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

### Link Definition
```xml
<link name="link_name">
  <!-- Visual properties -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Choose one geometry type -->
      <box size="1 1 1"/>
      <!-- <cylinder radius="0.1" length="1"/> -->
      <!-- <sphere radius="0.1"/> -->
      <!-- <mesh filename="package://my_robot/meshes/link.stl"/> -->
    </geometry>
    <material name="material_name">
      <color rgba="0.8 0.2 0.2 1.0"/>
    </material>
  </visual>

  <!-- Collision properties -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </collision>

  <!-- Inertial properties -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

### Joint Definition
```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Advanced URDF Concepts

### Materials and Colors
```xml
<!-- Define materials -->
<material name="blue">
  <color rgba="0 0 0.8 1"/>
</material>

<material name="red">
  <color rgba="0.8 0 0 1"/>
</material>

<material name="white">
  <color rgba="1 1 1 1"/>
</material>

<!-- Use material in link -->
<link name="colored_link">
  <visual>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="blue"/>
  </visual>
</link>
```

### Transmission Elements
```xml
<transmission name="simple_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint_name">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="motor_name">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements
```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <self_collide>false</self_collide>
</gazebo>

<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/robot_name</robotNamespace>
  </plugin>
</gazebo>
```

## Xacro for Complex Models

### Xacro Basics
Xacro (XML Macros) allows for more complex robot descriptions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_width" value="0.5" />
  <xacro:property name="base_length" value="0.8" />
  <xacro:property name="base_height" value="0.2" />

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz *joint_origin *joint_axis">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <xacro:insert_block name="joint_origin"/>
      <xacro:insert_block name="joint_axis"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.1" length="0.05"/>
        </geometry>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
      </collision>
      <inertial>
        <mass value="1"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.3 0">
    <origin xyz="0.2 0.3 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </xacro:wheel>

</robot>
```

## Humanoid Robot URDF Example

### Simplified Humanoid Structure
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_humanoid">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="torso_mass" value="10.0"/>
  <xacro:property name="head_mass" value="2.0"/>
  <xacro:property name="arm_mass" value="1.5"/>
  <xacro:property name="leg_mass" value="3.0"/>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="0.001"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${torso_mass}"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="10" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="skin">
        <color rgba="0.9 0.8 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${head_mass}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 ${M_PI/2}"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="10" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${arm_mass}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Arm -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.2 0 0.1" rpy="0 0 ${-M_PI/2}"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="10" velocity="1"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${arm_mass}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.1 0 -0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="20" velocity="1"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${leg_mass}"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Leg -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.1 0 -0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="20" velocity="1"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${leg_mass}"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.002"/>
    </inertial>
  </link>

</robot>
```

## URDF Tools and Validation

### Checking URDF Models
```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Display robot model information
urdf_to_graphiz /path/to/robot.urdf

# Launch with RViz
roslaunch urdf_tutorial display.launch model:=/path/to/robot.urdf
```

### Launch File for URDF Display
```xml
<launch>
  <!-- Load robot description parameter -->
  <param name="robot_description" command="$(find xacro)/xacro --inorder '$(find my_robot_description)/urdf/robot.xacro'" />

  <!-- Publish robot state -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />

  <!-- Joint state publisher (for visualization) -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher">
    <param name="use_gui" value="true"/>
  </node>

  <!-- Launch RViz -->
  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find my_robot_description)/config/robot.rviz" />
</launch>
```

## Best Practices

### URDF Design Guidelines
- **Mass Properties**: Accurate mass and inertia values for simulation
- **Collision Models**: Simplified but accurate collision geometry
- **Visual Models**: Detailed for visualization, separate from collision
- **Joint Limits**: Realistic limits based on physical constraints
- **Naming Conventions**: Consistent and descriptive link/joint names

### Performance Considerations
- **Mesh Complexity**: Simplify collision meshes for better performance
- **Tree Structure**: Ensure single tree structure (no loops)
- **Inertial Properties**: Properly calculated for stable simulation
- **Material Definitions**: Use standard materials when possible

## Practical Lab: Create a Simple Robot Model

### Lab Objective
Create a wheeled robot model with proper URDF structure for simulation.

### Implementation Steps
1. Create a basic differential drive robot with 4 wheels
2. Define proper visual and collision properties
3. Include material definitions and colors
4. Validate the URDF model and visualize in RViz

### Expected Outcome
- Working URDF model that displays correctly in RViz
- Proper joint structure for differential drive
- Validated mass and inertial properties
- Demonstrated understanding of URDF concepts

## Review Questions

1. What are the three main components of a URDF link?
2. Explain the difference between visual and collision properties.
3. What is the purpose of the inertial element in URDF?
4. How do you define joint limits in URDF?
5. What is Xacro and why is it useful for complex robots?

## Next Steps
After mastering URDF basics, students should proceed to:
- Advanced robot modeling techniques
- Integration with Gazebo simulation
- Creating custom meshes and visual assets
- Working with MoveIt for motion planning

This comprehensive introduction to URDF provides the foundation for creating robot models that can be used in simulation, visualization, and motion planning applications.