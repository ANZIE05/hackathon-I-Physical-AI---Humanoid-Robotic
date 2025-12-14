---
sidebar_position: 3
---

# URDF Basics

This section covers URDF (Unified Robot Description Format), which is essential for describing robot models in ROS 2. URDF allows you to define the physical and visual properties of robots, including links, joints, and other components.

## Learning Objectives

- Understand the structure and components of URDF files
- Create basic robot models using URDF
- Define links, joints, and their properties
- Visualize URDF models in RViz2

## Key Concepts

URDF (Unified Robot Description Format) is an XML format for representing a robot model. It defines the kinematic and dynamic properties of a robot, including its physical structure, visual appearance, and collision properties.

### Links

Links represent rigid bodies in a robot. Each link has properties such as:
- Mass and inertia
- Visual representation (shape, color, material)
- Collision properties
- Name and unique identification

### Joints

Joints define the relationship between links, specifying how they can move relative to each other. Joint types include:
- **Fixed**: No movement between links
- **Revolute**: Single-axis rotation with limits
- **Continuous**: Single-axis rotation without limits
- **Prismatic**: Single-axis translation with limits
- **Floating**: Six degrees of freedom
- **Planar**: Motion constrained to a plane

### Transmissions

Transmissions define how actuators (motors) connect to joints, specifying the mechanical relationship between them.

## URDF Structure

A typical URDF file includes:

```xml
<?xml version="1.0"?>
<robot name="robot_name">
  <!-- Define links -->
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

  <!-- Define joints -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Practical Examples

[Practical examples would be included here in the full textbook]

## Review Questions

[Review questions would be included here in the full textbook]

## Next Steps

Continue to the next section to learn about rclpy integration.