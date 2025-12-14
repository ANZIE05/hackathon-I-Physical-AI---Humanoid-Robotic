---
sidebar_position: 2
---

# Physics Simulation

## Overview
Physics simulation is the core component of Gazebo that enables realistic modeling of robot-environment interactions. This section delves into the physics principles, configuration, and optimization techniques for creating accurate and stable simulations.

## Learning Objectives
By the end of this section, students will be able to:
- Understand the fundamental physics principles underlying robotic simulation
- Configure physics engines for different simulation scenarios
- Optimize physics parameters for stability and performance
- Validate simulation accuracy against real-world physics
- Troubleshoot common physics-related issues

## Key Concepts

### Rigid Body Dynamics
- **Newtonian Mechanics**: Application of Newton's laws to simulate motion
- **Mass Properties**: Mass, center of mass, and moments of inertia
- **Forces and Torques**: Gravitational, frictional, and applied forces
- **Collision Response**: Impulse-based collision handling

### Collision Detection
- **Broad Phase**: Fast culling of non-colliding pairs
- **Narrow Phase**: Precise collision detection and contact point generation
- **Continuous Collision Detection**: Prevention of tunneling for fast-moving objects
- **Contact Manifold**: Set of contact points between colliding objects

### Constraint Solvers
- **Sequential Impulse**: Iterative method for solving contact constraints
- **Projected Gauss-Seidel**: Numerical method for constraint solving
- **Error Correction**: Techniques to maintain constraint satisfaction
- **Stability vs. Performance**: Trade-offs in solver configuration

## Physics Engine Configuration

### ODE (Open Dynamics Engine) Parameters
```xml
<physics type="ode">
  <!-- Time stepping -->
  <max_step_size>0.001</max_step_size>          <!-- Simulation time step -->
  <real_time_factor>1</real_time_factor>        <!-- Real-time scaling -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- Hz -->

  <!-- Solver configuration -->
  <ode>
    <solver>
      <type>quick</type>                        <!-- Solver type -->
      <iters>10</iters>                         <!-- Iterations for convergence -->
      <sor>1.3</sor>                            <!-- Successive Over-Relaxation -->
    </solver>

    <!-- Constraint parameters -->
    <constraints>
      <cfm>0.0</cfm>                            <!-- Constraint Force Mixing -->
      <erp>0.2</erp>                            <!-- Error Reduction Parameter -->
      <contact_max_correcting_vel>100</contact_max_correcting_vel>  <!-- Max correction velocity -->
      <contact_surface_layer>0.001</contact_surface_layer>          <!-- Contact surface layer -->
    </constraints>
  </ode>
</physics>
```

### Bullet Physics Configuration
```xml
<physics type="bullet">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>

  <bullet>
    <solver>
      <type>sequential_impulse</type>
      <max_iterations>10</max_iterations>
      <tau>0.1</tau>
      <damping>0.01</damping>
    </solver>

    <constraints>
      <contact_surface_layer>0.001</contact_surface_layer>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
    </constraints>
  </bullet>
</physics>
```

## Inertial Properties

### Mass and Inertia Calculation
```xml
<inertial>
  <!-- Mass of the link -->
  <mass>1.0</mass>

  <!-- Inertia matrix -->
  <inertia>
    <!-- Diagonal elements -->
    <ixx>0.1</ixx>
    <iyy>0.1</iyy>
    <izz>0.1</izz>

    <!-- Off-diagonal elements (usually zero for principal axes) -->
    <ixy>0.0</ixy>
    <ixz>0.0</ixz>
    <iyz>0.0</iyz>
  </inertia>
</inertial>
```

### Common Inertia Formulas
For standard geometric shapes:

**Box** (length l, width w, height h):
- Ixx = (1/12) * m * (w² + h²)
- Iyy = (1/12) * m * (l² + h²)
- Izz = (1/12) * m * (l² + w²)

**Cylinder** (radius r, height h):
- Ixx = Iyy = (1/12) * m * (3*r² + h²)
- Izz = (1/2) * m * r²

**Sphere** (radius r):
- Ixx = Iyy = Izz = (2/5) * m * r²

## Friction and Contact Properties

### Surface Properties
```xml
<surface>
  <!-- Contact parameters -->
  <contact>
    <ode>
      <soft_cfm>0.0</soft_cfm>
      <soft_erp>0.2</soft_erp>
      <kp>1e+13</kp>  <!-- Contact stiffness -->
      <kd>1.0</kd>    <!-- Contact damping -->
      <max_vel>100.0</max_vel>
      <min_depth>0.0</min_depth>
    </ode>
  </contact>

  <!-- Friction parameters -->
  <friction>
    <ode>
      <mu>1.0</mu>        <!-- Primary friction coefficient -->
      <mu2>1.0</mu2>      <!-- Secondary friction coefficient -->
      <fdir1>0 0 0</fdir1> <!-- Friction direction -->
      <slip1>0.0</slip1>  <!-- Primary slip coefficient -->
      <slip2>0.0</slip2>  <!-- Secondary slip coefficient -->
    </ode>
  </friction>

  <!-- Bounce parameters -->
  <bounce>
    <restitution_coefficient>0.01</restitution_coefficient>
    <threshold>100000</threshold>
  </bounce>
</surface>
```

## Collision Geometry Optimization

### Simplified Collision Meshes
```xml
<!-- For complex visual models, use simplified collision geometry -->
<link name="complex_link">
  <!-- Visual geometry (detailed) -->
  <visual name="visual">
    <geometry>
      <mesh>
        <uri>model://my_robot/meshes/complex_visual.dae</uri>
      </mesh>
    </geometry>
  </visual>

  <!-- Collision geometry (simplified) -->
  <collision name="collision">
    <geometry>
      <!-- Use simpler primitive shapes -->
      <box>
        <size>0.5 0.3 0.2</size>
      </box>
    </geometry>
  </collision>

  <!-- Multiple collision elements for complex shapes -->
  <collision name="collision_wheel_1">
    <geometry>
      <cylinder>
        <radius>0.1</radius>
        <length>0.05</length>
      </cylinder>
    </geometry>
  </collision>
</link>
```

### Compound Collision Shapes
```xml
<!-- Multiple collision elements for accurate representation -->
<link name="chassis">
  <!-- Main body collision -->
  <collision name="main_body">
    <geometry>
      <box>
        <size>0.8 0.5 0.3</size>
      </box>
    </geometry>
  </collision>

  <!-- Wheel collision (multiple) -->
  <collision name="front_left_wheel">
    <geometry>
      <cylinder>
        <radius>0.1</radius>
        <length>0.05</length>
      </cylinder>
    </geometry>
    <pose>0.3 0.2 -0.1 0 0 0</pose>
  </collision>

  <collision name="front_right_wheel">
    <geometry>
      <cylinder>
        <radius>0.1</radius>
        <length>0.05</length>
      </cylinder>
    </geometry>
    <pose>0.3 -0.2 -0.1 0 0 0</pose>
  </collision>
</link>
```

## Advanced Physics Features

### Joint Dynamics
```xml
<joint name="motor_joint" type="revolute">
  <parent>base_link</parent>
  <child>wheel_link</child>

  <!-- Joint limits -->
  <limit>
    <lower>-1e+16</lower>  <!-- Lower limit (radians) -->
    <upper>1e+16</upper>   <!-- Upper limit (radians) -->
    <effort>100</effort>   <!-- Maximum effort (N-m) -->
    <velocity>10</velocity> <!-- Maximum velocity (rad/s) -->
  </limit>

  <!-- Dynamics parameters -->
  <dynamics>
    <damping>0.1</damping>      <!-- Viscous damping -->
    <friction>0.0</friction>    <!-- Coulomb friction -->
    <spring_reference>0</spring_reference>  <!-- Spring reference angle -->
    <spring_stiffness>0</spring_stiffness>  <!-- Spring stiffness -->
  </dynamics>
</joint>
```

### Custom Physics Plugins
```xml
<!-- Plugin for custom physics behavior -->
<plugin name="custom_physics" filename="libCustomPhysicsPlugin.so">
  <link_name>wheel_link</link_name>
  <custom_parameter>value</custom_parameter>
  <update_rate>100</update_rate>
</plugin>
```

## Performance Optimization

### Physics Parameter Tuning
```xml
<physics type="ode">
  <!-- Balance accuracy and performance -->
  <max_step_size>0.001</max_step_size>    <!-- Smaller = more accurate, slower -->
  <real_time_factor>1</real_time_factor>  <!-- Target real-time factor -->

  <!-- Solver iterations: Higher = more stable, slower -->
  <ode>
    <solver>
      <iters>20</iters>        <!-- Start with 10-20 -->
      <sor>1.3</sor>           <!-- Typically 1.0-1.3 -->
    </solver>

    <!-- Error correction: Lower = more stable, more error -->
    <constraints>
      <erp>0.2</erp>           <!-- Error reduction (0.1-0.8) -->
      <cfm>0.0</cfm>           <!-- Constraint force mixing -->
    </constraints>
  </ode>
</physics>
```

### Optimization Strategies
- **Adaptive Time Stepping**: Adjust step size based on system dynamics
- **Multi-rate Simulation**: Different update rates for different systems
- **Spatial Partitioning**: Efficient collision detection for large worlds
- **Level of Detail**: Simplified models at distance

## Realism Considerations

### Real-World Physics Effects
```xml
<!-- Gravity vector -->
<gravity>0 0 -9.8</gravity>

<!-- Environmental effects -->
<world>
  <!-- Wind simulation -->
  <wind>
    <linear_velocity>0.1 0 0</linear_velocity>
    <force>0.01 0 0</force>
  </wind>

  <!-- Atmospheric properties -->
  <atmosphere type="adiabatic">
    <pressure>101325</pressure>
    <temperature>288.15</temperature>
    <temperature_gradient>-0.0065</temperature_gradient>
  </atmosphere>
</world>
```

### Sensor Physics Integration
```xml
<!-- Camera with realistic physics effects -->
<sensor name="camera" type="camera">
  <camera name="head">
    <!-- Include motion blur, lens distortion effects -->
    <lens>
      <type>stereographic</type>
      <scale_to_hfov>true</scale_to_hfov>
      <cutoff_angle>1.5707</cutoff_angle>
      <env_texture_size>512</env_texture_size>
    </lens>

    <!-- Distortion parameters -->
    <distortion>
      <k1>0.0</k1>
      <k2>0.0</k2>
      <k3>0.0</k3>
      <p1>0.0</p1>
      <p2>0.0</p2>
      <center>0.5 0.5</center>
    </distortion>
  </camera>
</sensor>
```

## Validation and Calibration

### Simulation-to-Reality Transfer
- **Parameter Identification**: Determine real-world parameters for simulation
- **System Identification**: Match dynamic behavior between sim and reality
- **Sensor Calibration**: Align sensor models with real sensors
- **Validation Experiments**: Test in both simulation and reality

### Performance Metrics
- **Stability**: Absence of jittering or unrealistic oscillations
- **Accuracy**: Closeness to real-world behavior
- **Performance**: Simulation speed relative to real-time
- **Robustness**: Ability to handle edge cases and disturbances

## Troubleshooting Physics Issues

### Common Problems and Solutions

#### Jittering Objects
**Symptoms**: Objects vibrating or shaking
**Solutions**:
- Increase solver iterations
- Adjust ERP and CFM values
- Check mass properties (avoid very small masses)
- Reduce time step size

#### Objects Falling Through Surfaces
**Symptoms**: Objects passing through collision geometry
**Solutions**:
- Check collision geometry definition
- Increase contact surface layer
- Verify mass properties
- Reduce time step size

#### Unstable Joint Behavior
**Symptoms**: Joints moving erratically
**Solutions**:
- Check joint limits and dynamics
- Verify mass distribution
- Adjust solver parameters
- Consider joint stiffness/damping

#### Slow Simulation
**Symptoms**: Simulation running slower than real-time
**Solutions**:
- Optimize collision meshes
- Reduce solver iterations
- Simplify world geometry
- Adjust real-time factor

## Practical Lab: Physics Parameter Optimization

### Lab Objective
Optimize physics parameters for a simple mobile robot to achieve stable and realistic behavior while maintaining good performance.

### Implementation Steps
1. Create a simple wheeled robot model
2. Test with default physics parameters
3. Identify stability/performance issues
4. Systematically adjust parameters
5. Validate results against requirements

### Expected Outcome
- Optimized physics parameters for the robot
- Understanding of parameter trade-offs
- Validated simulation behavior
- Documented optimization process

## Review Questions

1. Explain the relationship between time step size, solver iterations, and simulation stability.
2. How do you calculate the inertia matrix for a complex-shaped robot link?
3. What are the key differences between ODE and Bullet physics engines for robotics simulation?
4. Describe the process for calibrating simulation parameters to match real-world behavior.
5. How do friction coefficients affect robot mobility in simulation?

## Next Steps
After mastering physics simulation, students should proceed to:
- Sensor simulation and integration
- Advanced Gazebo workflows
- Simulation validation techniques
- Integration with perception and control systems

This comprehensive understanding of physics simulation enables the creation of realistic and stable robotic simulations essential for Physical AI and Humanoid Robotics development.