---
sidebar_position: 3
---

# Sensor Simulation

## Overview
Sensor simulation is a critical component of realistic robotic simulation, providing synthetic sensor data that closely matches real-world sensors. This section covers the simulation of various sensor types, configuration options, and integration with ROS 2 for Physical AI applications.

## Learning Objectives
By the end of this section, students will be able to:
- Configure and simulate various sensor types in Gazebo
- Understand the physics behind sensor simulation
- Calibrate simulated sensors to match real-world characteristics
- Integrate sensor data with ROS 2 communication
- Validate sensor simulation accuracy and performance

## Key Concepts

### Sensor Physics and Modeling
- **Ray Tracing**: Simulation of light/radio wave propagation for cameras and LiDAR
- **Noise Modeling**: Addition of realistic noise to sensor data
- **Distortion**: Modeling of sensor-specific distortions (lens, range, etc.)
- **Temporal Effects**: Latency, update rates, and synchronization

### Sensor Categories
- **Vision Sensors**: Cameras, depth cameras, stereo cameras
- **Range Sensors**: LiDAR, sonar, infrared, ToF sensors
- **Inertial Sensors**: IMU, gyroscope, accelerometer
- **Other Sensors**: GPS, magnetometer, force/torque, contact sensors

### Realism vs. Performance Trade-offs
- **Accuracy**: How closely simulation matches real sensor behavior
- **Performance**: Computational cost of sensor simulation
- **Calibration**: Parameters to match real sensor characteristics
- **Validation**: Testing against real sensor data

## Vision Sensor Simulation

### Camera Configuration
```xml
<sensor name="camera" type="camera">
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <camera name="head_camera">
    <!-- Camera intrinsic parameters -->
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>    <!-- Near clipping plane -->
      <far>100</far>      <!-- Far clipping plane -->
    </clip>

    <!-- Lens distortion -->
    <distortion>
      <k1>0.0</k1>       <!-- Radial distortion coefficient -->
      <k2>0.0</k2>
      <k3>0.0</k3>
      <p1>0.0</p1>       <!-- Tangential distortion coefficient -->
      <p2>0.0</p2>
      <center>0.5 0.5</center>  <!-- Principal point (normalized) -->
    </distortion>
  </camera>

  <!-- Noise model -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.007</stddev>
  </noise>

  <visualize>true</visualize>
  <topic>camera/image_raw</topic>
</sensor>
```

### Depth Camera Configuration
```xml
<sensor name="depth_camera" type="depth">
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <camera name="depth_head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>320</width>
      <height>240</height>
      <format>R16G16B16</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>

  <!-- Noise for depth measurements -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>  <!-- 1cm standard deviation -->
  </noise>

  <visualize>true</visualize>
  <!-- Multiple topics for different data types -->
  <topic>camera/depth/image_raw</topic>
  <point_cloud_topic>camera/depth/points</point_cloud_topic>
  <camera_info_topic>camera/camera_info</camera_info_topic>
</sensor>
```

### Stereo Camera Configuration
```xml
<sensor name="stereo_camera" type="multicamera">
  <always_on>1</always_on>
  <update_rate>30</update_rate>

  <!-- Left camera -->
  <camera name="left">
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

  <!-- Right camera (offset from left) -->
  <camera name="right">
    <pose>0.2 0 0 0 0 0</pose>  <!-- Baseline: 20cm -->
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

  <visualize>true</visualize>
  <topic>stereo_camera/left/image_raw</topic>
</sensor>
```

## Range Sensor Simulation

### LiDAR Configuration
```xml
<sensor name="3d_lidar" type="ray">
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <ray>
    <!-- Horizontal scan pattern -->
    <scan>
      <horizontal>
        <samples>640</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -180 degrees -->
        <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
      </horizontal>
      <!-- Vertical scan pattern (for 3D LiDAR) -->
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.2618</min_angle>  <!-- -15 degrees -->
        <max_angle>0.2618</max_angle>   <!-- 15 degrees -->
      </vertical>
    </scan>

    <!-- Range properties -->
    <range>
      <min>0.1</min>
      <max>30</max>
      <resolution>0.01</resolution>
    </range>
  </ray>

  <!-- Noise model for range measurements -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>  <!-- 1cm standard deviation -->
  </noise>

  <visualize>true</visualize>
  <topic>laser_scan</topic>
</sensor>
```

### 2D LiDAR Configuration
```xml
<sensor name="2d_lidar" type="ray">
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>  <!-- 0.5 degree resolution -->
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -180 degrees -->
        <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>20</max>
      <resolution>0.01</resolution>
    </range>
  </ray>

  <!-- Add intensity information -->
  <always_on>1</always_on>
  <visualize>true</visualize>
  <topic>scan</topic>
</sensor>
```

### Sonar/IR Sensor Configuration
```xml
<sensor name="sonar_sensor" type="ray">
  <always_on>1</always_on>
  <update_rate>20</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>1</samples>
        <resolution>1</resolution>
        <min_angle>-0.1745</min_angle>  <!-- -10 degrees -->
        <max_angle>0.1745</max_angle>   <!-- 10 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.05</min>    <!-- 5cm minimum -->
      <max>4</max>       <!-- 4m maximum -->
      <resolution>0.01</resolution>
    </range>
  </ray>

  <!-- Higher noise for sonar sensors -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.05</stddev>  <!-- 5cm standard deviation -->
  </noise>

  <visualize>false</visualize>
  <topic>sonar_range</topic>
</sensor>
```

## Inertial Sensor Simulation

### IMU Configuration
```xml
<sensor name="imu_sensor" type="imu">
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <imu>
    <!-- Noise parameters for each measurement type -->
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>  <!-- 0.01 rad/s standard deviation -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </z>
    </angular_velocity>

    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>  <!-- 0.1 m/s² standard deviation -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>

  <visualize>false</visualize>
  <topic>imu/data</topic>
</sensor>
```

### GPS Sensor Configuration
```xml
<sensor name="gps_sensor" type="gps">
  <always_on>1</always_on>
  <update_rate>1</update_rate>
  <gps>
    <!-- Noise characteristics -->
    <noise>
      <position>
        <horizontal>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2.0</stddev>  <!-- 2m horizontal accuracy -->
          </noise>
        </horizontal>
        <vertical>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>4.0</stddev>  <!-- 4m vertical accuracy -->
          </noise>
        </vertical>
      </position>
    </noise>
  </gps>

  <visualize>false</visualize>
  <topic>gps/fix</topic>
</sensor>
```

## Advanced Sensor Features

### Custom Sensor Plugins
```xml
<!-- Custom sensor for specialized applications -->
<sensor name="custom_sensor" type="custom_type">
  <plugin name="custom_sensor_plugin" filename="libCustomSensorPlugin.so">
    <sensor_type>force_torque</sensor_type>
    <update_rate>100</update_rate>
    <topic>custom_sensor/data</topic>
    <noise_stddev>0.01</noise_stddev>
  </plugin>
</sensor>
```

### Sensor Fusion Simulation
```xml
<!-- Simulate fused sensor data -->
<sensor name="sensor_fusion" type="sensor_fusion">
  <always_on>1</always_on>
  <update_rate>50</update_rate>

  <!-- Simulated fusion of multiple inputs -->
  <fusion>
    <input_topic>imu/data</input_topic>
    <input_topic>gps/fix</input_topic>
    <input_topic>odometry/filtered</input_topic>
  </fusion>

  <topic>state_estimator/pose</topic>
</sensor>
```

## ROS 2 Integration

### Sensor Message Types
```python
# Example of processing sensor data in ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, NavSatFix
from cv_bridge import CvBridge
import numpy as np

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        self.bridge = CvBridge()

        # Subscribers for different sensor types
        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, 'gps/fix', self.gps_callback, 10)

    def camera_callback(self, msg):
        """Process camera image data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Process image (e.g., object detection, feature extraction)
            self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f'Camera processing error: {e}')

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        # Convert to numpy array for processing
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges
        valid_ranges = ranges[(ranges > msg.range_min) & (ranges < msg.range_max)]

        # Process scan data (e.g., obstacle detection, mapping)
        obstacles = self.detect_obstacles(valid_ranges, msg.angle_min, msg.angle_increment)

    def imu_callback(self, msg):
        """Process IMU data"""
        # Extract orientation, angular velocity, linear acceleration
        orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        angular_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        linear_acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        # Process IMU data (e.g., attitude estimation, motion detection)
        self.process_imu(orientation, angular_vel, linear_acc)

    def gps_callback(self, msg):
        """Process GPS data"""
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude

        # Process GPS data (e.g., localization, navigation)
        self.process_gps(latitude, longitude, altitude)
```

### Sensor Calibration
```python
# Calibration node for sensor parameters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float64

class SensorCalibrator(Node):
    def __init__(self):
        super().__init__('sensor_calibrator')

        # Camera calibration publisher
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        # Timer for publishing calibration data
        self.calibration_timer = self.create_timer(1.0, self.publish_calibration)

        # Calibration parameters
        self.camera_matrix = [
            525.0, 0.0, 319.5,  # fx, 0, cx
            0.0, 525.0, 239.5,  # 0, fy, cy
            0.0, 0.0, 1.0       # 0, 0, 1
        ]

        self.distortion_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]  # k1, k2, p1, p2, k3

    def publish_calibration(self):
        """Publish camera calibration information"""
        camera_info = CameraInfo()
        camera_info.header.stamp = self.get_clock().now().to_msg()
        camera_info.header.frame_id = 'camera_link'

        camera_info.height = 480
        camera_info.width = 640
        camera_info.distortion_model = 'plumb_bob'
        camera_info.d = self.distortion_coeffs
        camera_info.k = self.camera_matrix
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.p = self.camera_matrix + [0.0, 0.0, 0.0, 0.0]  # 4x4 projection matrix

        self.camera_info_pub.publish(camera_info)
```

## Noise Modeling and Realism

### Noise Types and Parameters
```xml
<!-- Different noise models for various sensors -->
<sensor name="realistic_camera" type="camera">
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

  <!-- Realistic noise model -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>
    <!-- Additional parameters for more realistic noise -->
    <bias_mean>0.001</bias_mean>
    <bias_stddev>0.0005</bias_stddev>
  </noise>

  <visualize>true</visualize>
  <topic>camera/image_raw</topic>
</sensor>
```

### Environmental Effects
```xml
<!-- Simulate environmental effects on sensors -->
<sensor name="weather_affected_sensor" type="camera">
  <camera name="weather_camera">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>50</far>  <!-- Reduced visibility in bad weather -->
    </clip>
  </camera>

  <!-- Additional weather effects -->
  <always_on>1</always_on>
  <visualize>true</visualize>
  <topic>weather_camera/image_raw</topic>
</sensor>
```

## Performance Optimization

### Sensor Update Rates
```xml
<!-- Optimize update rates for different sensor types -->
<sdf version="1.7">
  <model name="sensor_optimized_robot">
    <!-- High-rate sensors (100Hz) -->
    <sensor name="imu" type="imu">
      <update_rate>100</update_rate>
      <!-- ... other IMU config -->
    </sensor>

    <!-- Medium-rate sensors (30Hz) -->
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <!-- ... other camera config -->
    </sensor>

    <!-- Low-rate sensors (10Hz) -->
    <sensor name="gps" type="gps">
      <update_rate>10</update_rate>
      <!-- ... other GPS config -->
    </sensor>
  </model>
</sdf>
```

### Sensor Filtering
```xml
<!-- Simulate hardware-level filtering -->
<sensor name="filtered_lidar" type="ray">
  <ray>
    <!-- Reduce effective resolution to simulate hardware filtering -->
    <scan>
      <horizontal>
        <samples>360</samples>  <!-- Reduced from 720 for 1-degree resolution -->
        <resolution>2</resolution>  <!-- Effectively 2-degree resolution -->
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>20</max>
      <resolution>0.02</resolution>  <!-- Reduced precision -->
    </range>
  </ray>

  <always_on>1</always_on>
  <visualize>false</visualize>
  <topic>filtered_scan</topic>
</sensor>
```

## Validation and Calibration

### Sensor Model Validation
- **Accuracy Testing**: Compare simulated vs. real sensor data
- **Noise Characterization**: Validate noise models match real sensors
- **Timing Analysis**: Ensure proper update rates and latencies
- **Edge Case Testing**: Test in challenging conditions

### Calibration Procedures
1. **Intrinsic Calibration**: Internal sensor parameters (focal length, distortion)
2. **Extrinsic Calibration**: Sensor position/orientation relative to robot
3. **Temporal Calibration**: Synchronization between different sensors
4. **Environmental Calibration**: Adaptation to different conditions

## Troubleshooting Common Issues

### Sensor Data Problems
- **No Data**: Check sensor plugin loading, topic names, and permissions
- **Incorrect Data**: Verify sensor configuration, coordinate frames, and units
- **Timing Issues**: Check update rates, system load, and real-time performance
- **Noise Problems**: Validate noise parameters and random number generation

### Performance Issues
- **Slow Simulation**: Reduce sensor update rates or simplify sensor models
- **High CPU Usage**: Optimize sensor processing and reduce unnecessary visualization
- **Memory Issues**: Monitor sensor data buffering and processing pipelines

## Practical Lab: Multi-Sensor Integration

### Lab Objective
Create a robot model with multiple sensor types (camera, LiDAR, IMU) and implement sensor fusion for state estimation.

### Implementation Steps
1. Create a robot model with multiple sensors
2. Configure sensors with realistic parameters
3. Implement ROS 2 nodes for sensor data processing
4. Create a simple sensor fusion algorithm
5. Validate the integrated system

### Expected Outcome
- Working multi-sensor robot model
- Proper ROS 2 integration for all sensors
- Basic sensor fusion implementation
- Demonstrated understanding of sensor simulation

## Review Questions

1. How do you configure a camera sensor in Gazebo and what parameters affect image quality?
2. Explain the differences between 2D and 3D LiDAR simulation in Gazebo.
3. What are the key considerations for IMU sensor simulation and calibration?
4. How do you implement sensor noise models to match real-world characteristics?
5. Describe the process for validating simulated sensor data against real sensors.

## Next Steps
After mastering sensor simulation, students should proceed to:
- Gazebo workflows and simulation environments
- Unity visualization for robotics
- Advanced perception algorithms
- Sim-to-real transfer techniques

This comprehensive understanding of sensor simulation enables the creation of realistic robotic perception systems essential for Physical AI and Humanoid Robotics applications.