---
sidebar_position: 3
---

# Sensor Stack

## Overview
A comprehensive sensor stack is essential for Physical AI and Humanoid Robotics systems. This section covers the selection, integration, and utilization of various sensors to enable perception, navigation, and interaction capabilities.

## Sensor Categories

### Vision Sensors
- **RGB Cameras**: Color image capture for object recognition
- **Depth Cameras**: 3D scene understanding
- **Stereo Cameras**: Depth estimation and 3D reconstruction
- **Thermal Cameras**: Temperature detection and night vision
- **Event Cameras**: High-speed motion detection

### Inertial Sensors
- **IMU (Inertial Measurement Unit)**: Acceleration, angular velocity, orientation
- **Gyroscope**: Angular velocity measurement
- **Accelerometer**: Linear acceleration measurement
- **Magnetometer**: Magnetic field and heading measurement

### Range Sensors
- **LiDAR**: 360-degree distance measurement
- **Ultrasonic Sensors**: Short-range obstacle detection
- **Time-of-Flight (ToF)**: Precise distance measurement
- **Radar**: All-weather detection capability

### Environmental Sensors
- **Temperature Sensors**: Environmental and internal temperature
- **Humidity Sensors**: Environmental conditions
- **Barometric Pressure**: Altitude estimation
- **Gas Sensors**: Air quality and chemical detection

### Proprioceptive Sensors
- **Joint Encoders**: Joint position and velocity
- **Force/Torque Sensors**: Contact force measurement
- **Tactile Sensors**: Touch and pressure detection
- **Current Sensors**: Motor current and load estimation

## Vision Sensors

### RGB Cameras
- **Resolution**: 720p to 4K depending on application
- **Frame Rate**: 30-120 FPS for real-time processing
- **Field of View**: 60-180 degrees
- **Interface**: USB, MIPI CSI-2, GigE Vision
- **ROS Integration**: image_transport, camera_info_manager

### Depth Cameras
- **Structured Light**: High accuracy at short range (Intel RealSense)
- **Stereo Vision**: Good range, moderate accuracy (ZED cameras)
- **ToF (Time of Flight)**: Fast, good for medium range (PMD CamBoard)
- **Accuracy**: 1-50mm depending on technology and range

### Camera Selection Guide
- **Close Range**: Intel RealSense D435/D455
- **Medium Range**: ZED 2i or stereo cameras
- **Long Range**: LiDAR + camera fusion
- **High Speed**: Event cameras for dynamic scenes

## Inertial Sensors

### IMU Types
- **6-axis IMU**: Accelerometer + gyroscope
- **9-axis IMU**: Accelerometer + gyroscope + magnetometer
- **10-axis IMU**: 9-axis + pressure sensor
- **Applications**: State estimation, balance control, navigation

### Key Specifications
- **Gyro Range**: ±250 to ±2000 °/s
- **Accel Range**: ±2 to ±16 g
- **Bias Stability**: &lt;1°/s for high-quality IMUs
- **Noise Density**: &lt;10 µg/√Hz for accelerometers

### Integration with Robotics
- **State Estimation**: Robot localization and mapping
- **Balance Control**: Humanoid robot stability
- **Motion Tracking**: Human-robot interaction
- **Calibration**: Temperature compensation and bias correction

## Range Sensors

### LiDAR Technologies
- **Spinning LiDAR**: 360° coverage (HDL-64E, VLP-16)
- **Solid State**: No moving parts, compact (Ouster, Velodyne Puck)
- **Flash LiDAR**: Instant 3D capture
- **MEMS Scanning**: Cost-effective solid state option

### LiDAR Selection Criteria
- **Range**: 5m to 300m depending on application
- **Resolution**: 0.1° to 2° angular resolution
- **FOV**: 360° horizontal, 10-40° vertical
- **Accuracy**: 1-3cm distance accuracy
- **Point Rate**: 10K to 2.6M points per second

### Alternative Range Sensors
- **Ultrasonic**: 2-4m range, low cost, good for close obstacles
- **IR Distance**: 0.1-5m, low power, simple integration
- **Laser Range Finders**: High accuracy, single point measurement

## Environmental Sensors

### Temperature and Humidity
- **DHT22**: Basic temperature/humidity sensing
- **BME280**: Temperature, humidity, pressure in one chip
- **SHT30**: High-accuracy temperature/humidity
- **Applications**: Environmental monitoring, thermal management

### Air Quality Sensors
- **MQ Series**: Gas detection (CO, LPG, smoke)
- **SGP30**: TVOC and CO2 equivalent
- **CCS811**: Low-power air quality monitoring
- **Integration**: Environmental awareness and safety

## Proprioceptive Sensors

### Joint Position Sensors
- **Incremental Encoders**: Relative position measurement
- **Absolute Encoders**: Absolute position measurement
- **Potentiometers**: Simple position sensing
- **Resolvers**: High-reliability position sensing

### Force/Torque Sensors
- **6-axis F/T Sensors**: Full force/torque vector measurement
- **Strain Gauge Sensors**: High-precision force measurement
- **Optical Sensors**: Non-contact force measurement
- **Applications**: Grasping, assembly, human-robot interaction

## Sensor Fusion

### Data Integration
- **Kalman Filters**: Optimal state estimation
- **Particle Filters**: Non-linear state estimation
- **Complementary Filters**: Simple sensor combination
- **Extended Kalman Filter**: Non-linear systems

### Fusion Strategies
- **Early Fusion**: Combine raw sensor data
- **Late Fusion**: Combine processed sensor outputs
- **Deep Fusion**: Neural network-based fusion
- **Multi-Modal Fusion**: Combine different sensor types

## ROS 2 Sensor Integration

### Standard Interfaces
- **sensor_msgs**: Standard message types for sensors
- **geometry_msgs**: Position and orientation messages
- **tf2**: Coordinate frame transformations
- **image_transport**: Compressed image transport

### Sensor Drivers
- **camera_driver**: Generic camera interface
- **imu_driver**: Standard IMU integration
- **laser_driver**: LiDAR data processing
- **joint_state_publisher**: Joint position reporting

## Sensor Calibration

### Camera Calibration
- **Intrinsic Parameters**: Focal length, principal point, distortion
- **Extrinsic Parameters**: Position and orientation relative to robot
- **Tools**: camera_calibration package, Kalibr
- **Frequency**: Regular recalibration recommended

### IMU Calibration
- **Bias Correction**: Accelerometer and gyroscope bias
- **Scale Factor**: Correct for sensor scaling errors
- **Alignment**: Coordinate frame alignment
- **Temperature Compensation**: Correct for temperature effects

### LiDAR Calibration
- **Extrinsic Calibration**: Position/orientation relative to robot
- **Intrinsic Calibration**: Range accuracy and angular precision
- **Multi-LiDAR**: Calibration between multiple units
- **Validation**: Regular accuracy verification

## Sensor Mounting and Placement

### Mechanical Considerations
- **Vibration Isolation**: Protect sensitive sensors
- **Clear View**: Ensure unobstructed sensor fields of view
- **Protection**: Weather and impact protection
- **Accessibility**: Easy maintenance and calibration

### Electrical Considerations
- **Power Supply**: Proper voltage regulation
- **Signal Integrity**: Minimize electromagnetic interference
- **Grounding**: Proper electrical grounding
- **EMI/RFI**: Electromagnetic compatibility

## Power and Bandwidth Requirements

### Power Consumption
- **Cameras**: 1-5W depending on resolution and processing
- **IMU**: 0.1-1W for typical units
- **LiDAR**: 5-20W depending on type and range
- **Total System**: 20-100W for comprehensive sensor suite

### Data Bandwidth
- **Cameras**: 10-100 MB/s depending on resolution and frame rate
- **LiDAR**: 1-10 MB/s for typical units
- **IMU**: 1-10 KB/s for typical units
- **Network**: 100 Mbps minimum, 1 Gbps recommended

## Safety and Redundancy

### Safety Considerations
- **Fail-Safe Operation**: Define behavior when sensors fail
- **Redundancy**: Multiple sensors for critical functions
- **Validation**: Cross-check sensor readings
- **Monitoring**: Continuous sensor health monitoring

### Redundancy Strategies
- **Triple Modular Redundancy**: Three sensors for critical measurements
- **Cross-Validation**: Use different sensor types for same measurement
- **Temporal Redundancy**: Multiple readings over time
- **Geographic Redundancy**: Multiple sensors at different locations

## Sensor Testing and Validation

### Performance Metrics
- **Accuracy**: How close measurements are to true values
- **Precision**: Repeatability of measurements
- **Resolution**: Smallest detectable change
- **Response Time**: Time to respond to changes

### Validation Procedures
- **Laboratory Testing**: Controlled environment validation
- **Field Testing**: Real-world performance validation
- **Calibration Verification**: Regular accuracy checks
- **Long-term Stability**: Monitor drift over time

## Cost Considerations

### Budget Sensors
- **Cameras**: $50-200 for basic units
- **IMU**: $10-100 for basic units
- **Ultrasonic**: $5-20 per sensor
- **Basic Setup**: $200-500 total

### Professional Sensors
- **High-Res Cameras**: $200-1000+ per camera
- **Professional IMU**: $100-1000+ per unit
- **LiDAR**: $500-50000+ depending on type
- **Complete Setup**: $2000-100000+ total

### ROI Considerations
- **Performance Requirements**: Match sensors to application needs
- **Integration Complexity**: Consider development time
- **Maintenance**: Ongoing calibration and support
- **Scalability**: Consider fleet deployment costs

## Troubleshooting Common Issues

### Sensor Noise
- **Electromagnetic Interference**: Check grounding and shielding
- **Vibration**: Secure mounting and vibration isolation
- **Temperature Effects**: Implement temperature compensation
- **Power Supply**: Ensure clean, stable power supply

### Communication Issues
- **Bandwidth Limitations**: Reduce data rates or compression
- **Protocol Errors**: Verify message formats and timing
- **Network Congestion**: Implement priority and buffering
- **Synchronization**: Ensure proper timestamping

This comprehensive sensor stack guide provides the foundation for selecting, integrating, and maintaining the sensor systems essential for Physical AI and Humanoid Robotics applications.