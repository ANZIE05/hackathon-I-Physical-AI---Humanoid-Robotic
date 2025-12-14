---
sidebar_position: 2
---

# Isaac Sim Workflows

## Overview
This section details the essential workflows for using NVIDIA Isaac Sim in Physical AI and Humanoid Robotics development. From initial setup to advanced simulation scenarios, these workflows provide systematic approaches to leveraging Isaac Sim's photorealistic capabilities for robotics research and development.

## Learning Objectives
By the end of this section, students will be able to:
- Set up and configure Isaac Sim environments for robotics applications
- Implement systematic workflows for simulation development and testing
- Create complex simulation scenarios with realistic environments
- Integrate Isaac Sim with ROS 2 development workflows
- Optimize simulation performance and realism
- Validate simulation results against real-world performance

## Isaac Sim Setup Workflows

### Initial Installation and Configuration
```bash
# Download and install Isaac Sim
wget https://developer.download.nvidia.com/isaac/isaac_sim/isaac_sim_2023.1.0.tar.gz
tar -xf isaac_sim_2023.1.0.tar.gz
cd isaac_sim_2023.1.0

# Install dependencies
./install_dependencies.sh

# Verify installation
python3 -c "import omni; print('Isaac Sim installed successfully')"
```

### Environment Configuration
```python
# Isaac Sim environment configuration script
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.utils.carb import set_carb_setting

def configure_isaac_sim_environment():
    """Configure Isaac Sim environment for optimal performance"""

    # Set physics solver parameters
    set_carb_setting("/physics/solverType", "TGS")  # TGS solver for stability
    set_carb_setting("/physics/iterations", 16)     # Solver iterations
    set_carb_setting("/physics/worker_thread_count", 8)  # Threading

    # Set rendering parameters
    set_carb_setting("/rtx/rendermode", "Interactive")  # Interactive rendering
    set_carb_setting("/renderer/resolution/width", 1280)  # Resolution
    set_carb_setting("/renderer/resolution/height", 720)

    # Set simulation parameters
    set_carb_setting("/app/player/playSimulations", True)
    set_carb_setting("/app/window/dockSpace", True)

    print("Isaac Sim environment configured successfully")

def setup_world_with_defaults():
    """Set up Isaac Sim world with default configurations"""

    # Create world with 1-meter stage units
    world = World(stage_units_in_meters=1.0)

    # Add default ground plane
    world.scene.add_default_ground_plane()

    # Add default lighting
    from omni.isaac.core.utils.prims import create_prim
    create_prim(
        prim_path="/World/Light",
        prim_type="DistantLight",
        position=[0, 0, 10],
        attributes={"color": [0.8, 0.8, 0.8]}
    )

    return world
```

### Project Structure Setup
```
isaac_sim_project/
├── assets/                 # Custom 3D models and environments
│   ├── robots/
│   │   ├── humanoid.urdf
│   │   └── mobile_base.urdf
│   ├── environments/
│   │   ├── office.usd
│   │   └── warehouse.usd
│   └── objects/
│       ├── furniture.usd
│       └── props.usd
├── configs/                # Configuration files
│   ├── robot_configs.yaml
│   └── simulation_params.yaml
├── scripts/                # Python scripts for automation
│   ├── spawn_robot.py
│   ├── setup_scene.py
│   └── run_simulation.py
├── launch/                 # ROS 2 launch files
│   └── isaac_sim.launch.py
└── logs/                   # Simulation logs and outputs
```

## Environment Creation Workflows

### Basic Environment Setup
```python
# Basic environment creation workflow
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.core.materials import OmniPBR

class EnvironmentCreator:
    def __init__(self, world_units=1.0):
        self.world = World(stage_units_in_meters=world_units)
        self.objects = []

    def create_basic_environment(self):
        """Create a basic environment with ground and lighting"""

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add lighting
        self.add_lighting()

        # Add basic objects
        self.add_basic_objects()

        return self.world

    def add_lighting(self):
        """Add realistic lighting to the environment"""

        # Add dome light for ambient lighting
        dome_light = self.world.scene.add(
            prim_path="/World/DomeLight",
            name="dome_light",
            light_type="DomeLight",
            color=[0.2, 0.2, 0.2],
            intensity=300
        )

        # Add directional light for shadows
        directional_light = self.world.scene.add(
            prim_path="/World/DirectionalLight",
            name="directional_light",
            light_type="DistantLight",
            color=[0.9, 0.9, 0.9],
            intensity=1000
        )

    def add_basic_objects(self):
        """Add basic objects to the environment"""

        # Add a table
        table = create_primitive(
            prim_path="/World/table",
            primitive_type="Cuboid",
            position=[2, 0, 0.5],
            scale=[1, 0.8, 1],
            color=[0.6, 0.4, 0.2]
        )

        # Add a box
        box = create_primitive(
            prim_path="/World/box",
            primitive_type="Cuboid",
            position=[2, 1, 0.25],
            scale=[0.5, 0.5, 0.5],
            color=[0.8, 0.2, 0.2]
        )

        self.objects.extend([table, box])

    def create_office_environment(self):
        """Create an office-style environment"""

        # Start with basic environment
        self.create_basic_environment()

        # Add office-specific objects
        self.add_office_desk()
        self.add_office_chair()
        self.add_office_equipment()

    def add_office_desk(self):
        """Add office desk to environment"""

        desk = create_primitive(
            prim_path="/World/desk",
            primitive_type="Cuboid",
            position=[0, 2, 0.4],
            scale=[1.5, 0.8, 0.8],
            color=[0.4, 0.4, 0.4]
        )

        # Add desk legs
        for i in range(4):
            x_offset = (-0.6 if i < 2 else 0.6)
            y_offset = (-0.3 if i % 2 == 0 else 0.3)

            leg = create_primitive(
                prim_path=f"/World/desk_leg_{i}",
                primitive_type="Cylinder",
                position=[x_offset, y_offset, 0.2],
                scale=[0.05, 0.05, 0.4],
                color=[0.3, 0.3, 0.3]
            )

        self.objects.append(desk)

    def add_office_chair(self):
        """Add office chair to environment"""

        chair_seat = create_primitive(
            prim_path="/World/chair_seat",
            primitive_type="Cuboid",
            position=[0, 1.5, 0.2],
            scale=[0.4, 0.4, 0.1],
            color=[0.2, 0.2, 0.2]
        )

        self.objects.append(chair_seat)

    def add_office_equipment(self):
        """Add office equipment and decorations"""

        # Add monitor
        monitor = create_primitive(
            prim_path="/World/monitor",
            primitive_type="Cuboid",
            position=[0, 2.1, 0.7],
            scale=[0.4, 0.3, 0.02],
            color=[0.1, 0.1, 0.1]
        )

        # Add plant
        plant_pot = create_primitive(
            prim_path="/World/plant_pot",
            primitive_type="Cylinder",
            position=[-1, 0, 0.25],
            scale=[0.2, 0.2, 0.5],
            color=[0.6, 0.4, 0.2]
        )

        self.objects.extend([monitor, plant_pot])
```

### Advanced Environment with USD
```python
# Advanced environment using USD composition
import omni
from pxr import Usd, Sdf, Gf, UsdGeom, UsdLux
from omni.isaac.core.utils.stage import add_reference_to_stage

class AdvancedEnvironmentBuilder:
    def __init__(self, stage_path="/World"):
        self.stage_path = stage_path
        self.stage = omni.usd.get_context().get_stage()

    def create_complex_office_scene(self):
        """Create a complex office scene using USD composition"""

        # Create main office room
        room_prim = self.stage.DefinePrim(f"{self.stage_path}/OfficeRoom", "Xform")

        # Add room geometry
        self.add_room_walls(room_prim)
        self.add_room_floor(room_prim)
        self.add_room_ceiling(room_prim)

        # Add furniture and objects
        self.add_desks_and_chairs(room_prim)
        self.add_meeting_area(room_prim)
        self.add_decorations(room_prim)

        # Add lighting
        self.add_office_lighting(room_prim)

    def add_room_walls(self, parent_prim):
        """Add room walls"""

        # Define room dimensions
        room_size = Gf.Vec3f(10, 8, 3)  # width, depth, height
        wall_thickness = 0.1

        # Create walls
        walls = [
            ("wall_front", Gf.Vec3f(0, room_size[1]/2, room_size[2]/2), Gf.Vec3f(room_size[0], wall_thickness, room_size[2])),
            ("wall_back", Gf.Vec3f(0, -room_size[1]/2, room_size[2]/2), Gf.Vec3f(room_size[0], wall_thickness, room_size[2])),
            ("wall_left", Gf.Vec3f(-room_size[0]/2, 0, room_size[2]/2), Gf.Vec3f(wall_thickness, room_size[1], room_size[2])),
            ("wall_right", Gf.Vec3f(room_size[0]/2, 0, room_size[2]/2), Gf.Vec3f(wall_thickness, room_size[1], room_size[2]))
        ]

        for name, position, size in walls:
            wall_prim = self.stage.DefinePrim(f"{parent_prim.GetPath()}/{name}", "Cube")
            UsdGeom.XformCommonAPI(wall_prim).SetTranslate(position)
            UsdGeom.XformCommonAPI(wall_prim).SetScale(size)

            # Apply wall material
            self.apply_wall_material(wall_prim)

    def apply_wall_material(self, prim):
        """Apply realistic wall material"""
        # This would apply OmniPBR material for realistic appearance
        pass

    def add_desks_and_chairs(self, parent_prim):
        """Add desks and chairs in office arrangement"""

        # Create desk arrangement
        for i in range(3):
            desk_x = -3 + i * 3
            self.add_desk_at_position(parent_prim, Gf.Vec3f(desk_x, 2, 0))

    def add_desk_at_position(self, parent_prim, position):
        """Add a complete desk setup at given position"""

        # Create desk group
        desk_group = self.stage.DefinePrim(f"{parent_prim.GetPath()}/Desk_{position[0]}", "Xform")
        UsdGeom.XformCommonAPI(desk_group).SetTranslate(position)

        # Add desk top
        desk_top = self.stage.DefinePrim(f"{desk_group.GetPath()}/Top", "Cube")
        UsdGeom.XformCommonAPI(desk_top).SetTranslate(Gf.Vec3f(0, 0, 0.75))
        UsdGeom.XformCommonAPI(desk_top).SetScale(Gf.Vec3f(1.5, 0.8, 0.05))

        # Add desk legs
        for j in range(4):
            leg_x = (-0.7 if j < 2 else 0.7)
            leg_y = (-0.35 if j % 2 == 0 else 0.35)

            leg = self.stage.DefinePrim(f"{desk_group.GetPath()}/Leg_{j}", "Cylinder")
            UsdGeom.XformCommonAPI(leg).SetTranslate(Gf.Vec3f(leg_x, leg_y, 0.35))
            UsdGeom.XformCommonAPI(leg).SetScale(Gf.Vec3f(0.05, 0.05, 0.7))
```

## Robot Integration Workflows

### Robot Spawning and Configuration
```python
# Robot integration workflow
import omni
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView

class RobotIntegrationManager:
    def __init__(self, world):
        self.world = world
        self.robots = {}

    def spawn_robot_from_usd(self, robot_name, usd_path, position=[0, 0, 0.5], orientation=[0, 0, 0, 1]):
        """Spawn robot from USD file"""

        prim_path = f"/World/{robot_name}"

        # Add robot to stage
        add_reference_to_stage(
            usd_path=usd_path,
            prim_path=prim_path
        )

        # Create robot object
        robot = Robot(
            prim_path=prim_path,
            name=robot_name,
            position=position,
            orientation=orientation
        )

        # Add to world
        self.world.scene.add(robot)

        # Store reference
        self.robots[robot_name] = robot

        return robot

    def spawn_mobile_robot(self, robot_name="mobile_robot", position=[0, 0, 0.1]):
        """Spawn a mobile robot (e.g., TurtleBot3)"""

        # Use a standard mobile robot model
        robot = self.spawn_robot_from_usd(
            robot_name=robot_name,
            usd_path="/Isaac/Robots/TurtleBot3Burger/turtlebot3_burger.usd",
            position=position
        )

        return robot

    def spawn_humanoid_robot(self, robot_name="humanoid_robot", position=[0, 0, 0.8]):
        """Spawn a humanoid robot"""

        # Use a humanoid robot model
        robot = self.spawn_robot_from_usd(
            robot_name=robot_name,
            usd_path="/Isaac/Robots/NVIDIA/Isaac/Robots/ant.usd",  # Example humanoid
            position=position
        )

        return robot

    def configure_robot_sensors(self, robot_name):
        """Configure sensors for the robot"""

        robot = self.robots[robot_name]

        # Add RGB camera
        self.add_camera_to_robot(robot)

        # Add IMU
        self.add_imu_to_robot(robot)

        # Add LiDAR (if applicable)
        self.add_lidar_to_robot(robot)

    def add_camera_to_robot(self, robot):
        """Add RGB camera to robot"""

        from omni.isaac.sensor import Camera

        camera = Camera(
            prim_path=f"{robot.prim_path}/chassis/camera",
            frequency=30,
            resolution=(640, 480)
        )

        self.world.scene.add(camera)

    def add_imu_to_robot(self, robot):
        """Add IMU to robot"""

        from omni.isaac.sensor import IMU

        imu = IMU(
            prim_path=f"{robot.prim_path}/chassis/imu",
            frequency=100
        )

        self.world.scene.add(imu)

    def add_lidar_to_robot(self, robot):
        """Add LiDAR to robot"""

        from omni.isaac.sensor import RotatingLidarSensor

        lidar = RotatingLidarSensor(
            prim_path=f"{robot.prim_path}/chassis/lidar",
            translation=np.array([0, 0, 0.2]),
            orientation=np.array([0, 0, 0, 1]),
            name="Lidar_Sensor",
            fov=360,
            horizontal_resolution=1,
            vertical_resolution=1,
            range=25.0,
            rotation_frequency=20,
            samples_per_cycle=360
        )

        self.world.scene.add(lidar)

    def configure_robot_controllers(self, robot_name):
        """Configure robot controllers"""

        robot = self.robots[robot_name]

        # Set up joint properties for control
        self.configure_joint_properties(robot)

        # Set up control interfaces
        self.setup_control_interfaces(robot)

    def configure_joint_properties(self, robot):
        """Configure robot joint properties"""

        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.utils.prims import get_prim_at_path
        from pxr import PhysxSchema, UsdPhysics

        stage = get_current_stage()

        # Configure joint friction and damping
        for joint_name in robot.joint_names:
            joint_path = f"{robot.prim_path}/{joint_name}"
            joint_prim = get_prim_at_path(joint_path)

            # Set joint properties
            if joint_prim:
                # Set joint friction
                PhysxSchema.PhysxJointAPI(joint_prim).CreateJointFrictionAttr(0.1)

                # Set joint damping
                PhysxSchema.PhysxJointAPI(joint_prim).CreateJointDampingAttr(0.01)

    def setup_control_interfaces(self, robot):
        """Set up control interfaces for the robot"""

        # This would set up ROS 2 control interfaces
        # or other control systems depending on the application
        pass
```

### Physics Configuration
```python
# Physics configuration for realistic robot simulation
from omni.isaac.core.utils.physics import set_articulation_properties
from omni.isaac.core.utils.prims import get_prim_at_path

def configure_robot_physics(robot):
    """Configure physics properties for realistic robot simulation"""

    # Get robot articulation
    articulation = get_prim_at_path(robot.prim_path)

    # Set physics properties
    set_articulation_properties(
        articulation=articulation,
        joint_friction=[0.1] * len(robot.joint_names),  # Joint friction
        joint_damping=[0.01] * len(robot.joint_names),  # Joint damping
        joint_stiffness=[0.0] * len(robot.joint_names)  # Joint stiffness
    )

    # Configure collision properties
    configure_collision_properties(robot)

def configure_collision_properties(robot):
    """Configure collision properties for the robot"""
    # Add collision filtering and material properties
    # This would include setting up proper collision groups
    # and material properties for realistic interactions
    pass
```

## Simulation Execution Workflows

### Basic Simulation Loop
```python
# Basic simulation execution workflow
import time
import threading
from collections import deque

class SimulationExecutor:
    def __init__(self, world, robots=None):
        self.world = world
        self.robots = robots or {}
        self.is_running = False
        self.simulation_time = 0.0
        self.real_time_factor = 1.0

        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.simulation_times = deque(maxlen=100)

    def run_simulation(self, duration=60.0, realtime=True):
        """Run simulation for specified duration"""

        self.is_running = True
        start_time = time.time()

        while self.is_running and (time.time() - start_time) < duration:
            frame_start = time.time()

            # Step simulation
            self.world.step(render=True)

            # Process robot control
            self.process_robot_control()

            # Process sensor data
            self.process_sensor_data()

            # Monitor performance
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)

            # Maintain real-time factor if requested
            if realtime and frame_time < (1.0 / 60.0):  # 60 FPS target
                sleep_time = (1.0 / 60.0) - frame_time
                time.sleep(sleep_time)

        self.is_running = False

    def process_robot_control(self):
        """Process robot control commands"""

        for robot_name, robot in self.robots.items():
            # Get control commands (from ROS 2 or other interfaces)
            control_cmd = self.get_robot_command(robot_name)

            if control_cmd:
                # Apply control to robot
                self.apply_robot_control(robot, control_cmd)

    def process_sensor_data(self):
        """Process sensor data from all robots"""

        for robot_name, robot in self.robots.items():
            # Get sensor data
            sensor_data = self.get_robot_sensor_data(robot_name)

            if sensor_data:
                # Process and publish sensor data
                self.process_and_publish_sensor_data(robot_name, sensor_data)

    def get_robot_command(self, robot_name):
        """Get control command for robot"""
        # This would interface with ROS 2 or other control systems
        return None

    def apply_robot_control(self, robot, control_cmd):
        """Apply control command to robot"""
        # Apply control command to robot
        pass

    def get_robot_sensor_data(self, robot_name):
        """Get sensor data from robot"""
        # Get sensor data from robot sensors
        return {}

    def process_and_publish_sensor_data(self, robot_name, sensor_data):
        """Process and publish sensor data"""
        # Process and publish sensor data to ROS 2 or other systems
        pass

    def get_performance_metrics(self):
        """Get current performance metrics"""

        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            # Calculate real-time factor
            avg_sim_step = sum(self.simulation_times) / len(self.simulation_times) if self.simulation_times else 0
            rtf = avg_sim_step / (1.0/60.0) if avg_sim_step > 0 else 0
        else:
            current_fps = 0
            rtf = 0

        return {
            'fps': current_fps,
            'real_time_factor': rtf,
            'frame_times': list(self.frame_times),
            'simulation_times': list(self.simulation_times)
        }
```

### Advanced Simulation with Recording
```python
# Advanced simulation with recording capabilities
import numpy as np
import cv2
from PIL import Image
import os

class AdvancedSimulationRecorder:
    def __init__(self, world, output_dir="simulation_recordings"):
        self.world = world
        self.output_dir = output_dir
        self.recordings = {}
        self.is_recording = False

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/poses", exist_ok=True)

    def start_recording(self, recording_name, record_images=True, record_depth=True, record_poses=True):
        """Start recording simulation data"""

        self.is_recording = True
        self.current_recording = recording_name

        # Initialize recording data
        self.recordings[recording_name] = {
            'images': [] if record_images else None,
            'depth': [] if record_depth else None,
            'poses': [] if record_poses else None,
            'timestamps': [],
            'record_images': record_images,
            'record_depth': record_depth,
            'record_poses': record_poses
        }

        print(f"Started recording: {recording_name}")

    def record_frame(self, robot, camera, timestamp):
        """Record current frame data"""

        if not self.is_recording:
            return

        recording = self.recordings[self.current_recording]
        recording['timestamps'].append(timestamp)

        # Record image if requested
        if recording['record_images'] and camera:
            image = self.get_camera_image(camera)
            if image is not None:
                recording['images'].append(image)
                # Save to file
                img_pil = Image.fromarray(image)
                img_pil.save(f"{self.output_dir}/images/frame_{len(recording['images']):06d}.png")

        # Record depth if requested
        if recording['record_depth'] and camera:
            depth = self.get_camera_depth(camera)
            if depth is not None:
                recording['depth'].append(depth)
                # Save to file
                depth_pil = Image.fromarray((depth * 1000).astype(np.uint16))  # Scale for 16-bit
                depth_pil.save(f"{self.output_dir}/depth/depth_{len(recording['depth']):06d}.png")

        # Record pose if requested
        if recording['record_poses'] and robot:
            pose = robot.get_world_pose()
            recording['poses'].append(pose)

    def get_camera_image(self, camera):
        """Get current camera image"""
        try:
            # Get image from Isaac Sim camera
            image_data = camera.get_rgb()
            return image_data
        except Exception as e:
            print(f"Error getting camera image: {e}")
            return None

    def get_camera_depth(self, camera):
        """Get current camera depth"""
        try:
            # Get depth from Isaac Sim depth sensor
            depth_data = camera.get_depth()
            return depth_data
        except Exception as e:
            print(f"Error getting camera depth: {e}")
            return None

    def stop_recording(self):
        """Stop current recording and save metadata"""

        if not self.is_recording:
            return

        # Save metadata
        recording = self.recordings[self.current_recording]
        metadata = {
            'recording_name': self.current_recording,
            'total_frames': len(recording['timestamps']),
            'duration': max(recording['timestamps']) - min(recording['timestamps']) if recording['timestamps'] else 0,
            'record_images': recording['record_images'],
            'record_depth': recording['record_depth'],
            'record_poses': recording['record_poses']
        }

        import json
        with open(f"{self.output_dir}/{self.current_recording}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Stopped recording: {self.current_recording}")
        print(f"Recorded {metadata['total_frames']} frames")

        self.is_recording = False
```

## ROS 2 Integration Workflows

### Isaac ROS Bridge Setup
```python
# Isaac ROS bridge integration workflow
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np

class IsaacROSIntegration(Node):
    def __init__(self):
        super().__init__('isaac_ros_integration')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers for Isaac Sim data
        self.rgb_publisher = self.create_publisher(Image, 'camera/rgb/image_raw', 10)
        self.depth_publisher = self.create_publisher(Image, 'camera/depth/image_raw', 10)
        self.odom_publisher = self.create_publisher(Odometry, 'odom', 10)
        self.scan_publisher = self.create_publisher(LaserScan, 'scan', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera/rgb/camera_info', 10)

        # Subscribers for robot commands
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.goal_subscriber = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, 10)

        # Timer for publishing sensor data
        self.publish_timer = self.create_timer(0.1, self.publish_sensor_data)

        # Isaac Sim integration
        self.isaac_connected = False
        self.robot_pose = np.eye(4)
        self.camera_data = None
        self.lidar_data = None

        self.get_logger().info('Isaac ROS Integration Node Started')

    def setup_isaac_connection(self):
        """Set up connection to Isaac Sim"""
        # This would establish connection to Isaac Sim
        # In practice, this might involve launching Isaac Sim
        # or connecting to a running instance
        self.isaac_connected = True
        self.get_logger().info('Connected to Isaac Sim')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS"""
        if not self.isaac_connected:
            return

        # Convert ROS Twist to Isaac Sim control
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Apply to Isaac Sim robot
        self.apply_robot_control(linear_x, angular_z)

        self.get_logger().debug(f'Received cmd_vel: linear_x={linear_x}, angular_z={angular_z}')

    def goal_callback(self, msg):
        """Handle navigation goals from ROS"""
        if not self.isaac_connected:
            return

        # Extract goal position
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        goal_z = msg.pose.position.z

        # Send to Isaac Sim navigation system
        self.set_navigation_goal(goal_x, goal_y, goal_z)

        self.get_logger().info(f'Set navigation goal: ({goal_x}, {goal_y}, {goal_z})')

    def publish_sensor_data(self):
        """Publish sensor data from Isaac Sim to ROS"""
        if not self.isaac_connected:
            return

        # Get data from Isaac Sim
        sensor_data = self.get_isaac_sensor_data()

        if sensor_data:
            # Publish RGB image
            if 'rgb_image' in sensor_data:
                rgb_msg = self.create_image_message(sensor_data['rgb_image'])
                rgb_msg.header.frame_id = 'camera_rgb_optical_frame'
                self.rgb_publisher.publish(rgb_msg)

                # Publish camera info
                self.publish_camera_info(rgb_msg.header)

            # Publish depth image
            if 'depth_image' in sensor_data:
                depth_msg = self.create_image_message(sensor_data['depth_image'])
                depth_msg.header.frame_id = 'camera_depth_optical_frame'
                self.depth_publisher.publish(depth_msg)

            # Publish odometry
            if 'odometry' in sensor_data:
                odom_msg = self.create_odometry_message(sensor_data['odometry'])
                self.odom_publisher.publish(odom_msg)

            # Publish laser scan
            if 'laser_scan' in sensor_data:
                scan_msg = self.create_laser_scan_message(sensor_data['laser_scan'])
                self.scan_publisher.publish(scan_msg)

    def create_image_message(self, image_array):
        """Create ROS Image message from numpy array"""
        # Convert numpy array to ROS Image
        image_msg = self.cv_bridge.cv2_to_imgmsg(image_array, encoding='rgb8')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        return image_msg

    def create_odometry_message(self, odometry_data):
        """Create ROS Odometry message"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set pose
        odom_msg.pose.pose.position.x = odometry_data['position'][0]
        odom_msg.pose.pose.position.y = odometry_data['position'][1]
        odom_msg.pose.pose.position.z = odometry_data['position'][2]

        odom_msg.pose.pose.orientation.x = odometry_data['orientation'][0]
        odom_msg.pose.pose.orientation.y = odometry_data['orientation'][1]
        odom_msg.pose.pose.orientation.z = odometry_data['orientation'][2]
        odom_msg.pose.pose.orientation.w = odometry_data['orientation'][3]

        # Set twist (velocity)
        odom_msg.twist.twist.linear.x = odometry_data['linear_velocity'][0]
        odom_msg.twist.twist.linear.y = odometry_data['linear_velocity'][1]
        odom_msg.twist.twist.linear.z = odometry_data['linear_velocity'][2]

        odom_msg.twist.twist.angular.x = odometry_data['angular_velocity'][0]
        odom_msg.twist.twist.angular.y = odometry_data['angular_velocity'][1]
        odom_msg.twist.twist.angular.z = odometry_data['angular_velocity'][2]

        return odom_msg

    def create_laser_scan_message(self, scan_data):
        """Create ROS LaserScan message"""
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'

        # Set scan parameters
        scan_msg.angle_min = scan_data['angle_min']
        scan_msg.angle_max = scan_data['angle_max']
        scan_msg.angle_increment = scan_data['angle_increment']
        scan_msg.time_increment = scan_data['time_increment']
        scan_msg.scan_time = scan_data['scan_time']
        scan_msg.range_min = scan_data['range_min']
        scan_msg.range_max = scan_data['range_max']

        # Set range data
        scan_msg.ranges = scan_data['ranges']
        scan_msg.intensities = scan_data.get('intensities', [])

        return scan_msg

    def publish_camera_info(self, header):
        """Publish camera calibration information"""
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.header.frame_id = 'camera_rgb_optical_frame'

        # Set camera parameters (adjust based on actual Isaac Sim camera)
        camera_info.width = 640
        camera_info.height = 480
        camera_info.distortion_model = 'plumb_bob'
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients
        camera_info.k = [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]  # Camera matrix
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Rectification matrix
        camera_info.p = [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]  # Projection matrix

        self.camera_info_publisher.publish(camera_info)

    def get_isaac_sensor_data(self):
        """Get sensor data from Isaac Sim"""
        # This would interface with Isaac Sim to get sensor data
        # Return a dictionary with sensor data
        return {
            'rgb_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # Placeholder
            'depth_image': np.random.random((480, 640)).astype(np.float32),  # Placeholder
            'odometry': {
                'position': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0],
                'linear_velocity': [0.0, 0.0, 0.0],
                'angular_velocity': [0.0, 0.0, 0.0]
            },
            'laser_scan': {
                'angle_min': -np.pi,
                'angle_max': np.pi,
                'angle_increment': np.pi / 180.0,  # 1 degree
                'time_increment': 0.0,
                'scan_time': 0.1,
                'range_min': 0.1,
                'range_max': 30.0,
                'ranges': [float(np.random.random() * 10) for _ in range(360)]  # Placeholder
            }
        }

    def apply_robot_control(self, linear_x, angular_z):
        """Apply control to Isaac Sim robot"""
        # This would interface with Isaac Sim to control the robot
        # Implementation depends on the robot model and control interface
        pass

    def set_navigation_goal(self, goal_x, goal_y, goal_z):
        """Set navigation goal in Isaac Sim"""
        # This would set a navigation goal in Isaac Sim
        # Implementation depends on Isaac Sim's navigation system
        pass
```

## Performance Optimization Workflows

### Simulation Performance Monitoring
```python
# Performance optimization workflow
import time
import threading
from collections import deque
import psutil
import GPUtil

class IsaacSimPerformanceOptimizer:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.frame_times = deque(maxlen=30)  # Last 30 frame times
        self.processing_times = deque(maxlen=30)
        self.memory_usage = deque(maxlen=30)
        self.cpu_usage = deque(maxlen=30)

        # Performance optimization parameters
        self.optimization_params = {
            'image_decimation': 1,  # Process every Nth frame
            'pointcloud_decimation': 4,  # Process every 4th point
            'feature_count': 1000,  # Number of features to track
            'bundle_adjustment_frequency': 10  # BA every N keyframes
        }

        # Threading for performance monitoring
        self.monitoring_thread = threading.Thread(target=self.monitor_performance)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def monitor_performance(self):
        """Monitor system performance in separate thread"""
        while True:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(cpu_percent)

            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            self.memory_usage.append(memory_percent)

            # GPU usage if available
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                self.gpu_usage.append(gpu_percent)
            else:
                self.gpu_usage.append(0)

            time.sleep(0.1)

    def adaptive_processing(self, data_type='image'):
        """Adapt processing based on performance"""
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            if current_fps < self.target_fps * 0.8:
                # Performance is low, reduce processing load
                if data_type == 'image':
                    self.optimization_params['feature_count'] = max(500,
                        int(self.optimization_params['feature_count'] * 0.9))
                    self.optimization_params['image_decimation'] = min(5,
                        self.optimization_params['image_decimation'] + 1)
                elif data_type == 'lidar':
                    self.optimization_params['pointcloud_decimation'] = min(16,
                        self.optimization_params['pointcloud_decimation'] * 2)
            elif current_fps > self.target_fps * 1.1:
                # Performance is good, can afford more processing
                if data_type == 'image':
                    self.optimization_params['feature_count'] = min(2000,
                        int(self.optimization_params['feature_count'] * 1.1))
                    self.optimization_params['image_decimation'] = max(1,
                        self.optimization_params['image_decimation'] - 1)
                elif data_type == 'lidar':
                    self.optimization_params['pointcloud_decimation'] = max(1,
                        self.optimization_params['pointcloud_decimation'] // 1.1)

    def optimize_rendering_settings(self):
        """Optimize rendering settings for performance"""

        # Reduce rendering quality for training data generation
        from omni.isaac.core.utils.carb import set_carb_setting

        set_carb_setting("/rtx/rendermode", "Interactive")  # Less demanding than Raytraced
        set_carb_setting("/rtx/indirectdiffuse:disable", True)  # Disable expensive effects
        set_carb_setting("/rtx/pathtracing:disable", True)  # Disable full path tracing
        set_carb_setting("/renderer/maxSamples", 16)  # Reduce max samples for faster rendering

        # Optimize for synthetic data generation
        set_carb_setting("/app/player/playSimulations", False)  # Don't play animations during data gen
        set_carb_setting("/renderer/resolution/width", 640)  # Lower resolution for faster processing
        set_carb_setting("/renderer/resolution/height", 480)  # Lower resolution for faster processing

    def optimize_physics_settings(self):
        """Optimize physics settings for performance"""

        from omni.isaac.core.utils.carb import set_carb_setting

        # Adjust physics parameters for better performance
        set_carb_setting("/physics/solverType", "TGS")  # Generally faster solver
        set_carb_setting("/physics/iterations", 8)  # Reduce iterations for speed
        set_carb_setting("/physics/maxDepenetrationVelocity", 10.0)  # Limit velocity for stability

        # Reduce contact processing
        set_carb_setting("/physics/contactCollection", 2)  # Reduce contact processing
        set_carb_setting("/physics/maxAngularSpeed", 50.0)  # Limit angular velocity

    def get_performance_metrics(self):
        """Get current performance metrics"""
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            current_fps = 0

        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        avg_cpu_usage = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory_usage = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0

        return {
            'fps': current_fps,
            'avg_processing_time': avg_processing_time,
            'cpu_usage': avg_cpu_usage,
            'memory_usage': avg_memory_usage,
            'optimization_params': self.optimization_params.copy()
        }

    def optimize_pipeline(self):
        """Optimize entire pipeline based on performance"""
        current_perf = self.get_current_performance()

        # Adjust pipeline parameters based on performance
        if current_perf['fps'] < self.target_fps * 0.5:
            # Significantly below target - aggressive optimization
            self.get_logger().warn('Significant performance degradation detected - applying aggressive optimization')
            self.optimization_params['feature_count'] = max(200, self.optimization_params['feature_count'] // 2)
            self.optimization_params['image_decimation'] = min(10, self.optimization_params['image_decimation'] * 2)
            self.optimization_params['pointcloud_decimation'] = min(16, self.optimization_params['pointcloud_decimation'] * 2)
        elif current_perf['fps'] > self.target_fps * 1.2:
            # Above target - can afford more processing
            self.optimization_params['feature_count'] = min(2000, self.optimization_params['feature_count'] * 1.1)
            self.optimization_params['image_decimation'] = max(1, self.optimization_params['image_decimation'] // 1.1)
```

## Troubleshooting and Best Practices

### Common Issues and Solutions
```python
# Isaac Sim troubleshooting guide
class IsaacSimTroubleshooter:
    def __init__(self):
        self.known_issues = {
            'gpu_not_detected': {
                'symptoms': ['CUDA errors', 'GPU not utilized'],
                'causes': ['Driver issues', 'CUDA version mismatch', 'GPU not properly configured'],
                'solutions': [
                    'Update NVIDIA drivers',
                    'Verify CUDA installation',
                    'Check Isaac Sim GPU requirements',
                    'Install proper Isaac ROS packages'
                ]
            },
            'performance_degradation': {
                'symptoms': ['Low FPS', 'High latency', 'Memory leaks'],
                'causes': ['Insufficient hardware', 'Inefficient algorithms', 'Memory management issues'],
                'solutions': [
                    'Optimize processing parameters',
                    'Implement adaptive processing',
                    'Add performance monitoring',
                    'Upgrade hardware if needed'
                ]
            },
            'sensor_data_issues': {
                'symptoms': ['No sensor data', 'Corrupted data', 'Wrong coordinate frames'],
                'causes': ['Incorrect sensor configuration', 'TF issues', 'Message format problems'],
                'solutions': [
                    'Verify sensor configuration in Isaac Sim',
                    'Check TF tree and transforms',
                    'Validate message formats',
                    'Test sensor separately'
                ]
            },
            'robot_control_problems': {
                'symptoms': ['Robot not responding', 'Unstable movement', 'Drifting'],
                'causes': ['Control parameter issues', 'Physics configuration', 'Joint limits'],
                'solutions': [
                    'Verify control parameters',
                    'Check physics configuration',
                    'Validate joint limits and properties',
                    'Test control separately'
                ]
            }
        }

    def diagnose_issue(self, error_message):
        """Diagnose issue based on error message"""
        for issue_type, issue_data in self.known_issues.items():
            for symptom in issue_data['symptoms']:
                if symptom.lower() in error_message.lower():
                    return {
                        'issue_type': issue_type,
                        'symptoms': issue_data['symptoms'],
                        'causes': issue_data['causes'],
                        'solutions': issue_data['solutions']
                    }

        return {'issue_type': 'unknown', 'solutions': ['Check general troubleshooting steps']}

    def check_system_compatibility(self):
        """Check system compatibility with Isaac Sim requirements"""
        import subprocess
        import platform

        checks = {
            'os_compatible': self.check_os_compatibility(),
            'gpu_available': self.check_gpu_availability(),
            'cuda_installed': self.check_cuda_installation(),
            'driver_version': self.check_driver_version(),
            'memory_sufficient': self.check_memory(),
            'disk_space': self.check_disk_space()
        }

        return checks

    def check_os_compatibility(self):
        """Check if OS is compatible with Isaac Sim"""
        os_name = platform.system().lower()
        os_version = platform.release()

        # Isaac Sim officially supports Ubuntu 20.04/22.04
        if os_name == 'linux':
            try:
                with open('/etc/os-release', 'r') as f:
                    os_info = f.read()
                    if 'ubuntu' in os_info.lower() and ('20.04' in os_info or '22.04' in os_info):
                        return True
            except:
                pass

        return False

    def check_gpu_availability(self):
        """Check if compatible GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'RTX' in result.stdout:
                return True
        except:
            pass

        return False

    def check_cuda_installation(self):
        """Check if CUDA is properly installed"""
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def check_driver_version(self):
        """Check if driver version is compatible"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_str = result.stdout.strip()
                version_parts = version_str.split('.')
                if len(version_parts) >= 2:
                    major_version = int(version_parts[0])
                    # Isaac Sim requires relatively recent drivers
                    return major_version >= 470
        except:
            pass

        return False

    def check_memory(self):
        """Check if system has sufficient memory"""
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        return memory_gb >= 16  # Isaac Sim recommends 16GB+

    def check_disk_space(self):
        """Check if sufficient disk space is available"""
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        return free_gb >= 50  # Recommend at least 50GB free

    def generate_system_report(self):
        """Generate comprehensive system compatibility report"""
        checks = self.check_system_compatibility()

        report = f"""
Isaac Sim System Compatibility Report
====================================

System Checks:
- OS Compatible: {'✓' if checks['os_compatible'] else '✗'}
- GPU Available: {'✓' if checks['gpu_available'] else '✗'}
- CUDA Installed: {'✓' if checks['cuda_installed'] else '✗'}
- Driver Version: {'✓' if checks['driver_version'] else '✗'}
- Memory Sufficient: {'✓' if checks['memory_sufficient'] else '✗'}
- Disk Space: {'✓' if checks['disk_space'] else '✗'}

Recommendations:
"""
        if not checks['os_compatible']:
            report += "- Upgrade to Ubuntu 20.04 or 22.04\n"
        if not checks['gpu_available']:
            report += "- Install NVIDIA RTX GPU\n"
        if not checks['cuda_installed']:
            report += "- Install CUDA toolkit\n"
        if not checks['driver_version']:
            report += "- Update NVIDIA drivers\n"
        if not checks['memory_sufficient']:
            report += "- Upgrade to 16GB+ RAM\n"
        if not checks['disk_space']:
            report += "- Free up disk space (need 50GB+)\n"

        return report
```

## Practical Lab: Complete Isaac Sim VSLAM Integration

### Lab Objective
Implement a complete VSLAM system integrated with Isaac Sim that includes camera simulation, feature tracking, pose estimation, and map building.

### Implementation Steps

#### Step 1: Set up Isaac Sim Environment
```python
# Complete VSLAM integration example
import omni
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

class IsaacVSLAMIntegration(Node):
    def __init__(self):
        super().__init__('isaac_vslam_integration')

        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)

        # ROS 2 publishers and subscribers
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'vslam/pose', 10)
        self.status_pub = self.create_publisher(String, 'vslam/status', 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        # Isaac Sim components
        self.camera = None
        self.robot = None

        # VSLAM state
        self.vslam_system = None
        self.isaac_connected = False
        self.ros_connected = True

        # Performance monitoring
        self.frame_count = 0
        self.last_published_time = self.get_clock().now()

        self.get_logger().info('Isaac VSLAM Integration Node Started')

    def setup_isaac_environment(self):
        """Set up Isaac Sim environment with robot and sensors"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add robot
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name="isaac_robot",
                usd_path="/Isaac/Robots/TurtleBot3Burger/turtlebot3_burger.usd",
                position=[0, 0, 0.1],
                orientation=[0, 0, 0, 1]
            )
        )

        # Add camera to robot
        self.camera = Camera(
            prim_path="/World/Robot/chassis/camera",
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.camera)

        # Set up lighting
        from omni.isaac.core.utils.prims import create_prim
        create_prim(
            prim_path="/World/Light",
            prim_type="DistantLight",
            position=[0, 0, 10],
            attributes={"color": [0.8, 0.8, 0.8]}
        )

        # Initialize VSLAM system
        camera_matrix = np.array([
            [525.0, 0.0, 319.5],  # fx, 0, cx
            [0.0, 525.0, 239.5],  # 0, fy, cy
            [0.0, 0.0, 1.0]       # 0, 0, 1
        ])
        self.vslam_system = ORBSLAM(camera_matrix, np.zeros(5))

        self.isaac_connected = True
        self.get_logger().info('Isaac Sim environment set up successfully')

    def run_simulation(self, steps=1000):
        """Run Isaac Sim with VSLAM integration"""
        self.world.reset()

        for step in range(steps):
            self.world.step(render=True)

            # Process Isaac Sim data and publish to ROS
            if self.isaac_connected:
                self.process_isaac_data()

            # Check for ROS commands
            rclpy.spin_once(self, timeout_sec=0)

    def process_isaac_data(self):
        """Process Isaac Sim sensor data and run VSLAM"""
        try:
            # Get camera image from Isaac Sim
            camera_image = self.camera.get_rgba()

            if camera_image is not None:
                # Convert Isaac image to ROS Image message
                ros_image = self.isaac_to_ros_image(camera_image)

                # Run VSLAM on image
                estimated_pose = self.vslam_system.process_frame(
                    camera_image, self.world.current_time
                )

                # Publish image
                self.image_pub.publish(ros_image)

                # Publish estimated pose
                self.publish_pose_estimate(estimated_pose)

                # Performance monitoring
                self.frame_count += 1
                current_time = self.get_clock().now()
                if (current_time - self.last_published_time).nanoseconds > 1e9:  # 1 second
                    fps = self.frame_count / ((current_time - self.last_published_time).nanoseconds / 1e9)
                    self.get_logger().info(f'VSLAM: {self.frame_count} frames, FPS: {fps:.1f}')
                    self.frame_count = 0
                    self.last_published_time = current_time

        except Exception as e:
            self.get_logger().error(f'Error processing Isaac data: {e}')

    def isaac_to_ros_image(self, isaac_image):
        """Convert Isaac Sim image to ROS Image message"""
        import numpy as np
        from cv_bridge import CvBridge

        # Isaac image format may need conversion
        # This is a simplified example - actual format depends on Isaac Sim version
        image_data = np.array(isaac_image)

        # Convert to ROS Image using CV Bridge
        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(image_data, encoding='rgba8')

        # Set header
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera_rgb_optical_frame'

        return ros_image

    def publish_pose_estimate(self, pose):
        """Publish VSLAM pose estimate"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'vslam_map'

        # Convert 4x4 pose matrix to position and orientation
        pose_msg.pose.position.x = float(pose[0, 3])
        pose_msg.pose.position.y = float(pose[1, 3])
        pose_msg.pose.position.z = float(pose[2, 3])

        # Convert rotation matrix to quaternion
        rotation_matrix = pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)
        pose_msg.pose.orientation.w = qw
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz

        self.pose_pub.publish(pose_msg)

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [w, x, y, z]"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return qw, qx, qy, qz

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS"""
        if self.robot is not None:
            # Apply velocity command to Isaac Sim robot
            # This would involve controlling the robot in Isaac Sim
            linear_x = msg.linear.x
            angular_z = msg.angular.z

            # In Isaac Sim, you would apply these velocities to the robot
            # The exact method depends on the robot model and control interface
            self.apply_robot_velocity(linear_x, angular_z)

    def apply_robot_velocity(self, linear_x, angular_z):
        """Apply velocity to Isaac Sim robot"""
        # This is a placeholder - actual implementation depends on robot model
        # You would typically use Isaac Sim's control interfaces
        self.get_logger().debug(f'Applying velocity: linear_x={linear_x}, angular_z={angular_z}')

def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Initialize Isaac Sim (this would be done in Isaac Sim's application)
    # For this example, we assume Isaac Sim is already running

    # Create integration node
    integration_node = IsaacVSLAMIntegration()

    try:
        # Set up Isaac environment
        integration_node.setup_isaac_environment()

        # Run simulation
        integration_node.run_simulation(steps=1000)

    except KeyboardInterrupt:
        pass
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Lab Exercise: Isaac Sim VSLAM Implementation
1. Set up Isaac Sim with a mobile robot model
2. Implement VSLAM system with feature tracking
3. Integrate with ROS 2 for visualization
4. Test system in various environments
5. Evaluate performance and accuracy

### Expected Results
- Working VSLAM system integrated with Isaac Sim
- Real-time pose estimation and mapping
- Proper ROS 2 integration and communication
- Performance within acceptable limits

## Best Practices

### Isaac Sim Workflows Best Practices
1. **Environment Design**: Create realistic but optimized environments
2. **Physics Configuration**: Tune physics parameters for stability
3. **Sensor Simulation**: Configure sensors to match real-world characteristics
4. **Performance Monitoring**: Continuously monitor and optimize performance
5. **Validation**: Validate simulation results against real-world data

### Integration Best Practices
1. **Modular Design**: Keep components modular and reusable
2. **Error Handling**: Implement robust error handling
3. **Performance Monitoring**: Monitor and optimize performance continuously
4. **Documentation**: Maintain clear documentation for all components
5. **Testing**: Implement comprehensive testing for all systems

## Review Questions

1. Explain the process for setting up Isaac Sim with a custom robot model.
2. How do you configure VSLAM systems for optimal performance in Isaac Sim?
3. What are the key considerations for ROS 2 integration with Isaac Sim?
4. How do you troubleshoot common Isaac Sim performance issues?
5. What are the best practices for creating realistic simulation environments?

## Next Steps
After mastering Isaac Sim workflows, students should proceed to:
- Advanced VSLAM implementation for humanoid robots
- Isaac ROS integration and advanced features
- Navigation systems with Isaac Sim
- Sim-to-real transfer techniques

This comprehensive guide to Isaac Sim workflows provides the practical foundation for creating sophisticated simulation environments essential for Physical AI and Humanoid Robotics development.