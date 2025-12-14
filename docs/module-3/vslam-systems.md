---
sidebar_position: 3
---

# VSLAM Systems

## Overview
Visual Simultaneous Localization and Mapping (VSLAM) is a critical component of autonomous robotic navigation, enabling robots to understand and navigate their environment in real-time. This section covers the theory, implementation, and integration of VSLAM systems with Isaac Sim and ROS 2 for Physical AI and Humanoid Robotics applications.

## Learning Objectives
By the end of this section, students will be able to:
- Understand the principles of Visual SLAM and its applications in robotics
- Implement VSLAM algorithms using Isaac Sim and ROS 2
- Integrate VSLAM systems with robot perception and navigation
- Optimize VSLAM performance for real-time robotics applications
- Evaluate VSLAM accuracy and robustness in various environments

## Key Concepts

### VSLAM Fundamentals
- **Localization**: Estimating the robot's position in the environment
- **Mapping**: Building a representation of the environment
- **Simultaneous**: Performing both tasks concurrently in real-time
- **Visual**: Using camera imagery as the primary sensor modality

### VSLAM Approaches
- **Feature-based**: Extract and track visual features
- **Direct Methods**: Use pixel intensities directly
- **Semi-direct**: Combine feature and direct approaches
- **Deep Learning**: Learn-based approaches using neural networks

### Key Components
- **Frontend**: Visual odometry and tracking
- **Backend**: Optimization and state estimation
- **Loop Closure**: Detect and correct for revisit
- **Mapping**: Create and maintain environment representation

## VSLAM Algorithms

### ORB-SLAM Architecture
```python
# ORB-SLAM implementation components
import numpy as np
import cv2
from collections import deque
import threading

class ORBSLAM:
    def __init__(self, camera_matrix, dist_coeffs):
        # Camera parameters
        self.K = camera_matrix  # Camera intrinsic matrix
        self.dist_coeffs = dist_coeffs

        # ORB detector and descriptor
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=20
        )

        # FLANN matcher for feature matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Tracking state
        self.current_frame = None
        self.previous_frame = None
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.keyframes = []
        self.map_points = []
        self.local_window = deque(maxlen=10)  # Local optimization window

        # Tracking parameters
        self.min_matches = 20
        self.reprojection_threshold = 3.0
        self.keyframe_threshold = 0.1  # Translation threshold for keyframes

    def process_frame(self, image, timestamp):
        """Process a new frame for VSLAM"""
        if self.previous_frame is None:
            # Initialize with first frame
            self.initialize_first_frame(image, timestamp)
            return self.current_pose

        # Extract features
        keypoints, descriptors = self.extract_features(image)

        # Match features with previous frame
        matches = self.match_features(descriptors)

        # Estimate motion
        if len(matches) >= self.min_matches:
            pose_change = self.estimate_motion(matches, keypoints)
            self.update_pose(pose_change)

            # Check if keyframe needed
            if self.should_add_keyframe():
                self.add_keyframe(image, keypoints, descriptors, timestamp)

            # Track map points
            self.track_map_points(keypoints)

        # Update frames
        self.previous_frame = self.current_frame.copy()
        self.current_frame = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'timestamp': timestamp
        }

        return self.current_pose

    def initialize_first_frame(self, image, timestamp):
        """Initialize VSLAM with first frame"""
        keypoints, descriptors = self.extract_features(image)

        self.current_frame = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'timestamp': timestamp
        }

        self.previous_frame = self.current_frame.copy()

    def extract_features(self, image):
        """Extract ORB features from image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect and compute ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is not None:
            # Convert keypoints to numpy array
            pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            return keypoints, descriptors
        else:
            return [], None

    def match_features(self, descriptors):
        """Match features with previous frame"""
        if descriptors is None or self.previous_frame['descriptors'] is None:
            return []

        try:
            # Use FLANN matcher
            matches = self.flann.knnMatch(
                descriptors, self.previous_frame['descriptors'], k=2
            )

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            return good_matches
        except cv2.error:
            return []

    def estimate_motion(self, matches, current_keypoints):
        """Estimate motion using matched features"""
        if len(matches) < self.min_matches:
            return np.eye(4)

        # Get corresponding points
        prev_pts = np.float32([self.previous_frame['keypoints'][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([current_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Undistort points
        prev_pts = cv2.undistortPoints(prev_pts, self.K, self.dist_coeffs, P=self.K)
        curr_pts = cv2.undistortPoints(curr_pts, self.K, self.dist_coeffs, P=self.K)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts, focal=self.K[0, 0], pp=(self.K[0, 2], self.K[1, 2]),
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is not None:
            # Recover pose
            _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, focal=self.K[0, 0], pp=(self.K[0, 2], self.K[1, 2]))

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()

            return T
        else:
            return np.eye(4)

    def update_pose(self, pose_change):
        """Update current pose with estimated motion"""
        self.current_pose = self.current_pose @ np.linalg.inv(pose_change)

    def should_add_keyframe(self):
        """Determine if current frame should be a keyframe"""
        if not self.keyframes:
            return True

        # Check translation distance from last keyframe
        last_keyframe_pose = self.keyframes[-1]['pose']
        current_translation = np.linalg.norm(
            self.current_pose[:3, 3] - last_keyframe_pose[:3, 3]
        )

        return current_translation > self.keyframe_threshold

    def add_keyframe(self, image, keypoints, descriptors, timestamp):
        """Add current frame as a keyframe"""
        keyframe = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': self.current_pose.copy(),
            'timestamp': timestamp
        }

        self.keyframes.append(keyframe)
        self.local_window.append(keyframe)

    def track_map_points(self, current_keypoints):
        """Track existing map points in current frame"""
        # This is a simplified implementation
        # In a full system, this would involve more sophisticated tracking
        pass

    def optimize_local_map(self):
        """Optimize local map using bundle adjustment"""
        # Perform local bundle adjustment on recent keyframes
        if len(self.local_window) < 2:
            return

        # This would implement local bundle adjustment
        # using optimization libraries like Ceres or scipy
        pass

    def detect_loop_closure(self):
        """Detect and handle loop closures"""
        # Compare current frame with past keyframes to detect revisits
        if len(self.keyframes) < 10:
            return

        # Use bag-of-words approach or DBoW2 for loop detection
        # This is a simplified placeholder
        pass
```

### Direct Methods (LSD-SLAM)
```python
# Semi-direct VSLAM implementation (LSD-SLAM style)
import numpy as np
import cv2
from scipy.spatial.distance import cdist

class LSDSLAM:
    def __init__(self, camera_matrix, width, height):
        self.K = camera_matrix
        self.width = width
        self.height = height

        # Line segment detector
        self.lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV)

        # Tracking parameters
        self.min_line_length = 20
        self.line_match_threshold = 5.0
        self.pose = np.eye(4)

        # Keyframe management
        self.keyframes = []
        self.lines = []  # 3D line segments

    def process_frame(self, image):
        """Process frame using direct line-based approach"""
        # Detect line segments
        lines = self.detect_lines(image)

        # Track lines across frames
        if self.keyframes:
            tracked_lines = self.track_lines(lines)
            self.update_pose_from_lines(tracked_lines)

        # Add keyframe if needed
        if self.should_add_keyframe():
            self.add_keyframe(image, lines)

        return self.pose

    def detect_lines(self, image):
        """Detect line segments in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Detect line segments
        lines, widths, precisions, nfa = self.lsd.detect(gray)

        if lines is not None:
            # Filter lines by length
            filtered_lines = []
            for i in range(len(lines)):
                pt1 = lines[i][0]
                pt2 = lines[i][1]
                length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                if length >= self.min_line_length:
                    filtered_lines.append({
                        'start': pt1,
                        'end': pt2,
                        'length': length,
                        'width': widths[i][0] if widths is not None else 1.0
                    })
            return filtered_lines
        else:
            return []

    def track_lines(self, current_lines):
        """Track lines across frames"""
        if not hasattr(self, 'previous_lines'):
            self.previous_lines = current_lines
            return []

        # Match current lines with previous lines
        matches = []
        for curr_line in current_lines:
            best_match = None
            best_distance = float('inf')

            for prev_line in self.previous_lines:
                # Calculate distance between line segments
                dist = self.line_distance(curr_line, prev_line)
                if dist < best_distance and dist < self.line_match_threshold:
                    best_distance = dist
                    best_match = prev_line

            if best_match:
                matches.append((curr_line, best_match))

        self.previous_lines = current_lines
        return matches

    def line_distance(self, line1, line2):
        """Calculate distance between two line segments"""
        # Calculate distance between midpoints of line segments
        mid1 = ((line1['start'][0] + line1['end'][0]) / 2,
                (line1['start'][1] + line1['end'][1]) / 2)
        mid2 = ((line2['start'][0] + line2['end'][0]) / 2,
                (line2['start'][1] + line2['end'][1]) / 2)

        dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
        return dist

    def update_pose_from_lines(self, matches):
        """Update pose using line correspondences"""
        if len(matches) < 3:
            return

        # Extract corresponding points
        prev_points = []
        curr_points = []

        for curr_line, prev_line in matches:
            # Use endpoints for pose estimation
            prev_points.extend([prev_line['start'], prev_line['end']])
            curr_points.extend([curr_line['start'], curr_line['end']])

        if len(prev_points) >= 6:  # At least 3 correspondences
            prev_pts = np.array(prev_points, dtype=np.float32)
            curr_pts = np.array(curr_points, dtype=np.float32)

            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                curr_pts, prev_pts, self.K, method=cv2.RANSAC, threshold=1.0
            )

            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, self.K)

                # Update pose
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.ravel()
                self.pose = self.pose @ T
```

## Isaac Sim VSLAM Integration

### Isaac Sim VSLAM Components
```python
# Isaac Sim VSLAM integration
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.sensor import Camera
from omni.isaac.core.robots import Robot
import numpy as np
import cv2

class IsaacVSLAMIntegration:
    def __init__(self, world, robot):
        self.world = world
        self.robot = robot

        # Initialize VSLAM system
        camera_matrix = np.array([[525.0, 0.0, 319.5],
                                  [0.0, 525.0, 239.5],
                                  [0.0, 0.0, 1.0]])

        self.vslam_system = ORBSLAM(camera_matrix, np.zeros(5))

        # Add camera to robot
        self.camera = Camera(
            prim_path="/World/Robot/chassis/camera",
            frequency=30,
            resolution=(640, 480)
        )

        self.world.scene.add(self.camera)

        # VSLAM state
        self.estimated_trajectory = []
        self.map_points = []
        self.keyframes = []

    def process_vslam_step(self):
        """Process one step of VSLAM"""
        # Get camera image from Isaac Sim
        image = self.get_camera_image()

        if image is not None:
            # Process with VSLAM system
            current_pose = self.vslam_system.process_frame(image, self.world.current_time)

            # Update robot pose estimate
            self.update_robot_estimate(current_pose)

            # Store trajectory
            self.estimated_trajectory.append({
                'timestamp': self.world.current_time,
                'pose': current_pose.copy()
            })

    def get_camera_image(self):
        """Get image from Isaac Sim camera"""
        # This would interface with Isaac Sim's camera system
        # Return numpy array of image data
        try:
            # Get image data from Isaac Sim camera
            image_data = self.camera.get_rgb()
            return image_data
        except Exception as e:
            print(f"Error getting camera image: {e}")
            return None

    def update_robot_estimate(self, estimated_pose):
        """Update robot pose estimate based on VSLAM"""
        # Compare with ground truth from Isaac Sim
        ground_truth_pose = self.robot.get_world_pose()

        # Calculate error
        position_error = np.linalg.norm(
            estimated_pose[:3, 3] - ground_truth_pose[0]
        )

        orientation_error = self.rotation_matrix_to_euler(
            estimated_pose[:3, :3]
        )
        gt_orientation = self.rotation_matrix_to_euler(
            ground_truth_pose[1].reshape(3, 3)
        )

        orientation_error_norm = np.linalg.norm(
            orientation_error - gt_orientation
        )

        print(f"VSLAM Error - Position: {position_error:.3f}m, "
              f"Orientation: {orientation_error_norm:.3f}rad")

    def rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to Euler angles"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])
```

### ROS 2 VSLAM Integration
```python
# ROS 2 VSLAM integration node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

class ROSVSLAMNode(Node):
    def __init__(self):
        super().__init__('ros_vslam_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Initialize VSLAM system
        self.camera_matrix = None
        self.vslam_system = None
        self.initialized = False

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/vslam/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/vslam/odometry', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/vslam/map', 10)
        self.traj_pub = self.create_publisher(Marker, '/vslam/trajectory', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)

        # Internal state
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.map_points = []
        self.frame_count = 0

        self.get_logger().info('ROS VSLAM Node Initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        if not self.initialized:
            # Extract camera matrix from camera info
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

            # Initialize VSLAM system
            self.vslam_system = ORBSLAM(self.camera_matrix, np.array(msg.d))
            self.initialized = True

            self.get_logger().info('VSLAM system initialized with camera calibration')

    def image_callback(self, msg):
        """Process incoming camera images for VSLAM"""
        if not self.initialized:
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process with VSLAM
            self.current_pose = self.vslam_system.process_frame(
                cv_image, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            )

            # Update trajectory
            self.trajectory.append(self.current_pose[:3, 3].copy())

            # Publish results
            self.publish_pose(msg.header)
            self.publish_odometry(msg.header)
            self.publish_trajectory()
            self.publish_map()

            # Broadcast TF
            self.broadcast_transform(msg.header)

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f'Error processing VSLAM frame: {e}')

    def publish_pose(self, header):
        """Publish estimated pose"""
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'vslam_map'

        # Convert 4x4 pose to Pose message
        pose_msg.pose.position.x = float(self.current_pose[0, 3])
        pose_msg.pose.position.y = float(self.current_pose[1, 3])
        pose_msg.pose.position.z = float(self.current_pose[2, 3])

        # Convert rotation matrix to quaternion
        rotation_matrix = self.current_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)
        pose_msg.pose.orientation.w = qw
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz

        self.pose_pub.publish(pose_msg)

    def publish_odometry(self, header):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.header.frame_id = 'vslam_map'
        odom_msg.child_frame_id = 'vslam_camera'

        # Position
        odom_msg.pose.pose.position.x = float(self.current_pose[0, 3])
        odom_msg.pose.position.y = float(self.current_pose[1, 3])
        odom_msg.pose.position.z = float(self.current_pose[2, 3])

        # Orientation
        rotation_matrix = self.current_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz

        # Set covariance to indicate uncertainty
        odom_msg.pose.covariance = [1e-3] * 36  # Simplified covariance

        self.odom_pub.publish(odom_msg)

    def publish_trajectory(self):
        """Publish trajectory visualization"""
        traj_marker = Marker()
        traj_marker.header.frame_id = 'vslam_map'
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.ns = 'vslam_trajectory'
        traj_marker.id = 0
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD

        # Set trajectory points
        for pos in self.trajectory[-100:]:  # Last 100 points
            point = PointStamped()
            point.point.x = float(pos[0])
            point.point.y = float(pos[1])
            point.point.z = float(pos[2])
            traj_marker.points.append(point.point)

        # Set visualization properties
        traj_marker.scale.x = 0.05  # Line width
        traj_marker.color.r = 1.0
        traj_marker.color.g = 0.0
        traj_marker.color.b = 0.0
        traj_marker.color.a = 1.0

        self.traj_pub.publish(traj_marker)

    def publish_map(self):
        """Publish map visualization"""
        marker_array = MarkerArray()

        # For now, publish keyframe positions as map points
        # In a real system, this would include 3D map points
        for i, keyframe in enumerate(getattr(self.vslam_system, 'keyframes', [])[-20:]):  # Last 20 keyframes
            marker = Marker()
            marker.header.frame_id = 'vslam_map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'vslam_map'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            pose = keyframe['pose']
            marker.pose.position.x = float(pose[0, 3])
            marker.pose.position.y = float(pose[1, 3])
            marker.pose.position.z = float(pose[2, 3])

            # Convert rotation to quaternion
            rot_q = self.rotation_matrix_to_quaternion(pose[:3, :3])
            marker.pose.orientation.w = rot_q[0]
            marker.pose.orientation.x = rot_q[1]
            marker.pose.orientation.y = rot_q[2]
            marker.pose.orientation.z = rot_q[3]

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.map_pub.publish(marker_array)

    def broadcast_transform(self, header):
        """Broadcast TF transform"""
        t = TransformStamped()

        t.header.stamp = header.stamp
        t.header.frame_id = 'vslam_map'
        t.child_frame_id = 'vslam_camera'

        t.transform.translation.x = float(self.current_pose[0, 3])
        t.transform.translation.y = float(self.current_pose[1, 3])
        t.transform.translation.z = float(self.current_pose[2, 3])

        rot_q = self.rotation_matrix_to_quaternion(self.current_pose[:3, :3])
        t.transform.rotation.w = rot_q[0]
        t.transform.rotation.x = rot_q[1]
        t.transform.rotation.y = rot_q[2]
        t.transform.rotation.z = rot_q[3]

        self.tf_broadcaster.sendTransform(t)

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
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

def main(args=None):
    rclpy.init(args=args)
    vslam_node = ROSVSLAMNode()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Deep Learning VSLAM

### Neural VSLAM Approaches
```python
# Deep learning based VSLAM (conceptual implementation)
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class NeuralVSLAM(nn.Module):
    def __init__(self, image_height=480, image_width=640):
        super(NeuralVSLAM, self).__init__()

        # Feature extraction backbone (e.g., ResNet)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification head

        # Pose estimation head
        self.pose_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)  # 6-DoF pose (3 translation + 3 rotation)
        )

        # Depth estimation head (monocular depth)
        self.depth_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, image_height * image_width)  # Dense depth map
        )

        # Mapping components
        self.map_encoder = nn.LSTM(512, 256, batch_first=True)
        self.map_decoder = nn.Linear(256, 3)  # 3D map point

        self.image_height = image_height
        self.image_width = image_width

    def forward(self, image_sequence):
        """
        Forward pass for neural VSLAM
        Args:
            image_sequence: Batch of image sequences [batch, seq_len, channels, height, width]
        Returns:
            poses: Estimated poses for each frame [batch, seq_len, 6]
            depths: Estimated depth maps [batch, seq_len, height, width]
            map_points: Reconstructed 3D points [batch, num_points, 3]
        """
        batch_size, seq_len = image_sequence.shape[:2]

        # Process each frame in the sequence
        features_list = []
        poses_list = []
        depths_list = []

        for t in range(seq_len):
            # Extract features
            features = self.backbone(image_sequence[:, t])  # [batch, 512]
            features_list.append(features)

            # Estimate pose
            pose = self.pose_head(features)  # [batch, 6]
            poses_list.append(pose)

            # Estimate depth
            depth = self.depth_head(features)  # [batch, height*width]
            depth = depth.view(batch_size, 1, self.image_height, self.image_width)
            depths_list.append(depth)

        # Stack results
        poses = torch.stack(poses_list, dim=1)  # [batch, seq_len, 6]
        depths = torch.stack(depths_list, dim=1)  # [batch, seq_len, 1, H, W]

        # Build map from features
        features_seq = torch.stack(features_list, dim=1)  # [batch, seq_len, 512]
        map_features, _ = self.map_encoder(features_seq)  # [batch, seq_len, 256]

        # Decode map points (simplified)
        map_points = self.map_decoder(map_features)  # [batch, seq_len, 3]

        return poses, depths, map_points

    def extract_features(self, image):
        """Extract features from a single image"""
        with torch.no_grad():
            features = self.backbone(image.unsqueeze(0))
        return features.squeeze(0)

    def estimate_pose(self, features):
        """Estimate pose from features"""
        with torch.no_grad():
            pose = self.pose_head(features)
        return pose

    def estimate_depth(self, features):
        """Estimate depth from features"""
        with torch.no_grad():
            depth_flat = self.depth_head(features)
            depth = depth_flat.view(1, self.image_height, self.image_width)
        return depth
```

## Performance Optimization

### Real-time VSLAM Optimization
```python
# Performance optimization for real-time VSLAM
import threading
import queue
import time
import numpy as np
from collections import deque

class OptimizedVSLAM:
    def __init__(self):
        # Processing pipeline
        self.input_queue = queue.Queue(maxsize=3)  # Limit input queue
        self.feature_queue = queue.Queue(maxsize=3)
        self.pose_queue = queue.Queue(maxsize=3)

        # Processing threads
        self.feature_thread = None
        self.pose_thread = None
        self.mapping_thread = None

        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.feature_times = deque(maxlen=30)
        self.pose_times = deque(maxlen=30)

        # Threading locks
        self.processing_lock = threading.Lock()
        self.running = False

        # Adaptive processing parameters
        self.target_fps = 30
        self.skip_frames = 0
        self.feature_count_target = 1000

    def start_processing(self):
        """Start multi-threaded VSLAM processing"""
        self.running = True

        # Start processing threads
        self.feature_thread = threading.Thread(target=self.feature_extraction_worker)
        self.pose_thread = threading.Thread(target=self.pose_estimation_worker)
        self.mapping_thread = threading.Thread(target=self.mapping_worker)

        self.feature_thread.start()
        self.pose_thread.start()
        self.mapping_thread.start()

    def stop_processing(self):
        """Stop VSLAM processing"""
        self.running = False

        if self.feature_thread:
            self.feature_thread.join()
        if self.pose_thread:
            self.pose_thread.join()
        if self.mapping_thread:
            self.mapping_thread.join()

    def process_frame_async(self, image, timestamp):
        """Asynchronously process a frame"""
        try:
            self.input_queue.put_nowait((image, timestamp))
        except queue.Full:
            # Drop frame if queue is full
            pass

    def feature_extraction_worker(self):
        """Worker thread for feature extraction"""
        while self.running:
            try:
                image, timestamp = self.input_queue.get(timeout=0.1)

                start_time = time.time()

                # Extract features
                keypoints, descriptors = self.extract_features_optimized(image)

                processing_time = time.time() - start_time
                self.feature_times.append(processing_time)

                # Put features in queue for pose estimation
                try:
                    self.feature_queue.put_nowait((keypoints, descriptors, timestamp))
                except queue.Full:
                    pass

            except queue.Empty:
                continue

    def pose_estimation_worker(self):
        """Worker thread for pose estimation"""
        prev_features = None

        while self.running:
            try:
                keypoints, descriptors, timestamp = self.feature_queue.get(timeout=0.1)

                start_time = time.time()

                # Estimate pose
                if prev_features is not None:
                    pose_change = self.estimate_pose_optimized(
                        prev_features, (keypoints, descriptors)
                    )

                    # Update global pose
                    self.update_global_pose(pose_change)

                processing_time = time.time() - start_time
                self.pose_times.append(processing_time)

                prev_features = (keypoints, descriptors)

            except queue.Empty:
                continue

    def extract_features_optimized(self, image):
        """Optimized feature extraction"""
        # Use optimized ORB parameters
        orb = cv2.ORB_create(
            nfeatures=min(1000, self.feature_count_target),
            scaleFactor=1.2,
            nlevels=4,  # Reduce levels for speed
            edgeThreshold=19,
            patchSize=19,
            fastThreshold=20
        )

        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def estimate_pose_optimized(self, prev_features, curr_features):
        """Optimized pose estimation"""
        prev_kp, prev_desc = prev_features
        curr_kp, curr_desc = curr_features

        if prev_desc is None or curr_desc is None:
            return np.eye(4)

        # Use BF matcher instead of FLANN for better performance on small datasets
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(curr_desc, prev_desc, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= 10:  # Minimum matches for pose estimation
            # Get corresponding points
            prev_pts = np.float32([prev_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([curr_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate motion
            E, mask = cv2.findEssentialMat(
                prev_pts, curr_pts, self.camera_matrix,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )

            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, self.camera_matrix)

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.ravel()
                return T

        return np.eye(4)

    def update_global_pose(self, pose_change):
        """Update global pose estimate"""
        with self.processing_lock:
            self.current_pose = self.current_pose @ np.linalg.inv(pose_change)

    def get_performance_metrics(self):
        """Get performance metrics"""
        if self.feature_times:
            avg_feature_time = sum(self.feature_times) / len(self.feature_times)
            avg_feature_fps = 1.0 / avg_feature_time if avg_feature_time > 0 else 0
        else:
            avg_feature_time = 0
            avg_feature_fps = 0

        if self.pose_times:
            avg_pose_time = sum(self.pose_times) / len(self.pose_times)
            avg_pose_fps = 1.0 / avg_pose_time if avg_pose_time > 0 else 0
        else:
            avg_pose_time = 0
            avg_pose_fps = 0

        return {
            'feature_processing_time': avg_feature_time,
            'pose_processing_time': avg_pose_time,
            'feature_fps': avg_feature_fps,
            'pose_fps': avg_pose_fps,
            'target_fps': self.target_fps
        }

    def adapt_parameters(self):
        """Adapt processing parameters based on performance"""
        metrics = self.get_performance_metrics()

        current_fps = min(metrics['feature_fps'], metrics['pose_fps'])

        if current_fps < self.target_fps * 0.8:
            # Reduce feature count to improve performance
            self.feature_count_target = max(500, int(self.feature_count_target * 0.9))
        elif current_fps > self.target_fps * 1.1:
            # Increase feature count for better accuracy
            self.feature_count_target = min(2000, int(self.feature_count_target * 1.1))
```

## Evaluation and Validation

### VSLAM Evaluation Metrics
```python
# VSLAM evaluation and validation
import numpy as np
from scipy.spatial.transform import Rotation as R

class VSLEvaluator:
    def __init__(self):
        self.estimated_trajectory = []
        self.ground_truth_trajectory = []
        self.timestamps = []

    def add_estimates(self, est_pose, gt_pose, timestamp):
        """Add pose estimates for evaluation"""
        self.estimated_trajectory.append(est_pose[:3, 3])  # Position only
        self.ground_truth_trajectory.append(gt_pose[:3, 3])
        self.timestamps.append(timestamp)

    def calculate_ate(self):
        """Calculate Absolute Trajectory Error"""
        if len(self.estimated_trajectory) < 2:
            return float('inf'), float('inf')

        est_traj = np.array(self.estimated_trajectory)
        gt_traj = np.array(self.ground_truth_trajectory)

        # Align trajectories using Umeyama algorithm
        est_aligned, R_align, t_align, s_align = self.align_trajectory(est_traj, gt_traj)

        # Calculate ATE
        errors = np.linalg.norm(est_aligned - gt_traj, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)

        return rmse, mean_error

    def calculate_rpe(self):
        """Calculate Relative Pose Error"""
        if len(self.estimated_trajectory) < 3:
            return float('inf'), float('inf')

        est_traj = np.array(self.estimated_trajectory)
        gt_traj = np.array(self.ground_truth_trajectory)

        # Calculate relative poses
        est_rel_poses = []
        gt_rel_poses = []

        for i in range(1, len(est_traj)):
            est_rel = est_traj[i] - est_traj[i-1]
            gt_rel = gt_traj[i] - gt_traj[i-1]

            est_rel_poses.append(est_rel)
            gt_rel_poses.append(gt_rel)

        est_rel = np.array(est_rel_poses)
        gt_rel = np.array(gt_rel_poses)

        # Calculate RPE
        errors = np.linalg.norm(est_rel - gt_rel, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)

        return rmse, mean_error

    def align_trajectory(self, est_traj, gt_traj):
        """Align estimated trajectory to ground truth using Umeyama algorithm"""
        # Calculate centroids
        est_centroid = np.mean(est_traj, axis=0)
        gt_centroid = np.mean(gt_traj, axis=0)

        # Center trajectories
        est_centered = est_traj - est_centroid
        gt_centered = gt_traj - gt_centroid

        # Calculate correlation matrix
        H = np.dot(gt_centered.T, est_centered)

        # Singular value decomposition
        U, S, Vt = np.linalg.svd(H)

        # Calculate rotation matrix
        R = np.dot(U, Vt)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(U, Vt)

        # Calculate scale
        var_a = np.mean(np.sum(est_centered**2, axis=1))
        scale = np.trace(np.dot(S, np.diag([1, 1, np.sign(np.linalg.det(R))]))) / var_a

        # Calculate translation
        t = gt_centroid - scale * np.dot(R, est_centroid)

        # Apply transformation
        est_aligned = scale * np.dot(est_traj, R.T) + t

        return est_aligned, R, t, scale

    def calculate_orientation_error(self):
        """Calculate orientation error between estimated and ground truth poses"""
        if len(self.estimated_trajectory) < 2:
            return float('inf'), float('inf')

        # This would require full pose matrices, not just positions
        # Implementation would compare rotation matrices/ quaternions
        pass

    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        ate_rmse, ate_mean = self.calculate_ate()
        rpe_rmse, rpe_mean = self.calculate_rpe()

        report = f"""
VSLAM Evaluation Report
=======================
Trajectory Length: {len(self.estimated_trajectory)} poses
Total Distance: {np.sum(np.linalg.norm(np.diff(self.ground_truth_trajectory, axis=0), axis=1)):.2f}m

Absolute Trajectory Error (ATE):
- RMSE: {ate_rmse:.4f}m
- Mean: {ate_mean:.4f}m
- Median: {np.median(np.linalg.norm(np.array(self.estimated_trajectory) - np.array(self.ground_truth_trajectory), axis=1)):.4f}m

Relative Pose Error (RPE):
- RMSE: {rpe_rmse:.4f}m
- Mean: {rpe_mean:.4f}m

Performance:
- Total poses processed: {len(self.estimated_trajectory)}
- Coverage: {(len(self.estimated_trajectory) / len(self.ground_truth_trajectory)) * 100 if self.ground_truth_trajectory else 0:.2f}%
        """

        return report
```

## Troubleshooting and Common Issues

### VSLAM Troubleshooting Guide
1. **Feature Depletion**: Increase ORB parameters or use alternative detectors
2. **Drift Accumulation**: Implement loop closure and global optimization
3. **Tracking Failure**: Use visual-inertial fusion for robustness
4. **Initialization Issues**: Ensure sufficient scene texture and motion
5. **Performance Bottlenecks**: Optimize feature detection and matching

### Robustness Considerations
- **Lighting Changes**: Use illumination-invariant features
- **Motion Blur**: Implement blur detection and rejection
- **Dynamic Objects**: Detect and exclude moving objects
- **Scale Ambiguity**: Use stereo or IMU for scale recovery
- **Degenerate Motions**: Detect pure rotation or forward motion

## Practical Lab: VSLAM Implementation

### Lab Objective
Implement a complete VSLAM system that integrates with Isaac Sim and ROS 2, including feature-based tracking, pose estimation, and trajectory visualization.

### Implementation Steps
1. Set up Isaac Sim environment with camera-equipped robot
2. Implement ORB-SLAM algorithm with keyframe management
3. Integrate with ROS 2 for pose publishing and visualization
4. Test system in various environments and lighting conditions
5. Evaluate performance using ground truth from Isaac Sim

### Expected Outcome
- Working VSLAM system with real-time performance
- ROS 2 integration with proper TF frames
- Visualization of trajectory and map
- Performance evaluation against ground truth

## Review Questions

1. Explain the main differences between feature-based and direct VSLAM methods.
2. How do you handle scale ambiguity in monocular VSLAM systems?
3. What are the key components of the VSLAM pipeline and their functions?
4. Describe the process for evaluating VSLAM system performance.
5. How do you integrate VSLAM with Isaac Sim for ground truth validation?

## Next Steps
After mastering VSLAM systems, students should proceed to:
- Advanced navigation for humanoid robots
- Sim-to-real transfer techniques
- Vision-Language-Action system integration
- Deep learning enhanced perception

This comprehensive guide to VSLAM systems provides the foundation for implementing sophisticated visual navigation and mapping capabilities essential for Physical AI and Humanoid Robotics applications.