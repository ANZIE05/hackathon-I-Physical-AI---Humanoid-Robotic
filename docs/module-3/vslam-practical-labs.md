---
sidebar_position: 5
---

# VSLAM Practical Labs

## Overview
This section provides hands-on practical laboratories for implementing and validating Visual Simultaneous Localization and Mapping (VSLAM) systems. The labs progress from basic concepts to advanced integration with Isaac Sim and ROS 2 for Physical AI and Humanoid Robotics applications.

## Lab 1: Basic Feature Detection and Tracking

### Objective
Implement fundamental computer vision components for VSLAM including feature detection, description, and tracking using ORB features.

### Prerequisites
- Basic Python and OpenCV knowledge
- Understanding of feature detection concepts
- ROS 2 Humble installation
- Basic familiarity with Isaac Sim concepts

### Implementation Steps

#### Step 1: Feature Detection Implementation
```python
# vslam_lab_1/feature_detector.py
import cv2
import numpy as np
from typing import List, Tuple, Optional

class FeatureDetector:
    def __init__(self, max_features: int = 1000, scale_factor: float = 1.2):
        """
        Initialize feature detector with ORB

        Args:
            max_features: Maximum number of features to detect
            scale_factor: Pyramid scale factor
        """
        self.max_features = max_features
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=20
        )

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect and compute ORB features for an image

        Args:
            image: Input image (grayscale or color)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None:
            # Return empty descriptors if no features detected
            return [], np.array([]).reshape(0, 0)

        return keypoints, descriptors

    def visualize_features(self, image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """
        Visualize detected features on image

        Args:
            image: Input image
            keypoints: Detected keypoints

        Returns:
            Image with features drawn
        """
        if len(image.shape) == 2:
            # Convert grayscale to color for visualization
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw keypoints
        result = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        return result

class FeatureTracker:
    def __init__(self, match_threshold: float = 0.75):
        """
        Initialize feature tracker

        Args:
            match_threshold: Lowe's ratio test threshold
        """
        self.match_threshold = match_threshold
        self.flann = cv2.FlannBasedMatcher(
            {'algorithm': 6, 'table_number': 6, 'key_size': 12, 'multi_probe_level': 1},
            {'checks': 50}
        )

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two descriptor sets

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image

        Returns:
            List of good matches after Lowe's ratio test
        """
        if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
            return []

        try:
            # Find 2 nearest neighbors for each descriptor
            matches = self.flann.knnMatch(desc1, desc2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_threshold * n.distance:
                        good_matches.append(m)

            return good_matches
        except cv2.error:
            # Fallback to brute force matcher if FLANN fails
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(desc1, desc2, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_threshold * n.distance:
                        good_matches.append(m)

            return good_matches

    def track_features(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                      matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matched keypoint coordinates

        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Good matches

        Returns:
            Tuple of (pts1, pts2) - matched point coordinates
        """
        if len(matches) == 0:
            return np.array([]), np.array([])

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        return pts1, pts2

def main():
    """Main function for testing feature detection and tracking"""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python feature_detector.py <image_path>")
        sys.exit(1)

    # Load image
    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)

    # Initialize feature detector
    detector = FeatureDetector(max_features=500)

    # Detect and compute features
    keypoints, descriptors = detector.detect_and_compute(image)

    print(f"Detected {len(keypoints)} features")

    # Visualize features
    feature_image = detector.visualize_features(image, keypoints)

    # Display result
    cv2.imshow('Detected Features', feature_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

#### Step 2: Feature Matching and Validation
```python
# vslam_lab_1/matcher_validator.py
import cv2
import numpy as np
from feature_detector import FeatureDetector, FeatureTracker

class FeatureMatcherValidator:
    def __init__(self):
        self.detector = FeatureDetector(max_features=1000)
        self.tracker = FeatureTracker(match_threshold=0.75)

    def validate_feature_matching(self, image1: np.ndarray, image2: np.ndarray) -> dict:
        """
        Validate feature matching between two images

        Args:
            image1: First image
            image2: Second image (similar scene)

        Returns:
            Dictionary with validation metrics
        """
        # Detect features in both images
        kp1, desc1 = self.detector.detect_and_compute(image1)
        kp2, desc2 = self.detector.detect_and_compute(image2)

        # Match features
        matches = self.tracker.match_features(desc1, desc2)

        # Calculate validation metrics
        metrics = {
            'num_features_img1': len(kp1),
            'num_features_img2': len(kp2),
            'num_matches': len(matches),
            'match_ratio': len(matches) / max(len(kp1), len(kp2), 1) if len(kp1) > 0 or len(kp2) > 0 else 0,
            'match_quality': self.assess_match_quality(matches, kp1, kp2)
        }

        return metrics

    def assess_match_quality(self, matches: List[cv2.DMatch], kp1: List[cv2.KeyPoint],
                           kp2: List[cv2.KeyPoint]) -> float:
        """
        Assess the quality of feature matches

        Args:
            matches: Good matches
            kp1: Keypoints from first image
            kp2: Keypoints from second image

        Returns:
            Quality score (0-1)
        """
        if len(matches) < 10:  # Need minimum matches for meaningful assessment
            return 0.0

        # Calculate average match distance
        avg_distance = np.mean([m.distance for m in matches])

        # Calculate spatial distribution of matches
        if len(matches) > 1:
            pts1, pts2 = self.tracker.track_features(kp1, kp2, matches)

            # Calculate variance of match locations (should be spread out)
            if pts1.size > 0:
                variance1 = np.var(pts1, axis=0).mean()
                variance2 = np.var(pts2, axis=0).mean()
                spatial_score = min(variance1, variance2) / 1000.0  # Normalize
            else:
                spatial_score = 0.0
        else:
            spatial_score = 0.0

        # Combine metrics (simple heuristic)
        # Lower average distance is better (higher quality)
        distance_score = max(0, 1.0 - (avg_distance / 100.0))  # Assume 100 is high distance
        overall_score = 0.6 * distance_score + 0.4 * spatial_score

        return min(1.0, max(0.0, overall_score))

    def visualize_matches(self, image1: np.ndarray, image2: np.ndarray,
                        kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                        matches: List[cv2.DMatch]) -> np.ndarray:
        """
        Visualize feature matches between two images

        Args:
            image1: First image
            image2: Second image
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Good matches

        Returns:
            Image with matches drawn
        """
        # Convert to color if grayscale
        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        # Create side-by-side image
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        vis_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_image[:h1, :w1] = image1
        vis_image[:h2, w1:w1+w2] = image2

        # Draw matches
        for match in matches[:50]:  # Limit number of matches shown
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            pt2_offset = (pt2[0] + w1, pt2[1])  # Offset for second image

            # Draw line and points
            cv2.line(vis_image, pt1, pt2_offset, (0, 255, 0), 1)
            cv2.circle(vis_image, pt1, 3, (255, 0, 0), -1)
            cv2.circle(vis_image, pt2_offset, 3, (255, 0, 0), -1)

        return vis_image

def test_feature_validation():
    """Test function for feature validation"""
    import sys

    if len(sys.argv) != 3:
        print("Usage: python matcher_validator.py <image1_path> <image2_path>")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Could not load images")
        sys.exit(1)

    validator = FeatureMatcherValidator()

    # Validate feature matching
    metrics = validator.validate_feature_matching(image1, image2)

    print("Feature Matching Validation Results:")
    print(f"- Features in image 1: {metrics['num_features_img1']}")
    print(f"- Features in image 2: {metrics['num_features_img2']}")
    print(f"- Number of matches: {metrics['num_matches']}")
    print(f"- Match ratio: {metrics['match_ratio']:.3f}")
    print(f"- Match quality: {metrics['match_quality']:.3f}")

    # Show visualization if matches exist
    if metrics['num_matches'] > 0:
        kp1, desc1 = validator.detector.detect_and_compute(image1)
        kp2, desc2 = validator.detector.detect_and_compute(image2)
        matches = validator.tracker.match_features(desc1, desc2)

        vis_image = validator.visualize_matches(image1, image2, kp1, kp2, matches)
        cv2.imshow('Feature Matches', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_feature_validation()
```

### Lab Exercise 1: Feature Detection Optimization
1. Test the feature detector with different image types (indoor/outdoor, textured/plain)
2. Experiment with different ORB parameters (nfeatures, scaleFactor, edgeThreshold)
3. Analyze the trade-off between feature count and matching quality
4. Implement a feature density optimization algorithm

### Expected Results
- Working feature detection and matching system
- Understanding of ORB parameters and their effects
- Ability to visualize and validate feature matches
- Recognition of scenarios where feature detection works well/poorly

## Lab 2: Pose Estimation and Essential Matrix

### Objective
Implement pose estimation from feature correspondences using essential matrix decomposition and RANSAC for robust estimation.

### Implementation Steps

#### Step 1: Essential Matrix Estimation
```python
# vslam_lab_2/pose_estimator.py
import cv2
import numpy as np
from typing import Tuple, Optional
from feature_detector import FeatureDetector, FeatureTracker

class PoseEstimator:
    def __init__(self, camera_matrix: np.ndarray):
        """
        Initialize pose estimator with camera calibration

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.K = camera_matrix
        self.ransac_threshold = 1.0  # Reprojection error threshold
        self.min_inliers = 10        # Minimum inliers for valid pose

    def estimate_pose_from_features(self,
                                  pts1: np.ndarray,
                                  pts2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Estimate relative pose from feature correspondences using essential matrix

        Args:
            pts1: Points in first image (Nx2)
            pts2: Points in second image (Nx2)

        Returns:
            Tuple of (rotation, translation, inlier_mask) or (None, None, empty_mask)
        """
        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            return None, None, np.array([], dtype=bool)

        try:
            # Estimate essential matrix using RANSAC
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.ransac_threshold
            )

            if E is None or E.size == 0:
                return None, None, np.array([], dtype=bool)

            # Recover pose from essential matrix
            _, R, t, mask_new = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

            # Combine rotation and translation into transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()

            # Create inlier mask
            if mask is not None and mask_new is not None:
                inlier_mask = (mask.flatten() > 0) & (mask_new.flatten() > 0)
            elif mask is not None:
                inlier_mask = mask.flatten() > 0
            elif mask_new is not None:
                inlier_mask = mask_new.flatten() > 0
            else:
                inlier_mask = np.ones(pts1.shape[0], dtype=bool)

            return R, t, inlier_mask

        except cv2.error:
            return None, None, np.array([], dtype=bool)

    def triangulate_points(self,
                          R1: np.ndarray, t1: np.ndarray,
                          R2: np.ndarray, t2: np.ndarray,
                          pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences

        Args:
            R1, t1: Pose of first camera
            R2, t2: Pose of second camera
            pts1: Points in first image
            pts2: Points in second image

        Returns:
            3D points (Nx3)
        """
        # Create projection matrices
        P1 = self.K @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = self.K @ np.hstack([R2, t2.reshape(3, 1)])

        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert from homogeneous to Euclidean coordinates
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T

    def compute_reprojection_error(self,
                                 points_3d: np.ndarray,
                                 R: np.ndarray, t: np.ndarray,
                                 observed_points: np.ndarray) -> float:
        """
        Compute reprojection error for 3D points

        Args:
            points_3d: 3D points (Nx3)
            R: Rotation matrix
            t: Translation vector
            observed_points: Observed 2D points (Nx2)

        Returns:
            Average reprojection error
        """
        # Project 3D points to 2D
        projected_points = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            R, t, self.K, None
        )[0].reshape(-1, 2)

        # Calculate reprojection error
        errors = np.linalg.norm(projected_points - observed_points, axis=1)
        avg_error = np.mean(errors)

        return avg_error

    def validate_pose_estimation(self, R: np.ndarray, t: np.ndarray) -> bool:
        """
        Validate estimated pose for physical plausibility

        Args:
            R: Rotation matrix
            t: Translation vector

        Returns:
            True if pose is physically plausible
        """
        # Check rotation matrix properties
        det_R = np.linalg.det(R)
        orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))

        # Check translation magnitude (reasonable for robot motion)
        translation_magnitude = np.linalg.norm(t)

        # Validation criteria
        valid_rotation = abs(det_R - 1.0) < 0.1 and orthogonality_error < 0.1
        reasonable_translation = 0.01 < translation_magnitude < 10.0  # Between 1cm and 10m

        return valid_rotation and reasonable_translation

class VSLAMPipeline:
    def __init__(self, camera_matrix: np.ndarray):
        self.camera_matrix = camera_matrix
        self.detector = FeatureDetector(max_features=1000)
        self.tracker = FeatureTracker(match_threshold=0.75)
        self.pose_estimator = PoseEstimator(camera_matrix)

        # State variables
        self.current_pose = np.eye(4)
        self.keyframes = []
        self.map_points = []

    def process_frame_pair(self, image1: np.ndarray, image2: np.ndarray) -> dict:
        """
        Process a pair of frames to estimate relative motion

        Args:
            image1: First image
            image2: Second image (subsequent frame)

        Returns:
            Dictionary with processing results
        """
        # Extract features from both images
        kp1, desc1 = self.detector.detect_and_compute(image1)
        kp2, desc2 = self.detector.detect_and_compute(image2)

        if len(kp1) < 10 or len(kp2) < 10:
            return {
                'success': False,
                'reason': 'Insufficient features',
                'num_features1': len(kp1),
                'num_features2': len(kp2)
            }

        # Match features
        matches = self.tracker.match_features(desc1, desc2)

        if len(matches) < 10:
            return {
                'success': False,
                'reason': 'Insufficient matches',
                'num_matches': len(matches)
            }

        # Extract matched points
        pts1, pts2 = self.tracker.track_features(kp1, kp2, matches)

        if pts1.size < 8:
            return {
                'success': False,
                'reason': 'Insufficient inliers after matching',
                'num_points': pts1.size // 2
            }

        # Estimate relative pose
        R_rel, t_rel, inlier_mask = self.pose_estimator.estimate_pose_from_features(
            pts1[inlier_mask], pts2[inlier_mask]
        )

        if R_rel is None or t_rel is None:
            return {
                'success': False,
                'reason': 'Pose estimation failed',
                'num_inliers': inlier_mask.sum()
            }

        # Validate pose
        is_valid = self.pose_estimator.validate_pose_estimation(R_rel, t_rel)

        if not is_valid:
            return {
                'success': False,
                'reason': 'Estimated pose is not physically plausible',
                'R': R_rel,
                't': t_rel
            }

        # Create transformation matrix
        T_rel = np.eye(4)
        T_rel[:3, :3] = R_rel
        T_rel[:3, 3] = t_rel.ravel()

        # Update global pose
        self.current_pose = self.current_pose @ T_rel

        # Calculate results
        results = {
            'success': True,
            'relative_pose': T_rel,
            'absolute_pose': self.current_pose,
            'num_matches': len(matches),
            'num_inliers': inlier_mask.sum(),
            'inlier_ratio': inlier_mask.sum() / len(matches),
            'translation_magnitude': np.linalg.norm(t_rel),
            'rotation_angle': self.rotation_matrix_to_angle(R_rel)
        }

        return results

    def rotation_matrix_to_angle(self, R: np.ndarray) -> float:
        """Convert rotation matrix to rotation angle in radians"""
        trace = np.trace(R)
        angle = np.arccos(max(-1, min(1, (trace - 1) / 2)))
        return angle

def test_pose_estimation():
    """Test function for pose estimation"""
    # Example camera matrix (replace with actual calibration)
    K = np.array([
        [525.0, 0.0, 319.5],
        [0.0, 525.0, 239.5],
        [0.0, 0.0, 1.0]
    ])

    # Initialize pipeline
    pipeline = VSLAMPipeline(K)

    import sys
    if len(sys.argv) != 3:
        print("Usage: python pose_estimator.py <image1_path> <image2_path>")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Could not load images")
        sys.exit(1)

    # Process frame pair
    results = pipeline.process_frame_pair(image1, image2)

    print("Pose Estimation Results:")
    if results['success']:
        print(f"- Relative pose computed successfully")
        print(f"- Translation magnitude: {results['translation_magnitude']:.3f}")
        print(f"- Rotation angle: {np.degrees(results['rotation_angle']):.2f} degrees")
        print(f"- Inlier ratio: {results['inlier_ratio']:.3f}")
        print(f"- Number of inliers: {results['num_inliers']}")
        print(f"- Absolute pose:\n{results['absolute_pose']}")
    else:
        print(f"- Failed: {results['reason']}")

if __name__ == "__main__":
    test_pose_estimation()
```

#### Step 2: Bundle Adjustment Implementation
```python
# vslam_lab_2/bundle_adjustment.py
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from typing import List, Dict, Tuple

class BundleAdjustment:
    def __init__(self):
        self.max_iterations = 50
        self.robust_loss = True  # Use robust loss function (Huber)

    def residual_function(self, params,
                         points_3d_indices,
                         camera_indices,
                         points_2d,
                         camera_matrix):
        """
        Residual function for bundle adjustment optimization

        Args:
            params: Flattened array of [camera_params, points_3d_params]
            points_3d_indices: Indices of 3D points for each observation
            camera_indices: Indices of cameras for each observation
            points_2d: Observed 2D points
            camera_matrix: Camera intrinsic matrix

        Returns:
            Flattened array of residuals
        """
        n_cameras = len(np.unique(camera_indices))
        n_points = len(np.unique(points_3d_indices))

        # Reshape parameters
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))  # [R (3), t (3)]
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        residuals = []

        for i in range(len(points_2d)):
            camera_idx = camera_indices[i]
            point_idx = points_3d_indices[i]

            # Extract camera pose (Rodrigues vector + translation)
            rvec = camera_params[camera_idx, :3]
            tvec = camera_params[camera_idx, 3:]

            # Get 3D point
            point_3d = points_3d[point_idx]

            # Project 3D point to 2D
            projected, _ = cv2.projectPoints(
                point_3d.reshape(1, 1, 3),
                rvec, tvec,
                camera_matrix,
                None
            )

            projected = projected[0, 0]
            observed = points_2d[i]

            # Calculate residual
            residual = projected - observed
            residuals.extend(residual)

        return np.array(residuals)

    def run_bundle_adjustment(self,
                             camera_poses: List[np.ndarray],
                             points_3d: List[np.ndarray],
                             observations: List[Dict],
                             camera_matrix: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run bundle adjustment optimization

        Args:
            camera_poses: List of 4x4 camera poses [R|t]
            points_3d: List of 3D points
            observations: List of observations (camera_idx, point_idx, point_2d)
            camera_matrix: Camera intrinsic matrix

        Returns:
            Tuple of (optimized_camera_poses, optimized_points_3d)
        """
        n_cameras = len(camera_poses)
        n_points = len(points_3d)

        # Convert camera poses to parameter form [R (Rodrigues), t]
        camera_params = []
        for pose in camera_poses:
            R = pose[:3, :3]
            t = pose[:3, 3]

            # Convert rotation matrix to Rodrigues vector
            rvec, _ = cv2.Rodrigues(R)
            params = np.concatenate([rvec.ravel(), t.ravel()])
            camera_params.append(params)

        camera_params = np.array(camera_params).flatten()

        # Prepare points and observations
        points_3d_array = np.array(points_3d)
        points_3d_flat = points_3d_array.flatten()

        # Combine all parameters
        all_params = np.concatenate([camera_params, points_3d_flat])

        # Prepare observation data
        points_3d_indices = np.array([obs['point_idx'] for obs in observations])
        camera_indices = np.array([obs['camera_idx'] for obs in observations])
        points_2d = np.array([obs['point_2d'] for obs in observations])

        # Define Jacobian sparsity structure (sparse optimization)
        def jac_sparsity(n_cameras, n_points, observations):
            m = len(observations) * 2  # Each observation contributes 2 residuals (x, y)
            n = n_cameras * 6 + n_points * 3  # 6 params per camera, 3 per point

            S = lil_matrix((m, n), dtype=int)

            for i, obs in enumerate(observations):
                camera_idx = obs['camera_idx']
                point_idx = obs['point_idx']

                # Residuals for this observation (2 residuals: x, y)
                for residual_j in range(2):  # x and y residuals
                    # Camera parameters (6 parameters)
                    for param_k in range(6):
                        S[2*i + residual_j, camera_idx * 6 + param_k] = 1

                    # Point parameters (3 parameters)
                    for param_k in range(3):
                        S[2*i + residual_j, n_cameras * 6 + point_idx * 3 + param_k] = 1

            return S

        # Create sparsity matrix
        sparsity = jac_sparsity(n_cameras, n_points, observations)

        # Run optimization
        result = least_squares(
            self.residual_function,
            all_params,
            jac_sparsity=sparsity,
            method='trf',
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=200,
            args=(points_3d_indices, camera_indices, points_2d, camera_matrix)
        )

        # Extract optimized parameters
        optimized_camera_params = result.x[:n_cameras * 6].reshape((n_cameras, 6))
        optimized_points_3d = result.x[n_cameras * 6:].reshape((n_points, 3))

        # Convert back to poses
        optimized_poses = []
        for params in optimized_camera_params:
            rvec = params[:3]
            tvec = params[3:]

            # Convert Rodrigues vector back to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = tvec
            optimized_poses.append(pose)

        return optimized_poses, optimized_points_3d.tolist()

    def optimize_local_window(self,
                             keyframes: List[Dict],
                             local_map: List[Dict],
                             camera_matrix: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Optimize a local window of keyframes and associated map points

        Args:
            keyframes: List of keyframes with poses and observations
            local_map: List of local map points
            camera_matrix: Camera intrinsic matrix

        Returns:
            Tuple of (optimized_keyframes, optimized_local_map)
        """
        # Prepare data for bundle adjustment
        camera_poses = [kf['pose'] for kf in keyframes]
        points_3d = [mp['coordinates'] for mp in local_map]

        # Collect observations
        observations = []
        for i, kf in enumerate(keyframes):
            for obs in kf.get('observations', []):
                # Find corresponding map point index
                for j, mp in enumerate(local_map):
                    if mp['id'] == obs['map_point_id']:
                        observations.append({
                            'camera_idx': i,
                            'point_idx': j,
                            'point_2d': obs['pixel_coordinates']
                        })
                        break

        # Run bundle adjustment
        optimized_poses, optimized_points = self.run_bundle_adjustment(
            camera_poses, points_3d, observations, camera_matrix
        )

        # Update keyframes with optimized poses
        optimized_keyframes = []
        for i, (orig_kf, opt_pose) in enumerate(zip(keyframes, optimized_poses)):
            new_kf = orig_kf.copy()
            new_kf['pose'] = opt_pose
            optimized_keyframes.append(new_kf)

        # Update map points with optimized coordinates
        optimized_local_map = []
        for i, (orig_mp, opt_coords) in enumerate(zip(local_map, optimized_points)):
            new_mp = orig_mp.copy()
            new_mp['coordinates'] = opt_coords
            optimized_local_map.append(new_mp)

        return optimized_keyframes, optimized_local_map
```

### Lab Exercise 2: Pose Estimation Evaluation
1. Test pose estimation on different types of motion (rotation, translation, combined)
2. Evaluate the impact of RANSAC parameters on pose estimation accuracy
3. Implement and compare different essential matrix estimation methods
4. Analyze the relationship between feature quality and pose accuracy

### Expected Results
- Working pose estimation system
- Understanding of essential matrix decomposition
- Ability to evaluate pose estimation quality
- Implementation of bundle adjustment for optimization

## Lab 3: Map Building and Loop Closure

### Objective
Implement map building with 3D point cloud construction and loop closure detection for map consistency.

### Implementation Steps

#### Step 1: Map Building and Point Cloud Construction
```python
# vslam_lab_3/map_builder.py
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import pickle

class MapPoint:
    def __init__(self, coordinates: np.ndarray, color: np.ndarray = None,
                 normal: np.ndarray = None, id: int = None):
        self.coordinates = coordinates  # 3D coordinates [x, y, z]
        self.color = color              # RGB color [r, g, b] (optional)
        self.normal = normal           # Surface normal (optional)
        self.id = id                   # Unique identifier
        self.observations = []         # List of observations [camera_pose, pixel_coordinates]
        self.descriptor = None         # Descriptor for loop closure
        self.tracking_count = 0        # Number of successful trackings
        self.first_observation_frame = 0  # Frame when first observed
        self.last_observation_frame = 0   # Frame when last observed

class MapBuilder:
    def __init__(self, max_map_size: int = 10000,
                 min_triangulation_angle: float = 5.0,
                 max_reprojection_error: float = 2.0):
        """
        Initialize map builder

        Args:
            max_map_size: Maximum number of map points to maintain
            min_triangulation_angle: Minimum angle for stable triangulation (degrees)
            max_reprojection_error: Maximum reprojection error for valid points (pixels)
        """
        self.max_map_size = max_map_size
        self.min_triangulation_angle = np.radians(min_triangulation_angle)
        self.max_reprojection_error = max_reprojection_error

        self.map_points: List[MapPoint] = []
        self.next_point_id = 0
        self.kdtree = None
        self.dirty = True  # Flag indicating if KD-tree needs rebuilding

    def triangulate_point(self,
                         pose1: np.ndarray,
                         pose2: np.ndarray,
                         point1: np.ndarray,
                         point2: np.ndarray,
                         camera_matrix: np.ndarray) -> Optional[np.ndarray]:
        """
        Triangulate a 3D point from two camera views

        Args:
            pose1, pose2: 4x4 camera poses
            point1, point2: 2D points in respective images
            camera_matrix: 3x3 camera intrinsic matrix

        Returns:
            3D coordinates of triangulated point or None if invalid
        """
        # Extract rotation and translation from poses
        R1, t1 = pose1[:3, :3], pose1[:3, 3]
        R2, t2 = pose2[:3, :3], pose2[:3, 3]

        # Create projection matrices
        P1 = camera_matrix @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = camera_matrix @ np.hstack([R2, t2.reshape(3, 1)])

        # Triangulate point
        point_4d = cv2.triangulatePoints(P1, P2, point1.reshape(2, 1), point2.reshape(2, 1))

        if point_4d[3, 0] == 0:
            return None

        # Convert to 3D coordinates
        point_3d = (point_4d[:3, 0] / point_4d[3, 0]).flatten()

        # Check triangulation angle for stability
        baseline = t2 - t1
        ray1 = point_3d - t1
        ray2 = point_3d - t2

        if np.linalg.norm(ray1) == 0 or np.linalg.norm(ray2) == 0:
            return None

        # Calculate angle between viewing rays
        cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(abs(cos_angle))

        if angle < self.min_triangulation_angle:
            return None  # Too small angle for stable triangulation

        # Check if point is in front of both cameras
        if not self.is_point_in_front_of_camera(R1, t1, point_3d) or \
           not self.is_point_in_front_of_camera(R2, t2, point_3d):
            return None

        return point_3d

    def is_point_in_front_of_camera(self, R: np.ndarray, t: np.ndarray, point_3d: np.ndarray) -> bool:
        """Check if 3D point is in front of camera"""
        # Transform point to camera coordinate system
        cam_coords = R @ point_3d + t
        return cam_coords[2] > 0  # Point is in front if z > 0

    def add_observation(self, camera_pose: np.ndarray,
                       pixel_coordinates: np.ndarray,
                       point_3d: np.ndarray,
                       descriptor: np.ndarray = None,
                       color: np.ndarray = None) -> int:
        """
        Add a new map point or associate with existing point

        Args:
            camera_pose: 4x4 camera pose
            pixel_coordinates: 2D pixel coordinates
            point_3d: 3D coordinates of the point
            descriptor: Feature descriptor (optional)
            color: RGB color of the point (optional)

        Returns:
            ID of the map point
        """
        # Check if this point is already in the map (by proximity)
        existing_point_id = self.find_closest_point(point_3d)

        if existing_point_id is not None:
            # Associate with existing point
            point = self.map_points[existing_point_id]
            point.observations.append({
                'camera_pose': camera_pose.copy(),
                'pixel_coordinates': pixel_coordinates.copy()
            })
            point.tracking_count += 1
            point.last_observation_frame += 1
            return existing_point_id
        else:
            # Create new map point
            new_point = MapPoint(
                coordinates=point_3d,
                color=color,
                id=self.next_point_id,
                descriptor=descriptor
            )

            new_point.observations.append({
                'camera_pose': camera_pose.copy(),
                'pixel_coordinates': pixel_coordinates.copy()
            })
            new_point.tracking_count = 1
            new_point.first_observation_frame = 0
            new_point.last_observation_frame = 0

            self.map_points.append(new_point)
            point_id = self.next_point_id
            self.next_point_id += 1

            self.dirty = True  # Mark KD-tree as needing rebuild

            return point_id

    def find_closest_point(self, point_3d: np.ndarray, threshold: float = 0.05) -> Optional[int]:
        """
        Find closest existing map point to given 3D coordinates

        Args:
            point_3d: 3D coordinates to match
            threshold: Distance threshold for considering points as same

        Returns:
            ID of closest point if within threshold, else None
        """
        if not self.map_points:
            return None

        if self.dirty:
            self.rebuild_kdtree()

        # Query KD-tree for nearest neighbor
        distances, indices = self.kdtree.query([point_3d], k=1)

        if distances[0] < threshold:
            return indices[0]
        else:
            return None

    def rebuild_kdtree(self):
        """Rebuild KD-tree for efficient nearest neighbor search"""
        if self.map_points:
            points_array = np.array([mp.coordinates for mp in self.map_points])
            self.kdtree = KDTree(points_array)
            self.dirty = False
        else:
            self.kdtree = None
            self.dirty = False

    def update_map_point_colors(self, camera_pose: np.ndarray,
                               image: np.ndarray,
                               camera_matrix: np.ndarray):
        """
        Update colors of map points based on latest image

        Args:
            camera_pose: Current camera pose
            image: Current image
            camera_matrix: Camera intrinsic matrix
        """
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]

        for point in self.map_points:
            # Project 3D point to 2D
            projected, _ = cv2.projectPoints(
                point.coordinates.reshape(1, 1, 3),
                cv2.Rodrigues(R)[0],
                t,
                camera_matrix,
                None
            )

            x, y = int(projected[0, 0, 0]), int(projected[0, 0, 1])

            # Check if projection is within image bounds
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                # Get color from image
                if len(image.shape) == 3:  # Color image
                    color = image[y, x]
                    if point.color is None:
                        point.color = color.astype(float)
                    else:
                        # Average with existing color
                        point.color = 0.7 * point.color + 0.3 * color.astype(float)

    def prune_bad_points(self):
        """Remove bad map points based on various criteria"""
        good_points = []

        for point in self.map_points:
            # Criteria for keeping points:
            # 1. Sufficient number of observations
            # 2. Good reprojection error
            # 3. Not too old without observation

            if (len(point.observations) >= 2 and
                point.tracking_count >= 3 and
                point.last_observation_frame >= point.first_observation_frame):

                # Calculate average reprojection error
                avg_error = self.calculate_reprojection_error(point)
                if avg_error <= self.max_reprojection_error:
                    good_points.append(point)

        self.map_points = good_points
        self.dirty = True

    def calculate_reprojection_error(self, point: MapPoint) -> float:
        """Calculate average reprojection error for a map point"""
        if len(point.observations) < 2:
            return float('inf')

        errors = []
        for obs in point.observations:
            pose = obs['camera_pose']
            expected_pixel = obs['pixel_coordinates']

            # Project 3D point to 2D
            R = pose[:3, :3]
            t = pose[:3, 3]

            projected, _ = cv2.projectPoints(
                point.coordinates.reshape(1, 1, 3),
                cv2.Rodrigues(R)[0],
                t,
                self.camera_matrix,
                None
            )

            actual_pixel = projected[0, 0]
            error = np.linalg.norm(actual_pixel - expected_pixel)
            errors.append(error)

        return np.mean(errors) if errors else float('inf')

    def get_triangulated_points(self) -> np.ndarray:
        """Get all triangulated map points as numpy array"""
        if not self.map_points:
            return np.array([]).reshape(0, 3)

        return np.array([point.coordinates for point in self.map_points])

    def save_map(self, filepath: str):
        """Save map to file"""
        map_data = {
            'points': [
                {
                    'coordinates': point.coordinates,
                    'color': point.color,
                    'id': point.id,
                    'tracking_count': point.tracking_count,
                    'observations': point.observations
                }
                for point in self.map_points
            ],
            'next_point_id': self.next_point_id
        }

        with open(filepath, 'wb') as f:
            pickle.dump(map_data, f)

    def load_map(self, filepath: str):
        """Load map from file"""
        with open(filepath, 'rb') as f:
            map_data = pickle.load(f)

        self.map_points = []
        for point_data in map_data['points']:
            point = MapPoint(
                coordinates=point_data['coordinates'],
                color=point_data['color'],
                id=point_data['id']
            )
            point.tracking_count = point_data['tracking_count']
            point.observations = point_data['observations']
            self.map_points.append(point)

        self.next_point_id = map_data['next_point_id']
        self.dirty = True
```

#### Step 2: Loop Closure Detection
```python
# vslam_lab_3/loop_closure.py
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Optional
import time

class LoopClosureDetector:
    def __init__(self,
                 db_size: int = 1000,
                 min_loop_matches: int = 20,
                 max_pose_distance: float = 5.0,
                 max_angle_difference: float = 30.0):
        """
        Initialize loop closure detector

        Args:
            db_size: Maximum size of place recognition database
            min_loop_matches: Minimum matches required for loop closure
            max_pose_distance: Maximum distance for potential loop closure (meters)
            max_angle_difference: Maximum angle difference for potential loop closure (degrees)
        """
        self.db_size = db_size
        self.min_loop_matches = min_loop_matches
        self.max_pose_distance = max_pose_distance
        self.max_angle_difference = np.radians(max_angle_difference)

        # Place recognition database
        self.place_descriptors = []  # List of keyframe descriptors
        self.place_poses = []        # Corresponding poses
        self.place_timestamps = []   # Timestamps
        self.place_ids = []          # Unique IDs

        # Bag-of-Words vocabulary (simplified approach)
        self.vocabulary = None
        self.vocabulary_initialized = False

        # Matching parameters
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.loop_candidates = []
        self.loop_threshold = 0.7   # Descriptor similarity threshold

    def add_keyframe(self, descriptor: np.ndarray, pose: np.ndarray,
                     timestamp: float, keyframe_id: int):
        """
        Add a keyframe to the place recognition database

        Args:
            descriptor: Feature descriptor for the keyframe
            pose: Camera pose at this keyframe
            timestamp: Timestamp of the keyframe
            keyframe_id: Unique ID for the keyframe
        """
        if len(self.place_descriptors) >= self.db_size:
            # Remove oldest entry
            self.place_descriptors.pop(0)
            self.place_poses.pop(0)
            self.place_timestamps.pop(0)
            self.place_ids.pop(0)

        self.place_descriptors.append(descriptor)
        self.place_poses.append(pose)
        self.place_timestamps.append(timestamp)
        self.place_ids.append(keyframe_id)

    def detect_loop_closure(self, current_descriptor: np.ndarray,
                           current_pose: np.ndarray) -> Optional[Tuple[int, np.ndarray, float]]:
        """
        Detect potential loop closure with previous keyframes

        Args:
            current_descriptor: Descriptor of current frame
            current_pose: Current camera pose

        Returns:
            Tuple of (keyframe_id, relative_transform, confidence) if loop detected, else None
        """
        if len(self.place_descriptors) < 2:
            return None

        # Find potential matches in database
        candidates = self.find_place_candidates(current_descriptor, current_pose)

        if not candidates:
            return None

        # Verify each candidate using geometric consistency
        for candidate_id, candidate_pose in candidates:
            # Get corresponding descriptor
            candidate_desc = self.place_descriptors[self.place_ids.index(candidate_id)]

            # Match current and candidate descriptors
            matches = self.match_descriptors(current_descriptor, candidate_desc)

            if len(matches) >= self.min_loop_matches:
                # Estimate relative transformation
                transform, inliers = self.estimate_geometric_consistency(
                    current_descriptor, candidate_desc, matches, current_pose, candidate_pose
                )

                if transform is not None and len(inliers) >= self.min_loop_matches:
                    confidence = len(inliers) / len(matches)  # Geometric consistency ratio
                    return candidate_id, transform, confidence

        return None

    def find_place_candidates(self, current_descriptor: np.ndarray,
                             current_pose: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Find potential place recognition candidates

        Args:
            current_descriptor: Descriptor of current frame
            current_pose: Current camera pose

        Returns:
            List of (keyframe_id, pose) tuples for potential matches
        """
        candidates = []

        for i, (desc, pose, kf_id) in enumerate(zip(self.place_descriptors, self.place_poses, self.place_ids)):
            # Quick geometric check
            pos_diff = np.linalg.norm(current_pose[:3, 3] - pose[:3, 3])

            if pos_diff > self.max_pose_distance:
                continue  # Too far apart geometrically

            # Check orientation difference
            R_current = current_pose[:3, :3]
            R_prev = pose[:3, :3]

            # Calculate rotation difference
            R_diff = R_current @ R_prev.T
            trace = np.trace(R_diff)
            angle_diff = np.arccos(np.clip((trace - 1) / 2, -1, 1))

            if angle_diff > self.max_angle_difference:
                continue  # Too different in orientation

            # Descriptor similarity check
            matches = self.match_descriptors(current_descriptor, desc)
            if len(matches) > 0:
                similarity = len(matches) / max(current_descriptor.shape[0], desc.shape[0])
                if similarity > self.loop_threshold:
                    candidates.append((kf_id, pose))

        return candidates

    def match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """Match two sets of descriptors"""
        if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
            return []

        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            return good_matches
        except cv2.error:
            return []

    def estimate_geometric_consistency(self, desc1: np.ndarray, desc2: np.ndarray,
                                     matches: List[cv2.DMatch],
                                     pose1: np.ndarray, pose2: np.ndarray) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Estimate geometric consistency of matches using essential matrix

        Args:
            desc1, desc2: Descriptors
            matches: Good matches between descriptors
            pose1, pose2: Corresponding poses

        Returns:
            Tuple of (relative_transform, inlier_indices) or (None, [])
        """
        if len(matches) < 8:
            return None, []

        # Extract matched points
        pts1 = np.float32([desc1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([desc2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

        # Estimate essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            cameraMatrix=np.eye(3),  # Use identity, relative pose already accounts for intrinsics
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None, []

        # Recover relative pose
        _, R_rel, t_rel, mask_new = cv2.recoverPose(E, pts1, pts2, mask=mask)

        # Create relative transformation matrix
        T_rel = np.eye(4)
        T_rel[:3, :3] = R_rel
        T_rel[:3, 3] = t_rel.ravel()

        # Get inlier indices
        inliers = []
        if mask_new is not None:
            inliers = [i for i, is_inlier in enumerate(mask_new.flatten()) if is_inlier]

        return T_rel, inliers

    def optimize_with_loop_closure(self, trajectory: List[np.ndarray],
                                  loop_constraints: List[Tuple[int, int, np.ndarray]]) -> List[np.ndarray]:
        """
        Optimize trajectory using loop closure constraints

        Args:
            trajectory: List of poses forming the trajectory
            loop_constraints: List of (frame_i, frame_j, relative_transform) constraints

        Returns:
            Optimized trajectory
        """
        if not loop_constraints:
            return trajectory

        # Simple pose graph optimization (could be replaced with more sophisticated methods)
        optimized_trajectory = trajectory.copy()

        # Apply loop closure corrections iteratively
        for _ in range(5):  # Multiple iterations for better optimization
            for frame_i, frame_j, relative_transform in loop_constraints:
                if frame_i < len(optimized_trajectory) and frame_j < len(optimized_trajectory):
                    # Calculate current relative transform
                    T_i_inv = np.linalg.inv(optimized_trajectory[frame_i])
                    current_rel = T_i_inv @ optimized_trajectory[frame_j]

                    # Calculate error
                    error_T = relative_transform @ np.linalg.inv(current_rel)

                    # Apply correction (simple averaging approach)
                    correction_factor = 0.1  # Small correction factor
                    corrected_T = optimized_trajectory[frame_j] @ (
                        np.eye(4) + correction_factor * (error_T - np.eye(4))
                    )

                    optimized_trajectory[frame_j] = corrected_T

        return optimized_trajectory

    def initialize_vocabulary(self, all_descriptors: List[np.ndarray], vocab_size: int = 1000):
        """
        Initialize vocabulary for bag-of-words place recognition

        Args:
            all_descriptors: List of all descriptors from keyframes
            vocab_size: Size of vocabulary
        """
        if not all_descriptors:
            return

        # Concatenate all descriptors
        all_desc = np.vstack(all_descriptors)

        # Use K-means clustering to create vocabulary
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(
            all_desc.astype(np.float32),
            vocab_size,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS
        )

        self.vocabulary = centers
        self.vocabulary_initialized = True

    def get_visual_words(self, descriptor: np.ndarray) -> List[int]:
        """Convert descriptor to visual words using vocabulary"""
        if not self.vocabulary_initialized:
            return []

        # Find nearest visual words for each feature
        distances = cdist(descriptor, self.vocabulary, metric='euclidean')
        visual_words = np.argmin(distances, axis=1)

        return visual_words.tolist()

    def compute_similarity_score(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Compute similarity score between two descriptors"""
        matches = self.match_descriptors(desc1, desc2)
        if len(matches) == 0:
            return 0.0

        # Compute ratio of good matches
        total_features = max(desc1.shape[0], desc2.shape[0])
        return len(matches) / total_features
```

### Lab Exercise 3: Map Building and Loop Closure Testing
1. Create a map from a sequence of images with known motion
2. Test loop closure detection on a trajectory that returns to start
3. Evaluate the impact of different parameters on map quality
4. Implement map optimization using loop closure constraints

### Expected Results
- Working map building system with 3D point cloud
- Functional loop closure detection
- Understanding of map optimization techniques
- Ability to evaluate mapping system performance

## Lab 4: Isaac Sim Integration

### Objective
Integrate the VSLAM system with Isaac Sim to create a complete simulation pipeline for humanoid robot navigation.

### Implementation Steps

#### Step 1: Isaac Sim Environment Setup
```python
# vslam_lab_4/isaac_integration.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.sensor import Camera
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import cv2
from vslam_lab_3.map_builder import MapBuilder, MapPoint
from vslam_lab_3.loop_closure import LoopClosureDetector

class IsaacVSLAMIntegration:
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)

        # Initialize VSLAM components
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],  # fx, 0, cx
            [0.0, 600.0, 240.0],  # 0, fy, cy
            [0.0, 0.0, 1.0]       # 0, 0, 1
        ])

        self.map_builder = MapBuilder()
        self.loop_detector = LoopClosureDetector()
        self.vslam_pipeline = VSLAMPipeline(self.camera_matrix)

        # Robot state
        self.robot = None
        self.camera = None
        self.trajectory = []
        self.keyframes = []

        # Simulation parameters
        self.keyframe_interval = 10  # Frames between keyframes
        self.frame_count = 0
        self.current_pose = np.eye(4)

    def setup_environment(self):
        """Set up Isaac Sim environment with robot and camera"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add robot (using a simple wheeled robot for this example)
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name="slam_robot",
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
            position=np.array([0, 0, 10]),
            attributes={"color": np.array([0.8, 0.8, 0.8])}
        )

    def run_simulation(self, num_steps: int = 1000):
        """Run VSLAM simulation in Isaac Sim"""
        self.world.reset()

        for step in range(num_steps):
            self.world.step(render=True)

            # Get camera image
            image = self.get_camera_image()

            if image is not None:
                # Get current robot pose (ground truth)
                robot_pose = self.get_robot_pose()

                # Process with VSLAM
                self.process_vslam_step(image, robot_pose, step)

                # Update visualization
                if step % 50 == 0:
                    self.update_visualization()

            self.frame_count += 1

    def get_camera_image(self):
        """Get current camera image from Isaac Sim"""
        try:
            # Get image data from Isaac Sim camera
            image_data = self.camera.get_rgb()
            return image_data
        except Exception as e:
            print(f"Error getting camera image: {e}")
            return None

    def get_robot_pose(self):
        """Get current robot pose from Isaac Sim"""
        try:
            position, orientation = self.robot.get_world_pose()

            # Convert to 4x4 transformation matrix
            R = self.quaternion_to_rotation_matrix(orientation)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = position

            return T
        except Exception as e:
            print(f"Error getting robot pose: {e}")
            return np.eye(4)

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q

        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

        return R

    def process_vslam_step(self, image, ground_truth_pose, frame_num):
        """Process one VSLAM step"""
        # Process with VSLAM pipeline
        results = self.vslam_pipeline.process_frame_pair(
            self.get_previous_image() if hasattr(self, '_prev_image') else image,
            image
        )

        if results['success']:
            # Update current pose estimate
            self.current_pose = results['absolute_pose']

            # Store trajectory
            self.trajectory.append({
                'frame': frame_num,
                'estimated_pose': self.current_pose.copy(),
                'ground_truth_pose': ground_truth_pose
            })

            # Add keyframe if needed
            if frame_num % self.keyframe_interval == 0:
                self.add_keyframe(image, self.current_pose, frame_num)

            # Check for loop closure
            if len(self.keyframes) > 10:  # Need enough keyframes for meaningful comparison
                loop_result = self.check_for_loop_closure(image, self.current_pose, frame_num)

                if loop_result:
                    print(f"Loop closure detected at frame {frame_num}")
                    keyframe_id, transform, confidence = loop_result
                    self.handle_loop_closure(keyframe_id, transform)

        # Store current image for next iteration
        self._prev_image = image

    def add_keyframe(self, image, pose, frame_num):
        """Add current frame as keyframe"""
        # Extract features for keyframe
        keypoints, descriptors = self.vslam_pipeline.detector.detect_and_compute(image)

        keyframe = {
            'frame_num': frame_num,
            'pose': pose,
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'timestamp': self.world.current_time
        }

        self.keyframes.append(keyframe)

        # Add to loop closure detector
        if descriptors is not None and descriptors.size > 0:
            self.loop_detector.add_keyframe(descriptors, pose, self.world.current_time, frame_num)

    def check_for_loop_closure(self, image, current_pose, frame_num):
        """Check for potential loop closure"""
        # Extract features from current image
        keypoints, descriptors = self.vslam_pipeline.detector.detect_and_compute(image)

        if descriptors is not None and descriptors.size > 0:
            # Check for loop closure
            loop_result = self.loop_detector.detect_loop_closure(descriptors, current_pose)
            return loop_result

        return None

    def handle_loop_closure(self, keyframe_id, transform):
        """Handle detected loop closure"""
        # Update map and trajectory based on loop closure
        # This would involve optimizing the map and trajectory
        print(f"Handling loop closure with keyframe {keyframe_id}")

    def update_visualization(self):
        """Update visualization of map and trajectory"""
        # This would update Isaac Sim visualization
        # For now, just print status
        print(f"Processed {self.frame_count} frames, {len(self.map_builder.map_points)} map points, "
              f"{len(self.trajectory)} trajectory points")

    def evaluate_performance(self):
        """Evaluate VSLAM performance against ground truth"""
        if not self.trajectory:
            print("No trajectory data for evaluation")
            return

        # Calculate trajectory errors
        position_errors = []
        orientation_errors = []

        for traj_point in self.trajectory:
            est_pose = traj_point['estimated_pose']
            gt_pose = traj_point['ground_truth_pose']

            # Position error
            pos_err = np.linalg.norm(est_pose[:3, 3] - gt_pose[:3, 3])
            position_errors.append(pos_err)

            # Orientation error
            R_est = est_pose[:3, :3]
            R_gt = gt_pose[:3, :3]

            # Calculate rotation error using Frobenius norm
            R_error = R_est @ R_gt.T - np.eye(3)
            rot_err = np.linalg.norm(R_error, 'fro')
            orientation_errors.append(rot_err)

        # Calculate statistics
        avg_pos_error = np.mean(position_errors) if position_errors else 0
        max_pos_error = np.max(position_errors) if position_errors else 0
        avg_rot_error = np.mean(orientation_errors) if orientation_errors else 0

        print(f"\nVSLAM Performance Evaluation:")
        print(f"- Average position error: {avg_pos_error:.3f}m")
        print(f"- Maximum position error: {max_pos_error:.3f}m")
        print(f"- Average orientation error: {avg_rot_error:.3f} (Frobenius norm)")
        print(f"- Total trajectory points: {len(self.trajectory)}")
        print(f"- Total map points: {len(self.map_builder.map_points)}")

def main():
    """Main function to run Isaac Sim VSLAM integration"""
    import argparse

    parser = argparse.ArgumentParser(description='Isaac Sim VSLAM Integration')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
    args = parser.parse_args()

    # Initialize integration
    vslam_integration = IsaacVSLAMIntegration()

    # Set up environment
    vslam_integration.setup_environment()

    # Run simulation
    vslam_integration.run_simulation(args.steps)

    # Evaluate performance
    vslam_integration.evaluate_performance()

if __name__ == "__main__":
    main()
```

#### Step 2: ROS 2 Bridge Integration
```python
# vslam_lab_4/ros2_bridge.py
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
from vslam_lab_2.pose_estimator import VSLAMPipeline
from vslam_lab_3.map_builder import MapBuilder
from vslam_lab_3.loop_closure import LoopClosureDetector

class IsaacVSLAMROS2Bridge(Node):
    def __init__(self):
        super().__init__('isaac_vslam_ros2_bridge')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Initialize VSLAM components
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

        self.vslam_pipeline = VSLAMPipeline(self.camera_matrix)
        self.map_builder = MapBuilder()
        self.loop_detector = LoopClosureDetector()

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/vslam/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/vslam/odometry', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/vslam/map', 10)
        self.traj_pub = self.create_publisher(Marker, '/vslam/trajectory', 10)
        self.vis_pub = self.create_publisher(Image, '/vslam/visualization', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/isaac_sim/camera/rgb/image', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/isaac_sim/camera/rgb/camera_info', self.camera_info_callback, 10)

        # Internal state
        self.camera_info = None
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.frame_count = 0
        self.prev_image = None

        # Processing parameters
        self.keyframe_interval = 10
        self.processing_enabled = True

        self.get_logger().info('Isaac VSLAM ROS2 Bridge Initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        if self.camera_info is None:
            # Initialize camera matrix from camera info
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

            # Reinitialize VSLAM with proper camera matrix
            self.vslam_pipeline = VSLAMPipeline(self.camera_matrix)

            self.get_logger().info('Camera calibration received and VSLAM initialized')

    def image_callback(self, msg):
        """Process incoming camera images for VSLAM"""
        if not self.processing_enabled or self.camera_info is None:
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process with VSLAM
            if self.prev_image is not None:
                results = self.vslam_pipeline.process_frame_pair(self.prev_image, cv_image)

                if results['success']:
                    # Update current pose
                    self.current_pose = results['absolute_pose']

                    # Store trajectory
                    self.trajectory.append({
                        'timestamp': msg.header.stamp,
                        'pose': self.current_pose.copy()
                    })

                    # Add keyframe if needed
                    if self.frame_count % self.keyframe_interval == 0:
                        self.add_keyframe(cv_image, self.current_pose, msg.header.stamp)

                    # Check for loop closure
                    if len(self.vslam_pipeline.keyframes) > 10:
                        self.check_loop_closure(cv_image, self.current_pose, msg.header)

                    # Publish results
                    self.publish_vslam_results(msg.header)

                    # Broadcast TF
                    self.broadcast_transform(msg.header)

            # Store current image for next iteration
            self.prev_image = cv_image.copy()
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f'Error processing VSLAM frame: {e}')

    def add_keyframe(self, image, pose, timestamp):
        """Add current frame as keyframe for mapping and loop closure"""
        # Extract features
        keypoints, descriptors = self.vslam_pipeline.detector.detect_and_compute(image)

        # Add to VSLAM pipeline
        keyframe_data = {
            'image': image,
            'pose': pose,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'timestamp': timestamp
        }

        self.vslam_pipeline.keyframes.append(keyframe_data)

        # Add to loop closure detector
        if descriptors is not None and descriptors.size > 0:
            self.loop_detector.add_keyframe(
                descriptors, pose,
                timestamp.sec + timestamp.nanosec * 1e-9,
                len(self.vslam_pipeline.keyframes) - 1
            )

    def check_loop_closure(self, image, current_pose, header):
        """Check for loop closure opportunities"""
        # Extract features from current image
        keypoints, descriptors = self.vslam_pipeline.detector.detect_and_compute(image)

        if descriptors is not None and descriptors.size > 0:
            # Check for loop closure
            loop_result = self.loop_detector.detect_loop_closure(descriptors, current_pose)

            if loop_result:
                keyframe_id, transform, confidence = loop_result
                self.get_logger().info(
                    f'Loop closure detected! Keyframe: {keyframe_id}, '
                    f'Confidence: {confidence:.3f}'
                )

                # Handle loop closure (would involve optimization)
                self.handle_loop_closure(keyframe_id, transform, header)

    def handle_loop_closure(self, keyframe_id, transform, header):
        """Handle detected loop closure"""
        # In a full implementation, this would trigger map optimization
        # For this example, just log the event
        self.get_logger().info(f'Loop closure handled for keyframe {keyframe_id}')

    def publish_vslam_results(self, header):
        """Publish VSLAM results to ROS topics"""
        # Publish pose estimate
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'vslam_map'

        pose_msg.pose.position.x = float(self.current_pose[0, 3])
        pose_msg.pose.position.y = float(self.current_pose[1, 3])
        pose_msg.pose.position.z = float(self.current_pose[2, 3])

        # Convert rotation matrix to quaternion
        R = self.current_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(R)
        pose_msg.pose.orientation.w = qw
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz

        self.pose_pub.publish(pose_msg)

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.header.frame_id = 'vslam_map'
        odom_msg.child_frame_id = 'vslam_camera'

        odom_msg.pose.pose = pose_msg.pose

        # Set velocity based on recent movement (simplified)
        if len(self.trajectory) > 1:
            prev_pose = self.trajectory[-2]['pose']
            dt = 0.1  # Assume 10Hz
            linear_vel = (self.current_pose[:3, 3] - prev_pose[:3, 3]) / dt
            odom_msg.twist.twist.linear.x = linear_vel[0]
            odom_msg.twist.twist.linear.y = linear_vel[1]
            odom_msg.twist.twist.linear.z = linear_vel[2]

        self.odom_pub.publish(odom_msg)

        # Publish map visualization
        self.publish_map_visualization(header)

        # Publish trajectory visualization
        self.publish_trajectory_visualization(header)

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

    def broadcast_transform(self, header):
        """Broadcast TF transform for VSLAM results"""
        t = TransformStamped()

        t.header.stamp = header.stamp
        t.header.frame_id = 'vslam_map'
        t.child_frame_id = 'vslam_camera'

        t.transform.translation.x = float(self.current_pose[0, 3])
        t.transform.translation.y = float(self.current_pose[1, 3])
        t.transform.translation.z = float(self.current_pose[2, 3])

        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(self.current_pose[:3, :3])
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz

        self.tf_broadcaster.sendTransform(t)

    def publish_map_visualization(self, header):
        """Publish map points as visualization markers"""
        marker_array = MarkerArray()

        # Create markers for map points
        for i, map_point in enumerate(self.map_builder.map_points[:1000]):  # Limit for performance
            marker = Marker()
            marker.header = header
            marker.header.frame_id = 'vslam_map'
            marker.ns = 'vslam_map'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(map_point.coordinates[0])
            marker.pose.position.y = float(map_point.coordinates[1])
            marker.pose.position.z = float(map_point.coordinates[2])

            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            # Color based on tracking count
            if map_point.tracking_count > 10:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.map_pub.publish(marker_array)

    def publish_trajectory_visualization(self, header):
        """Publish trajectory as visualization marker"""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = 'vslam_map'
        marker.ns = 'vslam_trajectory'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Add trajectory points
        for traj_point in self.trajectory[-100:]:  # Last 100 points
            pose = traj_point['pose']
            point = PointStamped()
            point.point.x = float(pose[0, 3])
            point.point.y = float(pose[1, 3])
            point.point.z = float(pose[2, 3])
            marker.points.append(point.point)

        marker.scale.x = 0.02  # Line width
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.traj_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    vslam_bridge = IsaacVSLAMROS2Bridge()

    try:
        rclpy.spin(vslam_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Lab Exercise 4: Complete VSLAM Integration
1. Integrate VSLAM system with Isaac Sim environment
2. Connect Isaac Sim to ROS 2 using the bridge
3. Test VSLAM performance in simulated environment
4. Evaluate results against ground truth from Isaac Sim

### Expected Results
- Complete integration pipeline from Isaac Sim to ROS 2
- Working VSLAM system in simulation environment
- Real-time performance with proper visualization
- Quantified performance metrics against ground truth

## Performance Optimization and Evaluation

### Real-time Performance Considerations
```python
# Performance optimization for VSLAM
import time
import threading
from collections import deque
import psutil
import GPUtil

class VSLAMPerformanceOptimizer:
    def __init__(self):
        self.frame_times = deque(maxlen=30)  # Last 30 frame times
        self.feature_times = deque(maxlen=30)
        self.pose_times = deque(maxlen=30)
        self.mapping_times = deque(maxlen=30)

        self.target_fps = 30
        self.current_fps = 0
        self.feature_count_target = 1000

        # Resource monitoring
        self.cpu_usage = deque(maxlen=30)
        self.gpu_usage = deque(maxlen=30)
        self.memory_usage = deque(maxlen=30)

        # Adaptive parameters
        self.adaptive_params = {
            'max_features': 1000,
            'matching_threshold': 0.75,
            'ransac_threshold': 1.0,
            'bundle_adjustment_frequency': 10
        }

    def monitor_performance(self):
        """Monitor system performance in separate thread"""
        while True:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.append(cpu_percent)

            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            self.memory_usage.append(memory_percent)

            # GPU usage (if available)
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                self.gpu_usage.append(gpu_percent)
            else:
                self.gpu_usage.append(0)

            # Calculate current FPS
            if len(self.frame_times) > 1:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    def adaptive_processing(self, image):
        """Adapt processing based on performance"""
        start_time = time.time()

        # Adjust feature detection based on performance
        if self.current_fps < self.target_fps * 0.8:
            # Reduce feature count to improve performance
            self.adaptive_params['max_features'] = max(500,
                int(self.adaptive_params['max_features'] * 0.9))
        elif self.current_fps > self.target_fps * 1.1:
            # Increase feature count for better accuracy
            self.adaptive_params['max_features'] = min(2000,
                int(self.adaptive_params['max_features'] * 1.1))

        # Process features
        features_start = time.time()
        keypoints, descriptors = self.detect_features_adaptive(image)
        self.feature_times.append(time.time() - features_start)

        # Continue with rest of VSLAM pipeline
        processing_time = time.time() - start_time
        self.frame_times.append(processing_time)

        return keypoints, descriptors

    def detect_features_adaptive(self, image):
        """Detect features with adaptive parameters"""
        detector = cv2.ORB_create(
            nfeatures=self.adaptive_params['max_features'],
            scaleFactor=1.2,
            nlevels=4,  # Reduce levels for speed
            edgeThreshold=19,
            patchSize=19,
            fastThreshold=20
        )

        keypoints, descriptors = detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def get_performance_metrics(self):
        """Get current performance metrics"""
        metrics = {
            'fps': self.current_fps,
            'avg_frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
            'avg_feature_time': sum(self.feature_times) / len(self.feature_times) if self.feature_times else 0,
            'avg_pose_time': sum(self.pose_times) / len(self.pose_times) if self.pose_times else 0,
            'avg_mapping_time': sum(self.mapping_times) / len(self.mapping_times) if self.mapping_times else 0,
            'cpu_usage_avg': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'gpu_usage_avg': sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0,
            'memory_usage_avg': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        }

        return metrics

    def optimize_parameters(self):
        """Optimize parameters based on performance"""
        metrics = self.get_performance_metrics()

        # Adjust parameters based on performance
        if metrics['fps'] < self.target_fps * 0.7:
            # Significantly below target, aggressive optimization needed
            self.adaptive_params['bundle_adjustment_frequency'] = max(20,
                self.adaptive_params['bundle_adjustment_frequency'] + 5)
        elif metrics['fps'] > self.target_fps * 1.05:
            # Above target, can afford more processing
            self.adaptive_params['bundle_adjustment_frequency'] = max(5,
                self.adaptive_params['bundle_adjustment_frequency'] - 1)

        return self.adaptive_params
```

## Evaluation and Validation

### VSLAM Evaluation Framework
```python
# VSLAM evaluation and validation
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class VSLAMEvaluator:
    def __init__(self):
        self.estimated_trajectory = []
        self.ground_truth_trajectory = []
        self.estimated_map = []
        self.ground_truth_map = []
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
        # Implementation would compare rotation matrices/quaternions
        pass

    def plot_results(self):
        """Plot evaluation results"""
        if not self.estimated_trajectory or not self.ground_truth_trajectory:
            print("No trajectory data to plot")
            return

        est_traj = np.array(self.estimated_trajectory)
        gt_traj = np.array(self.ground_truth_trajectory)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot trajectories
        axes[0, 0].plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', label='Ground Truth', linewidth=2)
        axes[0, 0].plot(est_traj[:, 0], est_traj[:, 1], 'r-', label='Estimated', linewidth=2)
        axes[0, 0].set_title('Trajectory Comparison')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot position errors over time
        if len(est_traj) == len(gt_traj):
            position_errors = np.linalg.norm(est_traj - gt_traj, axis=1)
            axes[0, 1].plot(position_errors)
            axes[0, 1].set_title('Position Error Over Time')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Error (m)')
            axes[0, 1].grid(True)

        # Plot X, Y, Z components separately
        time_axis = range(len(est_traj))
        axes[1, 0].plot(time_axis, gt_traj[:, 0], 'g-', label='GT X')
        axes[1, 0].plot(time_axis, est_traj[:, 0], 'r-', label='Est X')
        axes[1, 0].set_title('X Position Over Time')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('X (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(time_axis, gt_traj[:, 1], 'g-', label='GT Y')
        axes[1, 1].plot(time_axis, est_traj[:, 1], 'r-', label='Est Y')
        axes[1, 1].set_title('Y Position Over Time')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Y (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

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
- Average FPS: {self.average_fps:.2f}
- Feature Processing Time: {self.avg_feature_time:.3f}s
- Pose Estimation Time: {self.avg_pose_time:.3f}s
- Total Processing Time: {self.avg_total_time:.3f}s

Robustness:
- Tracking Success Rate: {self.tracking_success_rate:.2f}%
- Map Point Survival Rate: {self.map_survival_rate:.2f}%
- Loop Closure Success Rate: {self.loop_closure_rate:.2f}%

Overall Rating: {self.calculate_overall_score():.2f}/10.0
        """

        return report

    def calculate_overall_score(self):
        """Calculate overall VSLAM performance score"""
        # Weighted combination of different metrics
        ate_score = max(0, 10 - (self.ate_rmse * 5))  # Higher error = lower score
        rpe_score = max(0, 10 - (self.rpe_rmse * 10))
        fps_score = min(10, (self.average_fps / 30) * 10)  # Target 30 FPS
        robustness_score = self.tracking_success_rate

        overall_score = (ate_score * 0.3 + rpe_score * 0.3 +
                        fps_score * 0.2 + robustness_score * 0.2)

        return min(10, max(0, overall_score))
```

## Troubleshooting and Best Practices

### Common VSLAM Issues and Solutions
```python
# Troubleshooting guide for VSLAM systems
class VSLAMTroubleshooter:
    def __init__(self):
        self.common_issues = {
            'tracking_lost': {
                'symptoms': ['No pose output', 'Large jumps in trajectory', 'Feature depletion'],
                'causes': ['Fast motion', 'Low texture', 'Poor lighting'],
                'solutions': [
                    'Reduce motion speed',
                    'Move to textured environment',
                    'Improve lighting conditions',
                    'Increase feature count'
                ]
            },
            'drift_accumulation': {
                'symptoms': ['Trajectory divergence', 'Growing error over time'],
                'causes': ['Integration errors', 'Small loop closures', 'Insufficient optimization'],
                'solutions': [
                    'Implement loop closure',
                    'Regular bundle adjustment',
                    'Use IMU integration',
                    'Global optimization'
                ]
            },
            'map_degradation': {
                'symptoms': ['Decreasing map quality', 'Poor relocalization'],
                'causes': ['Map growing too large', 'Bad measurements accumulating'],
                'solutions': [
                    'Map management and pruning',
                    'Local map window',
                    'Quality-based filtering',
                    'Relocalization system'
                ]
            }
        }

    def diagnose_issue(self, symptoms):
        """Diagnose VSLAM issues based on symptoms"""
        possible_issues = []

        for issue_name, issue_data in self.common_issues.items():
            symptom_matches = sum(1 for symptom in symptoms if symptom in issue_data['symptoms'])
            if symptom_matches > 0:
                confidence = symptom_matches / len(issue_data['symptoms'])
                possible_issues.append({
                    'issue': issue_name,
                    'confidence': confidence,
                    'solutions': issue_data['solutions']
                })

        return sorted(possible_issues, key=lambda x: x['confidence'], reverse=True)

    def performance_monitoring(self):
        """Monitor VSLAM system performance"""
        # Track key metrics
        metrics = {
            'feature_count': [],
            'match_ratio': [],
            'tracking_inliers': [],
            'processing_time': [],
            'map_size': [],
            'keyframe_rate': []
        }

        # Set thresholds for alerting
        thresholds = {
            'low_features': 50,      # Alert if fewer than 50 features
            'low_matches': 0.1,      # Alert if match ratio < 10%
            'low_inliers': 0.3,      # Alert if inlier ratio < 30%
            'high_time': 0.1,        # Alert if processing > 100ms
            'large_map': 10000       # Alert if map > 10k points
        }

        return metrics, thresholds
```

## Practical Lab: Complete VSLAM System

### Lab Objective
Implement a complete VSLAM system that integrates all components: feature detection, pose estimation, mapping, loop closure, and evaluation.

### Implementation Steps
1. Set up Isaac Sim environment with robot and camera
2. Implement complete VSLAM pipeline with all components
3. Integrate with ROS 2 for communication and visualization
4. Test system in various environments and conditions
5. Evaluate performance using ground truth from simulation

### Expected Outcome
- Working complete VSLAM system
- Real-time performance with reasonable accuracy
- Proper ROS 2 integration and visualization
- Quantified performance metrics

## Review Questions

1. Explain the differences between feature-based, direct, and semi-direct VSLAM approaches.
2. How does loop closure detection improve VSLAM system accuracy?
3. What are the key challenges in real-time VSLAM implementation?
4. How do you evaluate VSLAM system performance quantitatively?
5. What are the main factors affecting VSLAM robustness in real environments?

## Next Steps
After mastering VSLAM systems, students should proceed to:
- Advanced navigation for humanoid robots
- Vision-Language-Action system integration
- Sim-to-real transfer techniques
- Deep learning enhanced perception systems

This comprehensive practical lab provides hands-on experience with complete VSLAM system development, essential for Physical AI and Humanoid Robotics applications.