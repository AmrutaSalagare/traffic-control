#!/usr/bin/env python3
"""
Final Traffic Detection with Tracking System using ONNX models

This version uses ONNX Runtime for optimized inference with the models we've optimized.
"""

import os
import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import deque, defaultdict
import argparse

# Import our ONNX-based detectors
from onnx_inference import ONNXYOLODetector, ONNXAmbulanceDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('traffic_control.log')
    ]
)
logger = logging.getLogger('TrafficControl')

# Load debug configuration
debug_config = {}
try:
    with open('config/debug_config.json', 'r') as f:
        debug_config = json.load(f)
    logger.info("Loaded debug configuration")
except Exception as e:
    logger.warning(f"Could not load debug config: {e}")
    debug_config = {
        'debug_ambulance': False,
        'debug_vehicle': False,
        'log_level': 'INFO',
        'save_debug_images': False,
        'debug_output_dir': 'debug_output'
    }

# Vehicle classes - all classified as "vehicle" for simplicity
VEHICLE_CLASSES = {
    0: 'vehicle',  # car
    1: 'vehicle',  # motorcycle
    2: 'vehicle',  # bus
    3: 'vehicle',  # truck
    4: 'vehicle',  # auto-rickshaw
    5: 'vehicle',  # truck (additional class)
    7: 'vehicle',  # truck (additional class)
}

# Colors for visualization
VEHICLE_COLORS = {
    'vehicle': (0, 255, 0),      # Green for all vehicles
    'ambulance': (0, 0, 255)     # Red for ambulances
}


class ONNXVehicleTracker:
    """Improved vehicle tracking system using ONNX models"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        """Initialize tracker"""
        self.next_id = 0
        # {id: {'bbox': [x1,y1,x2,y2], 'class': 'car', 'confidence': 0.8, 'history': []}}
        self.objects = {}
        self.disappeared = {}  # {id: frames_disappeared}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Enhanced smoothing parameters
        self.smoothing_factor = 0.65
        self.history_length = 7
        
        # For counting unique vehicles
        self.counted_vehicles = set()
        self.total_count = 0
        
        # For trajectory visualization
        self.trajectory_points = {}
        self.max_trajectory_length = 20
        
        # Line crossing detection
        self.crossed_ids = set()
        self.line_y = None
        
        # Color palette for tracks
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 255), (255, 20, 147), (34, 139, 34), (255, 140, 0)
        ]
        self.track_colors = {}
    
    def update(self, detections: List[Dict]):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # No detections, mark all as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return self.objects
        
        # Initialize arrays for tracking
        if len(self.objects) == 0:
            # First frame, register all detections
            for det in detections:
                self._register(det)
        else:
            # Match detections to existing objects
            object_ids = list(self.objects.keys())
            object_centers = [
                self._get_center(self.objects[obj_id]['bbox']) 
                for obj_id in object_ids
            ]
            
            # Calculate distances between objects and detections
            detection_centers = [self._get_center(det['bbox']) for det in detections]
            
            # Simple nearest neighbor matching
            matched_detections = set()
            matched_objects = set()
            
            for i, obj_id in enumerate(object_ids):
                if obj_id in matched_objects:
                    continue
                    
                obj_center = object_centers[i]
                min_dist = float('inf')
                best_match = -1
                
                for j, det_center in enumerate(detection_centers):
                    if j in matched_detections:
                        continue
                        
                    dist = np.linalg.norm(np.array(obj_center) - np.array(det_center))
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        best_match = j
                
                if best_match != -1:
                    # Update existing object
                    obj_id = object_ids[i]
                    det = detections[best_match]
                    
                    # Smooth the bounding box
                    old_bbox = self.objects[obj_id]['bbox']
                    new_bbox = det['bbox']
                    smoothed_bbox = self._smooth_bbox(old_bbox, new_bbox)
                    
                    # Update object
                    self.objects[obj_id].update({
                        'bbox': smoothed_bbox,
                        'class': det.get('class_name', 'vehicle'),
                        'confidence': det['confidence']
                    })
                    
                    # Reset disappeared counter
                    self.disappeared[obj_id] = 0
                    
                    # Update trajectory
                    self._update_trajectory(obj_id, smoothed_bbox)
                    
                    matched_detections.add(best_match)
                    matched_objects.add(obj_id)
            
            # Handle unmatched detections (new objects)
            for i, det in enumerate(detections):
                if i not in matched_detections:
                    self._register(det)
            
            # Handle disappeared objects
            for obj_id in list(self.objects.keys()):
                if obj_id not in matched_objects:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self._deregister(obj_id)
        
        return self.objects
    
    def _register(self, detection: Dict):
        """Register a new object"""
        object_id = self.next_id
        self.next_id += 1
        
        self.objects[object_id] = {
            'bbox': detection['bbox'].copy(),
            'class': detection.get('class_name', 'vehicle'),
            'confidence': detection['confidence'],
            'history': [detection['bbox'].copy()]
        }
        
        self.disappeared[object_id] = 0
        self.trajectory_points[object_id] = deque(maxlen=self.max_trajectory_length)
        self._update_trajectory(object_id, detection['bbox'])
        
        # Assign a color to this track
        if object_id not in self.track_colors:
            self.track_colors[object_id] = self.color_palette[object_id % len(self.color_palette)]
        
        return object_id
    
    def _deregister(self, object_id: int):
        """Deregister an object"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.trajectory_points:
            del self.trajectory_points[object_id]
    
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _smooth_bbox(self, old_bbox: List[float], new_bbox: List[float]) -> List[float]:
        """Smooth bounding box coordinates"""
        return [
            int(old * (1 - self.smoothing_factor) + new * self.smoothing_factor)
            for old, new in zip(old_bbox, new_bbox)
        ]
    
    def _update_trajectory(self, object_id: int, bbox: List[float]):
        """Update trajectory for an object"""
        if object_id not in self.trajectory_points:
            self.trajectory_points[object_id] = deque(maxlen=self.max_trajectory_length)
        
        # Add center point to trajectory
        center = self._get_center(bbox)
        self.trajectory_points[object_id].append(center)
    
    def check_line_crossing(self, object_id: int, line_y: int) -> bool:
        """Check if an object has crossed the counting line"""
        if object_id in self.crossed_ids:
            return False
            
        if object_id not in self.trajectory_points or len(self.trajectory_points[object_id]) < 2:
            return False
            
        # Get the last two points in the trajectory
        points = list(self.trajectory_points[object_id])
        y_prev = points[-2][1]
        y_curr = points[-1][1]
        
        # Check if the line was crossed
        if y_prev > line_y >= y_curr or y_prev <= line_y < y_curr:
            self.crossed_ids.add(object_id)
            return True
            
        return False
    
    def draw_trajectories(self, frame: np.ndarray):
        """Draw trajectories for all tracked objects"""
        for object_id, points in self.trajectory_points.items():
            if object_id not in self.track_colors:
                continue
                
            color = self.track_colors[object_id]
            points = np.array(list(points), np.int32).reshape((-1, 1, 2))
            if len(points) > 1:
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


class ONNXTrafficDetector:
    """Traffic detector using ONNX models for optimized inference"""
    
    def __init__(self, device: str = 'cpu'):
        """Initialize detector"""
        self.device = device
        self.vehicle_model = None
        self.ambulance_model = None
        self.tracker = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.vehicle_count = 0
        self.ambulance_detected = False
        self.last_ambulance_detection = 0
        self.ambulance_cooldown = 30  # frames
        
        # Enhanced temporal smoothing (inspired by dedicated detector)
        self.ambulance_detection_history = deque(maxlen=20)  # Increased window
        self.ambulance_confidence_history = deque(maxlen=20)  # Increased window  
        self.ambulance_position_history = deque(maxlen=15)    # Position tracking
        self.ambulance_tracklet_frames = 0  # Consecutive frames with ambulance
        self.min_tracklet_frames = 3  # Minimum frames to confirm ambulance
        
        # Stability parameters (from dedicated detector) - balanced for real ambulance detection
        self.stability_ratio = 0.6  # Need 60% detection rate (reduced back)
        self.min_confidence_for_stability = 0.04  # Lower threshold to catch real ambulances
        self.ambulance_stable = False
        self.stable_frames_count = 0
        
        # Adaptive confidence system - balanced settings 
        self.min_ambulance_confidence = 0.01   # Very low base threshold
        self.adaptive_confidence_enabled = True  # Enable adaptive confidence
        self.require_vehicle_overlap = True     # Must overlap with vehicle detection
        self.min_overlap_ratio = 0.1           # Very lenient 10% overlap
        
        # Debug settings from config
        self.debug_ambulance = debug_config.get('debug_ambulance', False)
        self.save_debug_images = debug_config.get('save_debug_images', False)
        self.debug_output_dir = debug_config.get('debug_output_dir', 'debug_output')
        
        # Create debug output directory if needed
        if self.save_debug_images and not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir, exist_ok=True)
        
        # Adaptive confidence parameters
        self.low_confidence_boost_threshold = 0.08  # Boost detections below this
        self.feature_boost_multiplier = 2.0         # Multiply feature boost for low confidence
        self.temporal_boost_for_low_conf = 0.15     # Extra boost for temporally consistent low conf
        
        # Fallback detection for ambulances not in training data
        self.use_vehicle_as_ambulance_fallback = True  # Classify certain vehicles as ambulances
        self.ambulance_keywords = ['ambulance', 'medical', 'emergency']  # For text detection
        self.fallback_confidence_boost = 0.3   # Boost for likely ambulance vehicles
        
        # ROI for ambulance detection (will be set dynamically)
        self.ambulance_roi = None
        self.roi_enabled = True
        
        # Advanced visual cues detection
        self.previous_frames = deque(maxlen=5)  # Store frames for flashing detection
        self.ambulance_visual_features = {}  # Store detected features per detection
        
        # Initialize models
        self._initialize_models()
        
        # Initialize tracker
        self.tracker = ONNXVehicleTracker()
        
        # Line for counting (y-coordinate, horizontal line at 2/3 of frame height)
        self.count_line_y = None
    
    def _initialize_models(self):
        """Initialize ONNX models"""
        logger.info("Initializing ONNX models...")
        
        # Get the absolute path to the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Paths to optimized models - using absolute paths
        vehicle_model_path = os.path.join(project_dir, "optimized_models", "yolo11n_optimized.onnx")
        ambulance_model_path = os.path.join(project_dir, "optimized_models", "indian_ambulance_yolov11n_best_optimized.onnx")
        
        logger.debug(f"Looking for vehicle model at: {vehicle_model_path}")
        logger.debug(f"Looking for ambulance model at: {ambulance_model_path}")
        
        # Initialize vehicle detector
        if os.path.exists(vehicle_model_path):
            logger.info(f"Loading vehicle model from {vehicle_model_path}")
            try:
                self.vehicle_model = ONNXYOLODetector(vehicle_model_path, conf_thres=0.25)
                logger.info("Vehicle model loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading vehicle model: {str(e)}")
                raise
        else:
            error_msg = f"Vehicle model not found at {vehicle_model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Initialize ambulance detector with lower threshold for better small ambulance detection
        if os.path.exists(ambulance_model_path):
            logger.info(f"Loading ambulance model from {ambulance_model_path}")
            try:
                # Use multiple confidence levels balanced for real ambulance detection
                self.ambulance_model = ONNXAmbulanceDetector(ambulance_model_path, conf_thres=0.01)  # Lower base threshold
                self.ambulance_confidence_levels = [0.15, 0.08, 0.04, 0.02]  # More sensitive levels including very low
                logger.info("Ambulance model loaded successfully!")
            except Exception as e:
                logger.warning(f"Error loading ambulance model: {str(e)}")
                logger.warning("Continuing without ambulance detection...")
                self.ambulance_model = None
        else:
            logger.warning(f"Ambulance model not found at {ambulance_model_path}")
            logger.warning("Continuing without ambulance detection...")
            self.ambulance_model = None
        
        logger.info("Model initialization complete!")
    
    def _filter_vehicle_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to only include vehicles (no persons, animals, etc.)"""
        vehicle_detections = []
        
        for det in detections:
            class_id = det['class_id']
            
            # Only keep vehicle classes (0-9 are typically vehicles in COCO-based models)
            # Person is usually class 0 in COCO, so we need to be more specific
            # YOLO vehicle classes are typically: car(2), motorcycle(3), bus(5), truck(7)
            # But our custom model might have different mapping
            
            # For our custom vehicle model, classes 0-4, 5, 7 should be vehicles
            if class_id in VEHICLE_CLASSES:
                # Map to generic "vehicle" class
                det['class_name'] = 'vehicle'
                det['original_class_id'] = class_id  # Keep original for reference
                vehicle_detections.append(det)
        
        return vehicle_detections
    
    def _apply_nms_to_ambulance_detections(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """Apply Non-Maximum Suppression to ambulance detections to reduce duplicates"""
        if len(detections) <= 1:
            return detections
        
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        indices = []
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            confidence = det['confidence']
            
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            scores.append(confidence)
            indices.append(i)
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply OpenCV's NMS
        try:
            keep_indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                scores.tolist(), 
                score_threshold=0.01,  # Very low threshold since we already filtered
                nms_threshold=iou_threshold
            )
            
            if len(keep_indices) > 0:
                keep_indices = keep_indices.flatten()
                return [detections[i] for i in keep_indices]
            else:
                return []
        except:
            # Fallback to simple overlap removal if OpenCV NMS fails
            return self._simple_nms_fallback(detections, iou_threshold)
    
    def _simple_nms_fallback(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Simple NMS fallback implementation"""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        while sorted_detections:
            # Take the highest confidence detection
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for det in sorted_detections:
                iou = self._calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            
            sorted_detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_ambulance_detections(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """Enhanced ambulance detection filtering with size validation and NMS"""
        if not detections:
            return []
        
        # Step 1: Apply size and shape filtering
        size_filtered = self._apply_size_shape_filtering(detections, frame_shape)
        
        # Step 2: Apply Non-Maximum Suppression to reduce duplicates
        nms_filtered = self._apply_nms_to_ambulance_detections(size_filtered, iou_threshold=0.4)
        
        # Step 3: Apply context-based confidence calibration
        calibrated = self._calibrate_detection_confidence(nms_filtered, frame_shape)
        
        return calibrated
    
    def _apply_size_shape_filtering(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """Apply size and shape-based filtering for ambulance detections"""
        filtered_detections = []
        h, w = frame_shape[:2]
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Calculate relative size (as percentage of frame)
            relative_area = area / (w * h)
            aspect_ratio = width / height if height > 0 else 0
            
            # Enhanced filtering criteria:
            # 1. Minimum size - more restrictive to reduce false positives
            min_area_threshold = 0.0008  # 0.08% of frame area
            
            # 2. Maximum size - ambulances shouldn't be too large 
            max_area_threshold = 0.25    # 25% of frame area
            
            # 3. Realistic aspect ratio for ambulances
            min_aspect_ratio = 0.5       # Not too tall
            max_aspect_ratio = 3.0       # Not too wide
            
            # 4. Adaptive confidence threshold based on size and position
            if relative_area < 0.002:    # Very small detections (distant)
                confidence_threshold = 0.20  # Higher threshold for distant objects
            elif relative_area < 0.01:   # Small detections
                confidence_threshold = 0.15  # Medium threshold
            else:                        # Regular size detections
                confidence_threshold = 0.12  # Base threshold
            
            # 5. Minimum pixel dimensions (absolute)
            min_width = max(20, int(w * 0.015))   # At least 1.5% of frame width
            min_height = max(15, int(h * 0.012))  # At least 1.2% of frame height
            
            # 6. Position-based filtering (ambulances unlikely at frame edges)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            edge_margin = 0.05  # 5% margin from edges
            
            in_valid_region = (edge_margin * w <= center_x <= (1 - edge_margin) * w and
                             edge_margin * h <= center_y <= (1 - edge_margin) * h)
            
            # Apply all filters
            if (confidence >= confidence_threshold and
                min_area_threshold <= relative_area <= max_area_threshold and
                min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
                width >= min_width and height >= min_height and
                in_valid_region):
                
                # Add validation metadata
                det['validation'] = {
                    'relative_area': relative_area,
                    'aspect_ratio': aspect_ratio,
                    'pixel_area': area,
                    'confidence_threshold_used': confidence_threshold
                }
                
                filtered_detections.append(det)
        
        return filtered_detections
    
    def _calibrate_detection_confidence(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """Calibrate detection confidence based on context and validation metrics"""
        calibrated_detections = []
        
        for det in detections:
            bbox = det['bbox']
            original_confidence = det['confidence']
            
            # Base calibration factors
            calibration_factor = 1.0
            
            # Factor 1: Size-based calibration
            if 'validation' in det:
                relative_area = det['validation']['relative_area']
                aspect_ratio = det['validation']['aspect_ratio']
                
                # Prefer medium-sized detections
                if 0.005 <= relative_area <= 0.05:  # Optimal size range
                    calibration_factor *= 1.1
                elif relative_area < 0.002:  # Very small - reduce confidence
                    calibration_factor *= 0.8
                elif relative_area > 0.1:    # Very large - reduce confidence
                    calibration_factor *= 0.7
                
                # Prefer realistic aspect ratios
                if 0.8 <= aspect_ratio <= 2.2:  # Typical ambulance proportions
                    calibration_factor *= 1.05
            
            # Factor 2: Position-based calibration
            x1, y1, x2, y2 = bbox
            center_y = (y1 + y2) / 2
            frame_height = frame_shape[0]
            
            # Ambulances more likely in middle-to-lower part of frame (road level)
            relative_y = center_y / frame_height
            if 0.4 <= relative_y <= 0.8:  # Road level
                calibration_factor *= 1.1
            elif relative_y < 0.3:  # Sky level - unlikely
                calibration_factor *= 0.6
            
            # Apply calibration
            calibrated_confidence = min(original_confidence * calibration_factor, 1.0)
            
            # Adaptive final threshold based on detection characteristics
            final_threshold = self._get_adaptive_confidence_threshold(det, calibrated_confidence)
            
            if calibrated_confidence >= final_threshold:
                det['confidence'] = calibrated_confidence
                det['original_confidence'] = original_confidence
                det['calibration_factor'] = calibration_factor
                det['adaptive_threshold_used'] = final_threshold
                calibrated_detections.append(det)
        
        return calibrated_detections
    
    def _get_adaptive_confidence_threshold(self, detection: Dict, calibrated_confidence: float) -> float:
        """Get adaptive confidence threshold based on detection characteristics"""
        base_threshold = 0.08
        
        if not self.adaptive_confidence_enabled:
            return base_threshold
        
        # Factor 1: Feature presence - lower threshold if strong features detected
        if 'features' in detection:
            features = detection['features']
            total_feature_score = features.get('total_boost', 0.0)
            
            # Strong ambulance features = lower threshold needed
            if total_feature_score > 0.2:  # Very strong features
                base_threshold *= 0.4  # Much lower threshold
            elif total_feature_score > 0.1:  # Good features
                base_threshold *= 0.6  # Lower threshold
            elif total_feature_score > 0.05:  # Some features
                base_threshold *= 0.8  # Slightly lower threshold
        
        # Factor 2: Size-based adjustment - larger ambulances can have lower threshold
        if 'validation' in detection:
            relative_area = detection['validation'].get('relative_area', 0.01)
            
            # Larger detections are more likely to be real
            if relative_area > 0.02:  # Large detection
                base_threshold *= 0.7
            elif relative_area > 0.01:  # Medium detection
                base_threshold *= 0.85
        
        # Factor 3: Temporal consistency - if we've been detecting something consistently
        if hasattr(self, 'ambulance_detection_history') and len(self.ambulance_detection_history) >= 5:
            recent_detections = list(self.ambulance_detection_history)[-5:]
            detection_rate = sum(recent_detections) / len(recent_detections)
            
            # High recent detection rate = lower threshold
            if detection_rate >= 0.8:  # Very consistent
                base_threshold *= 0.5
            elif detection_rate >= 0.6:  # Consistent
                base_threshold *= 0.7
        
        # Ensure minimum threshold (don't go too low)
        min_threshold = 0.03
        max_threshold = 0.15
        
        return max(min_threshold, min(base_threshold, max_threshold))
    
    def _validate_very_low_confidence_detections(self, detections: List[Dict], frame: np.ndarray, 
                                               vehicle_detections: List[Dict]) -> List[Dict]:
        """Special validation for very low confidence detections that might be genuine ambulances"""
        validated_detections = []
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Generate detection ID for feature analysis
            detection_id = f"lowconf_{len(validated_detections)}_{self.frame_count}"
            
            # Analyze ambulance features more thoroughly for low confidence
            features = self._detect_ambulance_features(frame, bbox, detection_id)
            total_feature_score = features['total_boost']
            
            # Check vehicle overlap (must have good overlap for low confidence)
            has_good_overlap = False
            if vehicle_detections:
                for vehicle in vehicle_detections:
                    overlap_ratio = self._calculate_overlap_ratio(bbox, vehicle['bbox'])
                    if overlap_ratio > 0.3:  # Higher overlap requirement for low confidence
                        has_good_overlap = True
                        break
            
            # Validation criteria for very low confidence detections
            validation_score = 0.0
            
            # Criterion 1: Strong ambulance features (most important)
            if total_feature_score > 0.15:  # Strong features
                validation_score += 0.4
            elif total_feature_score > 0.08:  # Moderate features
                validation_score += 0.2
            
            # Criterion 2: Good vehicle overlap
            if has_good_overlap:
                validation_score += 0.3
            
            # Criterion 3: Reasonable size and position
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            frame_area = frame.shape[0] * frame.shape[1]
            relative_area = area / frame_area
            
            if 0.002 <= relative_area <= 0.1:  # Reasonable size
                validation_score += 0.2
            
            # Criterion 4: Temporal consistency (if we have history)
            if len(self.ambulance_detection_history) >= 3:
                recent_detections = list(self.ambulance_detection_history)[-3:]
                if sum(recent_detections) >= 2:  # Recent detection activity
                    validation_score += 0.1
            
            # Accept if validation score is high enough
            if validation_score >= 0.6:  # Need strong evidence for low confidence
                # Boost the confidence based on validation
                boosted_confidence = min(confidence + total_feature_score + 0.1, 0.15)
                
                det['confidence'] = boosted_confidence
                det['original_confidence'] = confidence
                det['validation_score'] = validation_score
                det['feature_boost'] = total_feature_score
                det['features'] = features
                det['detection_id'] = detection_id
                det['low_conf_rescue'] = True
                
                validated_detections.append(det)
                
                if self.debug_ambulance:
                    logger.debug(f"LOW-CONF RESCUE: conf={confidence:.3f} → {boosted_confidence:.3f} "
                               f"(validation_score={validation_score:.2f}, features={total_feature_score:.3f})")
        
        return validated_detections
    
    def _calculate_overlap_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        return intersection / area1 if area1 > 0 else 0.0
    
    def _setup_ambulance_roi(self, frame_shape: Tuple[int, int]):
        """Setup Region of Interest for ambulance detection (road area)"""
        if self.ambulance_roi is not None:
            return
            
        h, w = frame_shape[:2]
        
        # Define ROI as the main road area (exclude top sky and bottom edges)
        # This focuses detection on areas where ambulances are likely to appear
        roi_top = int(h * 0.2)      # Skip top 20% (sky, buildings)
        roi_bottom = int(h * 0.95)  # Skip bottom 5% (very close foreground)
        roi_left = int(w * 0.05)    # Skip leftmost 5%
        roi_right = int(w * 0.95)   # Skip rightmost 5%
        
        self.ambulance_roi = {
            'x1': roi_left,
            'y1': roi_top, 
            'x2': roi_right,
            'y2': roi_bottom
        }
        
        logger.debug(f"Ambulance ROI set: ({roi_left}, {roi_top}) -> ({roi_right}, {roi_bottom})")
    
    def _is_in_ambulance_roi(self, bbox: List[float]) -> bool:
        """Check if detection is within ambulance ROI"""
        if not self.roi_enabled or self.ambulance_roi is None:
            return True
            
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        roi = self.ambulance_roi
        return (roi['x1'] <= center_x <= roi['x2'] and 
                roi['y1'] <= center_y <= roi['y2'])
    
    def _detect_ambulance_features(self, frame: np.ndarray, bbox: List[float], detection_id: str) -> Dict[str, float]:
        """Advanced ambulance feature detection: flashing lights, plus marks, text, etc."""
        features = {
            'flashing_lights': 0.0,
            'plus_cross_mark': 0.0,
            'ambulance_text': 0.0,
            'emergency_colors': 0.0,
            'light_patterns': 0.0,
            'total_boost': 0.0
        }
        
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return features
                
            # Extract region
            region = frame[y1:y2, x1:x2]
            if region.size == 0:
                return features
                
            # 1. FLASHING LIGHTS DETECTION
            features['flashing_lights'] = self._detect_flashing_lights(region, detection_id)
            
            # 2. PLUS/CROSS MARK DETECTION
            features['plus_cross_mark'] = self._detect_plus_cross_mark(region)
            
            # 3. AMBULANCE TEXT DETECTION (simple pattern matching)
            features['ambulance_text'] = self._detect_ambulance_text(region)
            
            # 4. EMERGENCY COLORS PATTERN
            features['emergency_colors'] = self._detect_emergency_color_patterns(region)
            
            # 5. LIGHT BAR PATTERNS (horizontal light arrangements)
            features['light_patterns'] = self._detect_light_bar_patterns(region)
            
            # Calculate total boost with weighted importance
            total_boost = (
                features['flashing_lights'] * 0.35 +      # Most important
                features['plus_cross_mark'] * 0.25 +      # Very distinctive 
                features['ambulance_text'] * 0.15 +       # Helpful but less reliable
                features['emergency_colors'] * 0.15 +     # Supporting evidence
                features['light_patterns'] * 0.10         # Additional confirmation
            )
            
            features['total_boost'] = min(total_boost, 0.4)  # Cap at 0.4 boost
            
            return features
            
        except Exception as e:
            return features
    
    def _detect_flashing_lights(self, region: np.ndarray, detection_id: str) -> float:
        """Detect flashing emergency lights by analyzing brightness changes"""
        try:
            # Convert to grayscale for brightness analysis
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Focus on top portion where lights are usually located
            top_region = gray[:region.shape[0]//3, :]
            
            # Calculate average brightness
            current_brightness = np.mean(top_region)
            
            # Store brightness history for this detection
            if detection_id not in self.ambulance_visual_features:
                self.ambulance_visual_features[detection_id] = {'brightness_history': deque(maxlen=8)}
            
            brightness_history = self.ambulance_visual_features[detection_id]['brightness_history']
            brightness_history.append(current_brightness)
            
            if len(brightness_history) < 4:
                return 0.0
            
            # Analyze brightness variation for flashing pattern
            brightness_values = list(brightness_history)
            brightness_std = np.std(brightness_values)
            brightness_range = max(brightness_values) - min(brightness_values)
            
            # Detect periodic flashing pattern
            flashing_score = 0.0
            
            # High variation indicates possible flashing
            if brightness_std > 15 and brightness_range > 30:
                flashing_score += 0.2
                
            # Check for periodic pattern (alternating high/low)
            if len(brightness_values) >= 6:
                diffs = [brightness_values[i+1] - brightness_values[i] for i in range(len(brightness_values)-1)]
                sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
                
                # Many sign changes indicate alternating pattern
                if sign_changes >= 3:
                    flashing_score += 0.15
            
            return min(flashing_score, 0.35)
            
        except:
            return 0.0
    
    def _detect_plus_cross_mark(self, region: np.ndarray) -> float:
        """Detect red cross or plus mark symbols"""
        try:
            # Convert to HSV for better red detection
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Enhanced red detection for medical cross
            red_lower1 = np.array([0, 70, 70])      # Bright red
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 70, 70])    # Dark red
            red_upper2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in red regions
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            plus_score = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Too small
                    continue
                    
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check for cross-like shape (roughly square)
                if 0.7 <= aspect_ratio <= 1.3 and area > 100:
                    plus_score += 0.15
                    
                # Additional check for cross pattern using template matching
                if self._matches_cross_template(red_mask[y:y+h, x:x+w]):
                    plus_score += 0.1
            
            return min(plus_score, 0.25)
            
        except:
            return 0.0
    
    def _matches_cross_template(self, mask_region: np.ndarray) -> bool:
        """Check if mask region matches cross/plus pattern"""
        try:
            if mask_region.shape[0] < 10 or mask_region.shape[1] < 10:
                return False
                
            # Simple cross template check
            h, w = mask_region.shape
            center_h, center_w = h // 2, w // 2
            
            # Check horizontal line through center
            horizontal_line = mask_region[center_h-1:center_h+2, :]
            horizontal_density = np.sum(horizontal_line > 0) / horizontal_line.size
            
            # Check vertical line through center  
            vertical_line = mask_region[:, center_w-1:center_w+2]
            vertical_density = np.sum(vertical_line > 0) / vertical_line.size
            
            # Both lines should have high density for a cross
            return horizontal_density > 0.6 and vertical_density > 0.6
            
        except:
            return False
    
    def _detect_ambulance_text(self, region: np.ndarray) -> float:
        """Detect ambulance-related text patterns"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Enhance text contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Look for text-like patterns (high contrast rectangular regions)
            edges = cv2.Canny(enhanced, 50, 150)
            
            # Find contours that might be text
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_score = 0.0
            text_like_regions = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 2000:  # Text-sized regions
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Text usually has certain aspect ratios
                    if 1.5 <= aspect_ratio <= 6.0:
                        text_like_regions += 1
            
            # More text-like regions suggest ambulance text
            if text_like_regions >= 3:
                text_score = 0.15
            elif text_like_regions >= 2:
                text_score = 0.10
            elif text_like_regions >= 1:
                text_score = 0.05
                
            return text_score
            
        except:
            return 0.0
    
    def _detect_emergency_color_patterns(self, region: np.ndarray) -> float:
        """Detect specific emergency vehicle color patterns"""
        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Define emergency colors with more precision
            colors = {
                'bright_red': ([0, 120, 120], [10, 255, 255]),
                'bright_red2': ([170, 120, 120], [180, 255, 255]),
                'bright_blue': ([100, 120, 120], [130, 255, 255]),
                'white': ([0, 0, 180], [180, 30, 255]),
                'bright_yellow': ([20, 120, 120], [30, 255, 255]),
                'orange': ([10, 120, 120], [20, 255, 255])
            }
            
            color_scores = {}
            total_pixels = region.shape[0] * region.shape[1]
            
            for color_name, (lower, upper) in colors.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                percentage = np.sum(mask > 0) / total_pixels
                color_scores[color_name] = percentage
            
            pattern_score = 0.0
            
            # Classic ambulance: Red + White
            red_total = color_scores['bright_red'] + color_scores['bright_red2']
            if red_total > 0.03 and color_scores['white'] > 0.2:
                pattern_score += 0.12
            
            # Emergency lights: Red + Blue
            if red_total > 0.02 and color_scores['bright_blue'] > 0.02:
                pattern_score += 0.08
                
            # High visibility: White + bright colors
            if color_scores['white'] > 0.3 and (color_scores['bright_yellow'] > 0.05 or color_scores['orange'] > 0.05):
                pattern_score += 0.06
            
            return min(pattern_score, 0.15)
            
        except:
            return 0.0
    
    def _detect_light_bar_patterns(self, region: np.ndarray) -> float:
        """Detect horizontal light bar patterns typical of emergency vehicles"""
        try:
            # Focus on top portion where light bars are typically located
            top_region = region[:region.shape[0]//3, :]
            hsv_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
            
            # Detect bright areas (potential lights)
            _, _, v_channel = cv2.split(hsv_top)
            bright_mask = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)[1]
            
            # Find horizontal structures
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            horizontal_lines = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Count horizontal light-like structures
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            light_bar_score = 0.0
            horizontal_structures = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Significant size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Light bars are typically wide and short
                    if aspect_ratio > 3.0:
                        horizontal_structures += 1
            
            if horizontal_structures >= 2:
                light_bar_score = 0.10
            elif horizontal_structures >= 1:
                light_bar_score = 0.05
                
            return light_bar_score
            
        except:
            return 0.0
    
    def _enhance_frame_for_small_ambulances(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame to better detect small ambulances (from dedicated detector)"""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) 
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except:
            return frame  # Return original if enhancement fails
    
    def _detect_with_multiple_confidence_levels(self, frame: np.ndarray) -> List[Dict]:
        """Multi-level detection like dedicated detector for better small ambulance detection"""
        best_detections = []
        
        if not self.ambulance_model:
            return best_detections
        
        # Enhance frame for small ambulance detection
        enhanced_frame = self._enhance_frame_for_small_ambulances(frame)
        
        # Try multiple confidence levels (from dedicated detector approach)
        for conf_level in self.ambulance_confidence_levels:
            try:
                # Temporarily adjust model confidence
                original_conf = self.ambulance_model.conf_thres
                self.ambulance_model.conf_thres = conf_level
                
                # Get detections at this confidence level
                raw_detections = self.ambulance_model.detect(enhanced_frame)
                
                # Restore original confidence
                self.ambulance_model.conf_thres = original_conf
                
                # Process detections
                for detection in raw_detections:
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Enhanced validation for small ambulances
                    if self._is_valid_small_ambulance(bbox, confidence, frame.shape):
                        detection['enhanced'] = True
                        detection['conf_level'] = conf_level
                        best_detections.append(detection)
                        
                        # Early exit if we found high-confidence detection
                        if confidence > 0.15:
                            return best_detections[:1]  # Return only the best one
                            
            except Exception as e:
                continue
        
        return best_detections
    
    def _is_valid_small_ambulance(self, bbox: List[float], confidence: float, frame_shape: Tuple) -> bool:
        """Enhanced validation for small ambulances (from dedicated detector)"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Enhanced size check - more lenient for high confidence small detections
        if confidence > 0.2:  # High confidence - allow smaller areas
            min_area_threshold = 200
        elif confidence > 0.1:  # Medium confidence
            min_area_threshold = 300
        else:  # Low confidence - use standard threshold
            min_area_threshold = 400
        
        if area < min_area_threshold:
            return False
        
        # Maximum area check
        max_area_threshold = 200000
        if area > max_area_threshold:
            return False
        
        # More lenient aspect ratio for small ambulances
        if height > 0:
            aspect_ratio = width / height
            if area < 1000:  # Small ambulance - more lenient aspect ratio
                if not (0.2 <= aspect_ratio <= 4.0):
                    return False
            else:  # Normal size - standard aspect ratio
                if not (0.3 <= aspect_ratio <= 3.5):
                    return False
        
        # Minimum dimension check
        if width < 15 or height < 10:
            return False
        
        return True
    
    def _check_vehicle_overlap(self, ambulance_bbox: List[float], vehicle_detections: List[Dict]) -> bool:
        """Check if ambulance detection overlaps with any vehicle detection to reduce false positives"""
        if not self.require_vehicle_overlap or not vehicle_detections:
            return True  # Skip check if disabled or no vehicles
        
        ax1, ay1, ax2, ay2 = ambulance_bbox
        ambulance_area = (ax2 - ax1) * (ay2 - ay1)
        
        for vehicle in vehicle_detections:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            
            # Calculate intersection
            ix1 = max(ax1, vx1)
            iy1 = max(ay1, vy1)
            ix2 = min(ax2, vx2)
            iy2 = min(ay2, vy2)
            
            if ix1 < ix2 and iy1 < iy2:  # There's an intersection
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = intersection_area / ambulance_area
                
                # If ambulance overlaps significantly with a vehicle, it's likely valid
                if overlap_ratio > self.min_overlap_ratio:  # Use configurable threshold
                    return True
        
        return False  # No significant overlap found
    
    def _detect_ambulance_from_vehicles(self, vehicle_detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Fallback: Try to identify ambulances from regular vehicle detections based on visual cues"""
        if not self.use_vehicle_as_ambulance_fallback:
            return []
        
        ambulance_candidates = []
        
        for i, vehicle in enumerate(vehicle_detections):
            bbox = vehicle['bbox']
            
            # Generate detection ID for features
            detection_id = f"fallback_{i}_{self.frame_count}"
            
            # Check for ambulance visual features
            features = self._detect_ambulance_features(frame, bbox, detection_id)
            total_feature_score = features['total_boost']
            
            # If vehicle has strong ambulance features, consider it an ambulance
            if total_feature_score > 0.15:  # Significant feature presence
                ambulance_detection = {
                    'bbox': bbox,
                    'confidence': 0.02 + total_feature_score,  # Base confidence + features
                    'fallback': True,
                    'original_vehicle': vehicle,
                    'feature_score': total_feature_score,
                    'features': features,
                    'detection_id': detection_id
                }
                ambulance_candidates.append(ambulance_detection)
                
                if self.debug_ambulance:
                    logger.debug(f"FALLBACK Ambulance candidate: vehicle with feature_score={total_feature_score:.3f}")
        
        return ambulance_candidates
    
    def _apply_enhanced_temporal_analysis(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Enhanced temporal analysis with improved stability checks and detection validation"""
        frame_has_detection = len(detections) > 0
        
        # Update detection history
        self.ambulance_detection_history.append(frame_has_detection)
        
        # Process detections with advanced features and temporal validation
        enhanced_detections = []
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Check ROI
            if not self._is_in_ambulance_roi(bbox):
                continue
                
            # Generate detection ID for feature tracking
            detection_id = f"det_{i}_{self.frame_count}"
            
            # Advanced ambulance feature detection
            features = self._detect_ambulance_features(frame, bbox, detection_id)
            feature_boost = features['total_boost']
            
            # Enhanced confidence boosting for low confidence detections
            if self.adaptive_confidence_enabled and confidence < self.low_confidence_boost_threshold:
                # Apply stronger feature boost for low confidence detections
                enhanced_feature_boost = feature_boost * self.feature_boost_multiplier
                
                # Add temporal consistency boost for low confidence
                temporal_boost = 0.0
                if len(self.ambulance_detection_history) >= 3:
                    recent_detections = list(self.ambulance_detection_history)[-3:]
                    if sum(recent_detections) >= 2:  # At least 2 of last 3 frames had detection
                        temporal_boost = self.temporal_boost_for_low_conf
                
                boosted_confidence = min(confidence + enhanced_feature_boost + temporal_boost, 1.0)
                
                if self.debug_ambulance and (enhanced_feature_boost > feature_boost or temporal_boost > 0):
                    logger.debug(f"LOW-CONF BOOST: orig={confidence:.3f} → boosted={boosted_confidence:.3f} "
                               f"(feature_boost={enhanced_feature_boost:.3f}, temporal_boost={temporal_boost:.3f})")
            else:
                # Standard confidence boosting
                boosted_confidence = min(confidence + feature_boost, 1.0)
            
            # Add to confidence history
            self.ambulance_confidence_history.append(boosted_confidence)
            
            # Calculate center for position tracking
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            self.ambulance_position_history.append((center_x, center_y))
            
            # Enhanced temporal validation
            temporal_score = self._calculate_temporal_consistency_score(bbox, boosted_confidence)
            final_confidence = min(boosted_confidence * temporal_score, 1.0)
            
            # Update detection with enhanced information
            det['confidence'] = final_confidence
            det['original_confidence'] = confidence
            det['feature_boost'] = feature_boost
            det['temporal_score'] = temporal_score
            det['features'] = features
            det['detection_id'] = detection_id
            
            enhanced_detections.append(det)
        
        # Enhanced stability check with multiple criteria
        self.ambulance_stable = self._is_enhanced_stable_detection(enhanced_detections)
        
        if self.ambulance_stable:
            self.stable_frames_count += 1
        else:
            if not frame_has_detection:
                self.stable_frames_count = 0
        
        # Apply final filtering based on stability and confidence
        return self._filter_by_stability_and_confidence(enhanced_detections)
    
    def _calculate_temporal_consistency_score(self, bbox: List[float], confidence: float) -> float:
        """Calculate temporal consistency score based on detection history"""
        base_score = 1.0
        
        # Factor 1: Position consistency
        if len(self.ambulance_position_history) >= 3:
            recent_positions = list(self.ambulance_position_history)[-3:]
            current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Calculate position variance
            position_variance = self._calculate_position_variance(recent_positions + [current_center])
            
            # Lower variance = higher consistency
            if position_variance < 1000:  # Very stable
                base_score *= 1.2
            elif position_variance < 5000:  # Moderately stable
                base_score *= 1.1
            elif position_variance > 20000:  # Very unstable
                base_score *= 0.7
        
        # Factor 2: Confidence consistency
        if len(self.ambulance_confidence_history) >= 5:
            recent_confidences = list(self.ambulance_confidence_history)[-5:]
            confidence_std = np.std(recent_confidences)
            
            # Lower standard deviation = more consistent
            if confidence_std < 0.05:  # Very consistent
                base_score *= 1.15
            elif confidence_std < 0.1:   # Moderately consistent
                base_score *= 1.05
            elif confidence_std > 0.3:   # Very inconsistent
                base_score *= 0.8
        
        # Factor 3: Detection frequency in recent frames
        if len(self.ambulance_detection_history) >= 10:
            recent_detections = list(self.ambulance_detection_history)[-10:]
            detection_rate = sum(recent_detections) / len(recent_detections)
            
            # Higher detection rate = more reliable
            if detection_rate >= 0.8:    # Very frequent
                base_score *= 1.2
            elif detection_rate >= 0.6:  # Frequent
                base_score *= 1.1
            elif detection_rate < 0.3:   # Infrequent
                base_score *= 0.8
        
        return min(base_score, 1.5)  # Cap the boost
    
    def _calculate_position_variance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate variance in position coordinates"""
        if len(positions) < 2:
            return 0.0
        
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_var = np.var(x_coords) if len(x_coords) > 1 else 0
        y_var = np.var(y_coords) if len(y_coords) > 1 else 0
        
        return x_var + y_var
    
    def _is_enhanced_stable_detection(self, current_detections: List[Dict]) -> bool:
        """Enhanced stability check with multiple validation criteria"""
        if len(self.ambulance_detection_history) < self.min_tracklet_frames:
            return False
        
        # Criterion 1: Detection frequency (original approach)
        recent_detections = list(self.ambulance_detection_history)[-self.min_tracklet_frames:]
        detection_rate = sum(recent_detections) / len(recent_detections)
        
        if detection_rate < self.stability_ratio:
            return False
        
        # Criterion 2: Confidence consistency
        if len(self.ambulance_confidence_history) >= self.min_tracklet_frames:
            recent_confidences = [c for c in list(self.ambulance_confidence_history)[-self.min_tracklet_frames:] if c > 0]
            if recent_confidences:
                avg_confidence = np.mean(recent_confidences)
                confidence_std = np.std(recent_confidences)
                
                # Require both sufficient confidence and consistency
                if avg_confidence < self.min_confidence_for_stability:
                    return False
                if confidence_std > 0.25:  # Too much confidence variation
                    return False
        
        # Criterion 3: Position stability (new)
        if len(self.ambulance_position_history) >= 5:
            recent_positions = list(self.ambulance_position_history)[-5:]
            position_variance = self._calculate_position_variance(recent_positions)
            
            # Reject if position is too unstable (jumping around)
            if position_variance > 50000:  # Very large movements
                return False
        
        # Criterion 4: Current detection quality
        if current_detections:
            max_current_confidence = max(det['confidence'] for det in current_detections)
            if max_current_confidence < 0.1:  # Current detection too weak
                return False
        
        return True
    
    def _filter_by_stability_and_confidence(self, detections: List[Dict]) -> List[Dict]:
        """Apply final filtering based on stability and confidence thresholds"""
        if not detections:
            return []
        
        # If stable, use adaptive threshold for stable detections
        if self.ambulance_stable:
            filtered = []
            for det in detections:
                adaptive_threshold = self._get_adaptive_confidence_threshold(det, det['confidence'])
                # For stable detections, be more lenient
                stable_threshold = adaptive_threshold * 0.8
                
                if det['confidence'] >= stable_threshold:
                    det['stable_threshold_used'] = stable_threshold
                    filtered.append(det)
            
            if self.debug_ambulance and filtered:
                best_det = max(filtered, key=lambda x: x['confidence'])
                logger.debug(f"STABLE MODEL Ambulance {len(filtered)}: conf={best_det['confidence']:.3f} "
                           f"(orig={best_det['original_confidence']:.3f}, boost=+{best_det['feature_boost']:.3f}) "
                           f"[level={best_det.get('temporal_score', 1.0):.1f}] [stable_frames={self.stable_frames_count}]")
                
                # Print feature breakdown
                features = best_det.get('features', {})
                feature_str = " | ".join([f"{k.upper()}({v:.2f})" for k, v in features.items() 
                                        if k != 'total_boost' and v > 0])
                if feature_str:
                    logger.debug(f"   Features: {feature_str}")
            
            return filtered
        
        # If not stable, use adaptive threshold but be more conservative
        else:
            high_conf_detections = []
            for det in detections:
                adaptive_threshold = self._get_adaptive_confidence_threshold(det, det['confidence'])
                # For unstable detections, be more strict but still adaptive
                unstable_threshold = min(adaptive_threshold * 1.5, 0.25)
                
                if det['confidence'] >= unstable_threshold:
                    det['unstable_threshold_used'] = unstable_threshold
                    high_conf_detections.append(det)
            
            if self.debug_ambulance and high_conf_detections:
                logger.debug(f"HIGH-CONF Ambulance (unstable): {len(high_conf_detections)} detections")
            
            return high_conf_detections
    
    def _is_stable_detection(self) -> bool:
        """Legacy stability check - kept for compatibility"""
        return self._is_enhanced_stable_detection([])
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        if frame is None:
            return None
        
        # Initialize count line if not set
        if self.count_line_y is None:
            self.count_line_y = int(frame.shape[0] * 0.66)  # 2/3 from top
        
        # Setup ambulance ROI if not done
        self._setup_ambulance_roi(frame.shape)
        
        # Store frame for flashing detection
        self.previous_frames.append(frame.copy())
        
        # Make a copy for display
        display_frame = frame.copy()
        
        # Run vehicle detection
        raw_detections = self.vehicle_model.detect(frame)
        
        # Filter to only vehicle detections and map to generic "vehicle" class
        vehicle_detections = self._filter_vehicle_detections(raw_detections)
        
        # Enhanced ambulance detection with false positive reduction
        ambulance_detections = []
        detection_frequency = 1  # Process every frame like dedicated detector
        
        if self.ambulance_model and (self.frame_count % detection_frequency == 0):
            try:
                # Primary: Use multi-level detection with frame enhancement
                raw_ambulance_detections = self._detect_with_multiple_confidence_levels(frame)
                
                # Secondary: Fallback detection from vehicles with ambulance features
                fallback_ambulance_detections = self._detect_ambulance_from_vehicles(vehicle_detections, frame)
                
                # Combine both detection methods
                all_ambulance_detections = raw_ambulance_detections + fallback_ambulance_detections
                
                # Apply enhanced filtering pipeline
                valid_ambulance_detections = []
                
                if self.debug_ambulance and (raw_ambulance_detections or fallback_ambulance_detections):
                    logger.debug(f"Raw ambulance detections: {len(raw_ambulance_detections)}")
                    logger.debug(f"Fallback detections: {len(fallback_ambulance_detections)}")
                    logger.debug(f"Vehicle detections: {len(vehicle_detections)}")
                
                # Step 1: Apply enhanced filtering to all detections
                filtered_detections = self._filter_ambulance_detections(all_ambulance_detections, frame.shape)
                
                # Step 1.5: Special handling for very low confidence detections that might be genuine
                if not filtered_detections and all_ambulance_detections:
                    # Check if we have very low confidence detections that might be real ambulances
                    very_low_conf_detections = [det for det in all_ambulance_detections 
                                               if 0.02 <= det['confidence'] <= 0.06]
                    
                    if very_low_conf_detections:
                        # Apply special validation for very low confidence
                        special_validated = self._validate_very_low_confidence_detections(
                            very_low_conf_detections, frame, vehicle_detections)
                        filtered_detections.extend(special_validated)
                        
                        if self.debug_ambulance and special_validated:
                            logger.debug(f"Special low-conf validation: {len(special_validated)} detections rescued")
                
                if self.debug_ambulance and filtered_detections:
                    logger.debug(f"After enhanced filtering: {len(filtered_detections)} detections")
                
                # Step 2: Apply vehicle overlap check for non-fallback detections
                for i, detection in enumerate(filtered_detections):
                    bbox = detection['bbox']
                    conf = detection['confidence']
                    is_fallback = detection.get('fallback', False)
                    
                    if self.debug_ambulance:
                        detection_type = "FALLBACK" if is_fallback else "MODEL"
                        orig_conf = detection.get('original_confidence', conf)
                        calib_factor = detection.get('calibration_factor', 1.0)
                        logger.debug(f"{detection_type} Ambulance {i+1}: conf={conf:.3f} "
                                   f"(orig={orig_conf:.3f}, calib={calib_factor:.2f})")
                    
                    # Vehicle overlap check (always passes for fallback since it comes from vehicles)
                    if not is_fallback:
                        has_overlap = self._check_vehicle_overlap(bbox, vehicle_detections)
                        if not has_overlap:
                            if self.debug_ambulance:
                                logger.debug("→ FILTERED: No vehicle overlap")
                            continue
                    
                    if self.debug_ambulance:
                        detection_type = "FALLBACK" if is_fallback else "MODEL"
                        logger.debug(f"→ ACCEPTED: {detection_type} conf={conf:.3f}")
                    valid_ambulance_detections.append(detection)
                
                # Apply enhanced temporal analysis  
                ambulance_detections = self._apply_enhanced_temporal_analysis(valid_ambulance_detections, frame)
                
                # Ensure ambulance detections have correct class name for visualization
                for det in ambulance_detections:
                    det['class_name'] = 'ambulance'  # Ensure correct class for visualization
                    det['class'] = 'ambulance'       # Alternative field name
                
                # Update ambulance detection status - process all detections, not just stable ones
                if len(ambulance_detections) > 0:
                    self.ambulance_detected = True
                    self.last_ambulance_detection = self.frame_count
                    
                    logger.info(f"Ambulance detected! {len(ambulance_detections)} detections.")
                    
                    # Enhanced logging with stability and feature details
                    for i, det in enumerate(ambulance_detections):
                        features = det.get('features', {})
                        feature_boost = det.get('feature_boost', 0)
                        orig_conf = det.get('original_confidence', det['confidence'])
                        conf_level = det.get('conf_level', 'unknown')
                        is_fallback = det.get('fallback', False)
                        detection_method = "FALLBACK" if is_fallback else "MODEL"
                        stability_status = "STABLE" if self.ambulance_stable else "UNSTABLE"
                        
                        log_msg = (f"{stability_status} {detection_method} Ambulance {i+1}: "
                                  f"conf={det['confidence']:.3f} (orig={orig_conf:.3f}, "
                                  f"boost=+{feature_boost:.3f}) [level={conf_level}] "
                                  f"[stable_frames={self.stable_frames_count}]")
                        if self.ambulance_stable:
                            logger.info(log_msg)
                        else:
                            logger.debug(log_msg)
                        
                        # Log detected features
                        feature_details = []
                        if features.get('flashing_lights', 0) > 0:
                            feature_details.append(f"FLASH({features['flashing_lights']:.2f})")
                        if features.get('plus_cross_mark', 0) > 0:
                            feature_details.append(f"CROSS({features['plus_cross_mark']:.2f})")
                        if features.get('ambulance_text', 0) > 0:
                            feature_details.append(f"TEXT({features['ambulance_text']:.2f})")
                        if features.get('emergency_colors', 0) > 0:
                            feature_details.append(f"COLORS({features['emergency_colors']:.2f})")
                        if features.get('light_patterns', 0) > 0:
                            feature_details.append(f"LIGHTS({features['light_patterns']:.2f})")
                        
                        if feature_details and self.debug_ambulance:
                            logger.debug(f"   Features: {' | '.join(feature_details)}")
                        
                elif self.frame_count - self.last_ambulance_detection > self.ambulance_cooldown:
                    self.ambulance_detected = False
                    
            except Exception as e:
                logger.error(f"Error in enhanced ambulance detection: {str(e)}", exc_info=True)
        
        # Update tracker with detections (ensure ambulance detections are properly formatted)
        all_detections = vehicle_detections + ambulance_detections
        
        # Debug: Log detection info before tracking
        if self.debug_ambulance:
            logger.debug(f"Total detections for tracker: vehicles={len(vehicle_detections)}, ambulances={len(ambulance_detections)}")
            if ambulance_detections:
                logger.debug(f"Adding {len(ambulance_detections)} ambulance detections to tracker")
                for i, det in enumerate(ambulance_detections):
                    logger.debug(f"Ambulance {i+1}: class='{det.get('class_name', 'unknown')}', conf={det['confidence']:.3f}")
        
        tracked_objects = self.tracker.update(all_detections)
        
        # Check for line crossings
        for obj_id, obj in tracked_objects.items():
            if self.tracker.check_line_crossing(obj_id, self.count_line_y):
                self.vehicle_count += 1
                logger.info(f"Vehicle {obj_id} crossed the line! Total: {self.vehicle_count}")
        
        # Draw enhanced UI and detections
        self._draw_enhanced_ui(display_frame, self.fps, self.frame_count)
        self._draw_enhanced_detections(display_frame, tracked_objects)
        
        # Update frame counter and FPS
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            self._update_fps()
        
        return display_frame
    
    def _draw_enhanced_ui(self, frame, fps, frame_count):
        """Draw enhanced UI elements on the frame"""
        
        # Colors
        COLOR_WHITE = (255, 255, 255)
        COLOR_BLACK = (0, 0, 0)
        COLOR_RED = (0, 0, 255)
        COLOR_GREEN = (0, 255, 0)
        COLOR_BLUE = (255, 0, 0)  
        COLOR_YELLOW = (0, 255, 255)
        COLOR_ORANGE = (0, 165, 255)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create overlay for semi-transparent backgrounds
        overlay = frame.copy()
        
        # Draw main info panel (top-left)
        panel_width = 300
        panel_height = 150
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "TRAFFIC MONITORING SYSTEM", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        
        # System stats
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)
        cv2.putText(frame, f"Frame: {frame_count}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        cv2.putText(frame, f"Resolution: {w}x{h}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)
        
        # Vehicle count panel (top-right)
        count_panel_width = 200
        count_panel_height = 80
        cv2.rectangle(overlay, (w - count_panel_width - 10, 10), 
                      (w - 10, 10 + count_panel_height), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "VEHICLE COUNT", (w - count_panel_width, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 2)
        cv2.putText(frame, f"{self.vehicle_count}", (w - count_panel_width + 20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_GREEN, 3)
        
        # Ambulance alert (center-top) - only show if ambulance detected
        if self.ambulance_detected:
            alert_width = 280
            alert_height = 60
            alert_x = (w - alert_width) // 2
            alert_y = 20
            
            # Flashing red background
            flash_color = COLOR_RED if (frame_count // 10) % 2 == 0 else (0, 0, 150)
            cv2.rectangle(frame, (alert_x, alert_y), 
                          (alert_x + alert_width, alert_y + alert_height), flash_color, -1)
            cv2.rectangle(frame, (alert_x, alert_y), 
                          (alert_x + alert_width, alert_y + alert_height), COLOR_WHITE, 2)
            
            cv2.putText(frame, "** AMBULANCE DETECTED! **", (alert_x + 10, alert_y + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        
        # Detection statistics panel (bottom-left)
        stats_panel_width = 250
        stats_panel_height = 100
        cv2.rectangle(overlay, (10, h - stats_panel_height - 10), 
                      (10 + stats_panel_width, h - 10), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "DETECTION STATS", (20, h - stats_panel_height + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 2)
        
        # Count active detections
        active_objects = len(self.tracker.objects)
        cv2.putText(frame, f"Active Objects: {active_objects}", (20, h - stats_panel_height + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1)
        cv2.putText(frame, f"Total Crossed: {self.vehicle_count}", (20, h - stats_panel_height + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BLUE, 1)
        
        # Controls panel (bottom-right)
        controls_text = ["Controls:", "Q - Quit", "R - Reset", "S - Screenshot"]
        for i, text in enumerate(controls_text):
            cv2.putText(frame, text, (w - 150, h - 60 + i * 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_WHITE, 1)
        
        # Draw ambulance ROI if enabled
        if self.roi_enabled and self.ambulance_roi:
            roi = self.ambulance_roi
            cv2.rectangle(frame, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), 
                         (0, 255, 255), 1)  # Yellow ROI border
            cv2.putText(frame, "Ambulance ROI", (roi['x1'] + 5, roi['y1'] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def _draw_enhanced_detections(self, frame, tracked_objects):
        """Draw enhanced bounding boxes and trajectories"""
        
        # Draw trajectories first (so they appear behind boxes)
        self.tracker.draw_trajectories(frame)
        
        # Draw detection line
        line_y = self.count_line_y
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (frame.shape[1] // 2 - 80, line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw tracked objects with enhanced styling
        for obj_id, obj in tracked_objects.items():
            bbox = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']
            
            # Get appropriate color and enhanced info
            if class_name == 'ambulance':
                color = (0, 0, 255)  # Red for ambulance
                display_name = 'AMBULANCE'
                
                # Add enhanced feature info for ambulances
                features = obj.get('features', {})
                feature_boost = obj.get('feature_boost', 0)
                
                if feature_boost > 0:
                    display_name += f" +{feature_boost:.2f}"
                    
                # Add feature indicators
                feature_indicators = []
                if features.get('flashing_lights', 0) > 0.1:
                    feature_indicators.append("🚦")
                if features.get('plus_cross_mark', 0) > 0.1:
                    feature_indicators.append("➕")
                if features.get('ambulance_text', 0) > 0.05:
                    feature_indicators.append("📝")
                if features.get('emergency_colors', 0) > 0.05:
                    feature_indicators.append("🎨")
                if features.get('light_patterns', 0) > 0.05:
                    feature_indicators.append("💡")
                    
                if feature_indicators:
                    display_name += f" {''.join(feature_indicators)}"
                    
            else:
                color = (0, 255, 0)  # Green for all vehicles
                display_name = 'VEHICLE'
            
            # Draw bounding box with enhanced styling
            x1, y1, x2, y2 = map(int, bbox)
            
            # Main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Corner markers for better visibility
            corner_length = 15
            corner_thickness = 3
            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
            # Top-right corner  
            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)  
            # Bottom-left corner
            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
            # Bottom-right corner
            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
            
            # Label background
            label_text = f"{display_name} {confidence:.2f} [ID:{obj_id}]"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, color, -1)

    def _update_fps(self):
        """Update FPS counter"""
        elapsed = time.time() - self.start_time
        self.fps = self.frame_count / elapsed
    


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Traffic Detection with ONNX")
    parser.add_argument("--source", type=str, default="0", 
                       help="Video source (camera index or video file)")
    parser.add_argument("--output", type=str, default="",
                       help="Output video file (optional)")
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = ONNXTrafficDetector()
    except Exception as e:
        logger.error(f"Error initializing detector: {e}")
        return
    
    # Open video source
    if args.source.isdigit():
        # Camera
        cap = cv2.VideoCapture(int(args.source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        # Video file
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        logger.error(f"Could not open video source {args.source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output file is specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    
    logger.info("Starting detection. Press 'q' to quit...")
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video")
            break
        
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Display the frame
        cv2.imshow("Enhanced Traffic Detection System", processed_frame)
        
        # Write frame to output video
        if out is not None:
            out.write(processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.vehicle_count = 0
            detector.tracker.crossed_ids.clear()
            logger.info("Vehicle count reset!")
        elif key == ord('s'):
            screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, processed_frame)
            logger.info(f"Screenshot saved: {screenshot_name}")
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
