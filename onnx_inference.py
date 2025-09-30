"""
ONNX Runtime-based inference for optimized YOLO models
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional
import time

class ONNXYOLODetector:
    """ONNX Runtime-based YOLO detector for optimized inference"""
    
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45):
        """
        Initialize ONNX Runtime YOLO detector
        
        Args:
            model_path: Path to ONNX model
            conf_thres: Confidence threshold
            iou_thres: NMS IoU threshold
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Normalize path for Windows
        model_path = os.path.normpath(model_path)
        
        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in providers if p in ort.get_available_providers()]
        
        # Set session options for better performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        print(f"Loading model from: {model_path}")
        print(f"Using providers: {providers}")
        
        try:
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            
            # Get input details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape  # (batch, channel, height, width)
            
            # Handle dynamic input shapes
            self.input_height = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
            self.input_width = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640
            
            print(f"Model input shape: {self.input_shape}")
            print(f"Model output names: {[out.name for out in self.session.get_outputs()]}")
            
            # Warm up the model
            self.warmup()
            
        except Exception as e:
            print(f"Error initializing ONNX model: {str(e)}")
            raise
    
    def warmup(self):
        """Warm up the model with dummy input"""
        dummy_input = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        _ = self.session.run(None, {self.input_name: dummy_input})
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO model
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Tuple of (preprocessed image, ratio, (new_width, new_height))
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image shape
        img_height, img_width = img.shape[:2]
        
        # Calculate ratio for resizing
        ratio = min(self.input_width / img_width, self.input_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize with aspect ratio
        if (new_width, new_height) != (img_width, img_height):
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create a blank canvas with target size
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        
        # Calculate padding
        dx = (self.input_width - new_width) // 2
        dy = (self.input_height - new_height) // 2
        
        # Place the resized image on the canvas
        canvas[dy:dy+new_height, dx:dx+new_width] = img
        
        # Normalize and transpose to NCHW format
        canvas = canvas.astype(np.float32) / 255.0
        canvas = np.transpose(canvas, (2, 0, 1))  # HWC to CHW
        canvas = np.expand_dims(canvas, axis=0)   # Add batch dimension
        
        return canvas, ratio, (dx, dy)
    
    def postprocess(self, outputs: np.ndarray, ratio: float, pad: Tuple[int, int], 
                   img_shape: Tuple[int, int], conf_thres: float = None) -> List[Dict]:
        """
        Postprocess YOLO model outputs
        
        Args:
            outputs: Model outputs from ONNX runtime
            ratio: Scale ratio used in preprocessing
            pad: Padding (dx, dy) used in preprocessing
            img_shape: Original image shape (height, width)
            conf_thres: Confidence threshold
            
        Returns:
            List of detection dictionaries
        """
        if conf_thres is None:
            conf_thres = self.conf_thres
        
        img_height, img_width = img_shape
        dx, dy = pad
        
        try:
            # Get predictions - YOLO v8/v11 format: (1, 84, 8400)
            predictions = outputs[0]  # (1, 84, 8400)
            
            # Transpose to (1, 8400, 84) then squeeze to (8400, 84)
            if len(predictions.shape) == 3:
                predictions = predictions.transpose(0, 2, 1).squeeze(0)  # (8400, 84)
            
            # Extract box coordinates and class scores
            # Format: [x_center, y_center, width, height, class_0_score, class_1_score, ...]
            box_data = predictions[:, :4]  # (8400, 4) - box coordinates
            class_scores = predictions[:, 4:]  # (8400, 80) - class scores
            
            # Get the best class for each detection
            max_scores = np.max(class_scores, axis=1)  # (8400,)
            class_ids = np.argmax(class_scores, axis=1)  # (8400,)
            
            # Filter by confidence threshold
            mask = max_scores >= conf_thres
            if not np.any(mask):
                return []
                
            # Apply mask to get valid detections
            valid_boxes = box_data[mask]  # (n, 4)
            valid_scores = max_scores[mask]  # (n,)
            valid_class_ids = class_ids[mask]  # (n,)
            
            # The coordinates are already in pixel space relative to model input (640x640)
            # No need to scale them - use directly
            x_center = valid_boxes[:, 0]   
            y_center = valid_boxes[:, 1]
            width = valid_boxes[:, 2]
            height = valid_boxes[:, 3]
            
            # Convert center format to corner format (in model input space)
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Stack into boxes array
            boxes = np.column_stack([x1, y1, x2, y2])
            
            # Apply NMS in model input space
            if len(boxes) > 0:
                keep_indices = self.non_max_suppression(boxes, valid_scores, self.iou_thres)
                boxes = boxes[keep_indices]
                valid_scores = valid_scores[keep_indices]
                valid_class_ids = valid_class_ids[keep_indices]
            
            if len(boxes) == 0:
                return []
            
            # Convert from model input space to original image space
            # Step 1: Remove padding (subtract offset)
            boxes[:, 0] -= dx  # x1
            boxes[:, 1] -= dy  # y1  
            boxes[:, 2] -= dx  # x2
            boxes[:, 3] -= dy  # y2
            
            # Step 2: Scale back to original image size
            boxes /= ratio
            
            # Clip to original image boundaries
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_height)
            
            # Convert to detection format
            detections = []
            invalid_count = 0
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                
                # Skip invalid boxes (zero area after clipping)
                if x2 <= x1 or y2 <= y1:
                    invalid_count += 1
                    continue
                    
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(valid_scores[i]),
                    'class_id': int(valid_class_ids[i]),
                    'class_name': str(int(valid_class_ids[i]))
                })
            
            return detections
            
        except Exception as e:
            print(f"Error in postprocess: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @staticmethod
    def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Non-maximum suppression
        
        Args:
            boxes: [N, 4] array of boxes in format [x1, y1, x2, y2]
            scores: [N] array of scores
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Indices of kept boxes
        """
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        # Sort by score (descending)
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]
        
        keep = []
        
        while len(boxes) > 0:
            # Keep the box with highest score
            keep.append(order[0])
            
            if len(boxes) == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = ONNXYOLODetector.bbox_iou(boxes[0:1], boxes[1:])[0]
            
            # Remove boxes with IoU > threshold
            mask = ious <= iou_threshold
            boxes = boxes[1:][mask]
            order = order[1:][mask]
        
        return np.array(keep, dtype=int)
    
    @staticmethod
    def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between two sets of boxes
        
        Args:
            box1: [N, 4] array of boxes in format [x1, y1, x2, y2]
            box2: [M, 4] array of boxes in format [x1, y1, x2, y2]
            
        Returns:
            [N, M] array of IoU values
        """
        # Expand dimensions for broadcasting
        box1 = np.expand_dims(box1, 1)  # [N, 1, 4]
        box2 = np.expand_dims(box2, 0)  # [1, M, 4]
        
        # Calculate intersection coordinates
        x1 = np.maximum(box1[..., 0], box2[..., 0])  # [N, M]
        y1 = np.maximum(box1[..., 1], box2[..., 1])  # [N, M]
        x2 = np.minimum(box1[..., 2], box2[..., 2])  # [N, M]
        y2 = np.minimum(box1[..., 3], box2[..., 3])  # [N, M]
        
        # Calculate intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)  # [N, M]
        
        # Calculate union area
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])  # [N, 1]
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])  # [1, M]
        union = area1 + area2 - intersection  # [N, M]
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
        
        return iou
    
    def detect(self, img: np.ndarray, conf_thres: float = None) -> List[Dict]:
        """
        Run inference on a single image
        
        Args:
            img: Input image (BGR format)
            conf_thres: Confidence threshold (overrides class default if provided)
            
        Returns:
            List of detections, each as a dictionary with keys:
                - 'bbox': [x1, y1, x2, y2] in original image coordinates
                - 'confidence': Detection confidence
                - 'class_id': Class ID
                - 'class_name': Class name
        """
        # Store original image shape
        img_height, img_width = img.shape[:2]
        
        # Preprocess image
        img_preprocessed, ratio, pad = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: img_preprocessed})
        
        # Postprocess outputs
        detections = self.postprocess(outputs, ratio, pad, (img_height, img_width), conf_thres)
        
        return detections


class ONNXAmbulanceDetector:
    """ONNX Runtime-based ambulance detector using the optimized model"""
    
    def __init__(self, model_path: str, conf_thres: float = 0.1):
        """
        Initialize ONNX Runtime ambulance detector
        
        Args:
            model_path: Path to ONNX model
            conf_thres: Confidence threshold
        """
        self.conf_thres = conf_thres
        
        # Normalize path for Windows
        model_path = os.path.normpath(model_path)
        
        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in providers if p in ort.get_available_providers()]
        
        # Set session options for better performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        print(f"Loading ambulance model from: {model_path}")
        print(f"Using providers: {providers}")
        
        try:
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            
            # Get input details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape  # (batch, channel, height, width)
            
            # Handle dynamic input shapes
            self.input_height = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
            self.input_width = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640
            
            print(f"Ambulance model input shape: {self.input_shape}")
            print(f"Ambulance model output names: {[out.name for out in self.session.get_outputs()]}")
            
            # Warm up the model
            self.warmup()
            
        except Exception as e:
            print(f"Error initializing ambulance model: {str(e)}")
            raise
    
    def warmup(self):
        """Warm up the model with dummy input"""
        dummy_input = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        _ = self.session.run(None, {self.input_name: dummy_input})
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO model
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Tuple of (preprocessed image, ratio, (dx, dy))
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image shape
        img_height, img_width = img.shape[:2]
        
        # Calculate ratio for resizing
        ratio = min(self.input_width / img_width, self.input_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize with aspect ratio
        if (new_width, new_height) != (img_width, img_height):
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create a blank canvas with target size
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        
        # Calculate padding
        dx = (self.input_width - new_width) // 2
        dy = (self.input_height - new_height) // 2
        
        # Place the resized image on the canvas
        canvas[dy:dy+new_height, dx:dx+new_width] = img
        
        # Normalize and transpose to NCHW format
        canvas = canvas.astype(np.float32) / 255.0
        canvas = np.transpose(canvas, (2, 0, 1))  # HWC to CHW
        canvas = np.expand_dims(canvas, axis=0)   # Add batch dimension
        
        return canvas, ratio, (dx, dy)
    
    def detect(self, img: np.ndarray, conf_thres: float = None) -> List[Dict]:
        """
        Detect ambulances in the image
        
        Args:
            img: Input image (BGR format)
            conf_thres: Confidence threshold (overrides class default if provided)
            
        Returns:
            List of detections, each as a dictionary with keys:
                - 'bbox': [x1, y1, x2, y2] in original image coordinates
                - 'confidence': Detection confidence
                - 'class_id': Always 0 for ambulance
                - 'class_name': 'ambulance'
        """
        if conf_thres is None:
            conf_thres = self.conf_thres
        
        # Store original image shape
        img_height, img_width = img.shape[:2]
        
        # Preprocess image
        img_preprocessed, ratio, (dx, dy) = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: img_preprocessed})
        
        # Process outputs (assuming single class - ambulance)
        detections = []
        
        # The output format might be different, let's handle multiple formats
        try:
            # Try YOLOv8/v11 format first (1x5x8400)
            if len(outputs) == 1 and len(outputs[0].shape) == 3 and outputs[0].shape[0] == 1:
                # Output is 1x5x8400 where 5 = (x, y, w, h, conf)
                output = outputs[0]  # Get first batch (1, 5, 8400)
                
                # Transpose to (8400, 5) - correct format for processing
                output = output.transpose(0, 2, 1).squeeze(0)  # (1, 8400, 5)
                
                # Extract box coordinates and confidence
                boxes = output[:, :4]  # x_center, y_center, width, height (normalized)
                confidences = output[:, 4]  # confidence scores
                
                # Apply confidence threshold
                mask = confidences >= conf_thres
                
                if not np.any(mask):
                    return []
                
                # Filter detections
                filtered_boxes = boxes[mask]
                filtered_confidences = confidences[mask]
                
                if len(filtered_boxes) == 0:
                    return []
                
                # The coordinates are already in pixel space relative to model input (640x640)
                # Convert center coordinates to corner coordinates
                x_center = filtered_boxes[:, 0]
                y_center = filtered_boxes[:, 1]
                width = filtered_boxes[:, 2]
                height = filtered_boxes[:, 3]
                
                # Convert to corner format (in model input space)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # Stack into boxes array
                boxes_xyxy = np.column_stack([x1, y1, x2, y2])
                
                # Apply NMS in model input space
                nms_threshold = 0.4
                keep_indices = cv2.dnn.NMSBoxes(
                    boxes_xyxy.tolist(), 
                    filtered_confidences.flatten().tolist(), 
                    conf_thres, 
                    nms_threshold
                )
                
                if len(keep_indices) == 0:
                    return []
                
                # Get the filtered detections
                if isinstance(keep_indices, tuple) or isinstance(keep_indices, list):
                    keep_indices = np.array(keep_indices).flatten()
                else:  # numpy array
                    keep_indices = keep_indices.flatten()
                
                boxes_xyxy = boxes_xyxy[keep_indices]
                filtered_confidences = filtered_confidences[keep_indices]
                
                if len(boxes_xyxy) > 0:
                    print(f"Ambulance detected! {len(boxes_xyxy)} detections.")
                
                # Convert from model input space to original image space
                # Step 1: Remove padding (subtract offset)
                boxes_xyxy[:, 0] -= dx  # x1
                boxes_xyxy[:, 1] -= dy  # y1  
                boxes_xyxy[:, 2] -= dx  # x2
                boxes_xyxy[:, 3] -= dy  # y2
                
                # Step 2: Scale back to original image size
                boxes_xyxy /= ratio
                
                # Clip to original image boundaries
                boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_width)
                boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_height)
                
                # Create detections
                detections = []
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    
                    # Skip invalid boxes (zero area after clipping)
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(filtered_confidences[i]),
                        'class_id': 0,  # Ambulance class
                        'class_name': 'ambulance'
                    })
                
                return detections
                
            # Handle other output formats if needed
            else:
                return []
                
        except Exception as e:
            print(f"Error processing detections: {str(e)}")
            return []


def test_onnx_detection():
    """Test ONNX model inference with enhanced visualization"""
    import time
    from collections import deque
    
    # Paths to optimized models
    vehicle_model_path = "optimized_models/yolo11n_optimized.onnx"
    ambulance_model_path = "optimized_models/indian_ambulance_yolov11n_best_optimized.onnx"
    
    # Initialize detectors
    print("Initializing vehicle detector...")
    vehicle_detector = ONNXYOLODetector(vehicle_model_path, conf_thres=0.25)
    
    print("Initializing ambulance detector...")
    ambulance_detector = ONNXAmbulanceDetector(ambulance_model_path, conf_thres=0.1)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting inference. Press 'q' to quit, 'r' to reset counter...")
    
    # For FPS calculation
    frame_count = 0
    fps = 0
    start_time = time.time()
    
    # For ambulance detection tracking
    ambulance_count = 0
    last_ambulance_time = 0
    detection_history = deque(maxlen=30)  # Store last 30 frames' detection status
    
    # Colors and styling
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Make a copy for display
        display_frame = frame.copy()
        
        # Run vehicle detection
        vehicle_detections = vehicle_detector.detect(frame)
        
        # Run ambulance detection
        start_inference = time.time()
        ambulance_detections = ambulance_detector.detect(frame)
        inference_time = (time.time() - start_inference) * 1000  # in ms
        
        # Update detection history
        current_detection = len(ambulance_detections) > 0
        detection_history.append(current_detection)
        
        # Count consecutive detections to avoid flickering
        consecutive_detections = sum(1 for d in detection_history if d)
        is_ambulance_detected = consecutive_detections > (len(detection_history) // 2)
        
        # Update counter if new ambulance is detected
        if is_ambulance_detected and (time.time() - last_ambulance_time) > 2.0:  # 2s cooldown
            ambulance_count += 1
            last_ambulance_time = time.time()
        
        # Draw vehicle detections (semi-transparent)
        for det in vehicle_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det.get('class_name', 'vehicle')
            
            # Create overlay for semi-transparent box
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GREEN, -1)  # Filled rectangle
            cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)
            
            # Draw label with background
            label = f"{class_name} {conf:.1f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, 
                         (x1, y1 - 25), 
                         (x1 + label_size[0] + 5, y1), 
                         COLOR_GREEN, -1)
            cv2.putText(display_frame, label, (x1 + 2, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 2)
        
        # Draw ambulance detections with enhanced visualization
        for det in ambulance_detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            
            # Create pulsing effect for ambulance detection
            pulse = int(5 * (1 + np.sin(time.time() * 5) * 0.3))  # Pulsing border
            
            # Draw filled rectangle with transparency
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_RED, -1)
            cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
            
            # Draw bounding box with pulsing effect
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLOR_RED, 3)
            cv2.rectangle(display_frame, 
                         (x1 - pulse, y1 - pulse), 
                         (x2 + pulse, y2 + pulse), 
                         COLOR_RED, 2)
            
            # Draw label with background
            label = f"🚑 AMBULANCE {conf:.1f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, 
                         (x1, y1 - 35), 
                         (x1 + label_size[0] + 10, y1), 
                         COLOR_RED, -1)
            cv2.putText(display_frame, label, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        
        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:  # Update FPS every 10 frames
            fps = frame_count / (time.time() - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Draw info panel
        panel_height = 120
        cv2.rectangle(display_frame, 
                     (0, 0), 
                     (display_frame.shape[1], panel_height), 
                     (30, 30, 30), -1)
        
        # Draw FPS and inference time
        cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                   (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_WHITE, 1)
        cv2.putText(display_frame, f"Inference: {inference_time:.1f}ms", 
                   (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_WHITE, 1)
        
        # Draw ambulance counter
        counter_text = f"Ambulance Count: {ambulance_count}"
        cv2.putText(display_frame, counter_text, 
                   (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_WHITE, 1)
        
        # Draw status indicator
        status_color = COLOR_RED if is_ambulance_detected else COLOR_GREEN
        status_text = "AMBULANCE DETECTED!" if is_ambulance_detected else "No ambulance detected"
        
        # Draw status with background
        status_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.rectangle(display_frame, 
                     (display_frame.shape[1] - status_size[0] - 30, 30),
                     (display_frame.shape[1] - 10, 70),
                     (50, 50, 50), -1)
        cv2.putText(display_frame, status_text, 
                   (display_frame.shape[1] - status_size[0] - 10, 60), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
        
        # Draw instructions
        cv2.putText(display_frame, "Press 'q' to quit | 'r' to reset counter", 
                   (display_frame.shape[1] - 400, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show the frame
        cv2.imshow("Ambulance Detection System", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset counter
            ambulance_count = 0
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_onnx_detection()
