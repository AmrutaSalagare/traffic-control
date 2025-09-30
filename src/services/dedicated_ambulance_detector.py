"""
🚑 Dedicated Ambulance Detector
Based on successful methodology from temp/final_ambulance_detector.py
Uses indian_ambulance_yolov11n_best.pt model specifically for ambulance detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

class DedicatedAmbulanceDetector:
    def __init__(self, model_path="models/indian_ambulance_yolov11n_best.pt"):
        """Initialize dedicated ambulance detector with proven methodology"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ambulance model not found: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"✅ Dedicated ambulance model loaded: {model_path}")
        
        # Detection parameters - optimized for speed while maintaining accuracy
        self.confidence_levels = [0.2, 0.05, 0.01]  # Reduced iterations for speed
        self.min_area = 400  # Keep small ambulance support
        self.max_area = 200000  # Prevent huge false positives
        
        # Remove multi-scale detection for speed (single scale is sufficient)
        self.use_multi_scale = False
        
        # Temporal smoothing - key to stability (RESTORED)
        self.detection_window = deque(maxlen=20)  # Original working window
        self.confidence_window = deque(maxlen=20)
        self.position_window = deque(maxlen=15)
        
        # Stability parameters - proven values (RESTORED)
        self.min_stable_frames = 3  # Original working value
        self.stability_ratio = 0.6  # Original working ratio
        self.min_confidence_for_stability = 0.015  # Original working threshold
        
        # Period consolidation parameters (RESTORED)
        self.max_gap_frames = 60  # Original working gap tolerance
        self.min_period_length = 30  # Original working minimum period
        
        # Current detection state
        self.current_detection = None
        self.is_stable = False
        self.stable_frames_count = 0
        
        # Performance optimization (reduced skipping)
        self.frame_skip_counter = 0
        self.detection_interval = 1  # Process every frame for accuracy (no skipping)
        
        print(f"🔧 Ambulance detector initialized with small ambulance optimization")
    
    def _is_valid_detection(self, box: List[float], confidence: float) -> bool:
        """Enhanced validation with special handling for small ambulances"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Enhanced size check - more lenient for high confidence small detections
        if confidence > 0.2:  # High confidence - allow smaller areas
            min_area_threshold = max(200, self.min_area * 0.5)
        elif confidence > 0.1:  # Medium confidence
            min_area_threshold = max(300, self.min_area * 0.7)
        else:  # Low confidence - use standard threshold
            min_area_threshold = self.min_area
        
        if area < min_area_threshold:
            return False
        
        # Maximum area check to prevent huge false positives
        if area > self.max_area:
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
        
        # Minimum dimension check - prevent tiny detections
        if width < 15 or height < 10:
            return False
        
        return True
    
    def _detect_best_ambulance(self, frame: np.ndarray) -> Optional[Dict]:
        """Optimized single-scale detection for maximum speed"""
        best_detection = None
        best_score = 0
        
        # Single-scale detection for speed
        for conf_threshold in self.confidence_levels:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf.item()
                    bbox = [x1, y1, x2, y2]
                    
                    if self._is_valid_detection(bbox, confidence):
                        # Simplified scoring for speed
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Simple scoring - prioritize confidence with area bonus
                        if area < 1000:  # Small ambulance
                            score = confidence * 1.3  # Small boost for small ambulances
                        else:
                            score = confidence * (1 + area / 20000)  # Standard scoring
                        
                        if score > best_score:
                            best_detection = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'score': score,
                                'area': area
                            }
                            best_score = score
                            
                # Early exit if we found a good detection
                if best_detection and best_detection['confidence'] > 0.15:
                    break
        
        return best_detection
    
    def _update_windows(self, detection: Optional[Dict]):
        """Update temporal windows for stability analysis"""
        if detection:
            self.detection_window.append(True)
            self.confidence_window.append(detection['confidence'])
            
            # Calculate center
            x1, y1, x2, y2 = detection['bbox']
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            self.position_window.append(center)
        else:
            self.detection_window.append(False)
            self.confidence_window.append(0.0)
    
    def _is_stable_detection(self) -> bool:
        """Check if current detection is stable using proven methodology (RESTORED)"""
        if len(self.detection_window) < self.min_stable_frames:
            return False
        
        # Check recent detection rate
        recent_detections = list(self.detection_window)[-self.min_stable_frames:]
        detection_rate = sum(recent_detections) / len(recent_detections)
        
        if detection_rate >= self.stability_ratio:
            # Check confidence consistency (original working logic)
            recent_confidences = [c for c in list(self.confidence_window)[-self.min_stable_frames:] if c > 0]
            if recent_confidences:
                avg_confidence = np.mean(recent_confidences)
                return avg_confidence > self.min_confidence_for_stability  # Original threshold
        
        return False
    
    def _enhance_frame_for_small_objects(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame to better detect small ambulances"""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve visibility
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
    
    def detect_ambulance(self, frame: np.ndarray) -> Dict:
        """
        Enhanced detection function with small ambulance optimization
        Returns detection result with stability information
        """
        # Frame skipping for performance optimization
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.detection_interval != 0:
            # Return previous detection state without processing
            return {
                'ambulance_detected': self.current_detection is not None,
                'stable_detection': self.is_stable,
                'detection': self.current_detection,
                'stable_frames': self.stable_frames_count,
                'confidence': self.current_detection['confidence'] if self.current_detection else 0.0,
                'bbox': self.current_detection['bbox'] if self.current_detection else None
            }
        
        # Enhance frame for better small object detection
        enhanced_frame = self._enhance_frame_for_small_objects(frame)
        
        # Detect ambulance in enhanced frame
        detection = self._detect_best_ambulance(enhanced_frame)
        
        # Update temporal windows
        self._update_windows(detection)
        
        # Check stability
        self.is_stable = self._is_stable_detection()
        
        if self.is_stable:
            self.stable_frames_count += 1
            self.current_detection = detection
        else:
            if detection is None:
                self.stable_frames_count = 0
        
        # Return comprehensive result
        result = {
            'ambulance_detected': detection is not None,
            'stable_detection': self.is_stable,
            'detection': detection,
            'stable_frames': self.stable_frames_count,
            'confidence': detection['confidence'] if detection else 0.0,
            'bbox': detection['bbox'] if detection else None
        }
        
        return result
    
    def get_detection_status(self) -> Dict:
        """Get current detection status"""
        return {
            'is_stable': self.is_stable,
            'stable_frames': self.stable_frames_count,
            'current_detection': self.current_detection,
            'detection_history_length': len(self.detection_window)
        }
    
    def reset_detection_state(self):
        """Reset detection state for new video/stream"""
        self.detection_window.clear()
        self.confidence_window.clear()
        self.position_window.clear()
        self.current_detection = None
        self.is_stable = False
        self.stable_frames_count = 0
        print("🔄 Ambulance detector state reset")

class AmbulanceTracker:
    """
    Ambulance-specific tracker that maintains ambulance presence across frames
    """
    
    def __init__(self):
        self.detector = DedicatedAmbulanceDetector()
        
        # Tracking state
        self.ambulance_present = False
        self.ambulance_bbox = None
        self.ambulance_confidence = 0.0
        self.frames_since_detection = 0
        self.max_frames_without_detection = 30  # 1 second at 30fps
        
        # Detection periods for analytics
        self.detection_periods = []
        self.current_period_start = None
        
        # Performance tracking
        self.total_frames = 0
        self.detection_frames = 0
    
    def update(self, frame: np.ndarray) -> Dict:
        """Update tracker with new frame"""
        self.total_frames += 1
        
        # Get detection result
        result = self.detector.detect_ambulance(frame)
        
        # Update tracking state
        if result['stable_detection'] and result['bbox'] is not None:
            if not self.ambulance_present:
                # Start new detection period
                self.ambulance_present = True
                self.current_period_start = self.total_frames
                # Reduced logging - only log significant events
                if self.total_frames % 30 == 0 or len(self.detection_periods) == 0:
                    print(f"🚑 AMBULANCE DETECTED at frame {self.total_frames}")
            
            # Update ambulance info with valid detection
            self.ambulance_bbox = result['bbox']
            self.ambulance_confidence = result['confidence']
            self.frames_since_detection = 0
            self.detection_frames += 1
            
        elif result['ambulance_detected'] and result['bbox'] is not None:
            # Update bbox even for unstable detections to maintain visual continuity
            self.ambulance_bbox = result['bbox']
            self.ambulance_confidence = result['confidence']
            self.frames_since_detection += 1
            
        else:
            # No valid detection - keep previous bbox for visual continuity but increment counter
            self.frames_since_detection += 1
            
            # Check if we should end current detection period
            if self.ambulance_present and self.frames_since_detection > self.max_frames_without_detection:
                # End current detection period
                if self.current_period_start is not None:
                    period_length = self.total_frames - self.current_period_start
                    self.detection_periods.append({
                        'start_frame': self.current_period_start,
                        'end_frame': self.total_frames,
                        'length': period_length
                    })
                    # Only log significant periods
                    if period_length > 15:
                        print(f"🚑 AMBULANCE PERIOD ENDED: {period_length} frames")
                
                self.ambulance_present = False
                self.ambulance_bbox = None
                self.ambulance_confidence = 0.0
                self.current_period_start = None
        
        # Return comprehensive tracking result
        tracking_result = {
            'ambulance_present': self.ambulance_present,
            'bbox': self.ambulance_bbox,
            'confidence': self.ambulance_confidence,
            'stable_detection': result['stable_detection'],
            'raw_detection': result['ambulance_detected'],
            'frames_since_detection': self.frames_since_detection,
            'detection_rate': (self.detection_frames / self.total_frames * 100) if self.total_frames > 0 else 0,
            'total_periods': len(self.detection_periods)
        }
        
        return tracking_result
    
    def get_analytics(self) -> Dict:
        """Get detection analytics"""
        return {
            'total_frames': self.total_frames,
            'detection_frames': self.detection_frames,
            'detection_rate': (self.detection_frames / self.total_frames * 100) if self.total_frames > 0 else 0,
            'detection_periods': self.detection_periods,
            'total_periods': len(self.detection_periods),
            'current_period_active': self.ambulance_present,
            'current_period_start': self.current_period_start
        }
    
    def reset(self):
        """Reset tracker state"""
        self.detector.reset_detection_state()
        self.ambulance_present = False
        self.ambulance_bbox = None
        self.ambulance_confidence = 0.0
        self.frames_since_detection = 0
        self.detection_periods = []
        self.current_period_start = None
        self.total_frames = 0
        self.detection_frames = 0
        print("🔄 Ambulance tracker reset")

# Simple interface functions for easy integration
def create_ambulance_detector(model_path: str = "models/indian_ambulance_yolov11n_best.pt") -> DedicatedAmbulanceDetector:
    """Create a dedicated ambulance detector instance"""
    return DedicatedAmbulanceDetector(model_path)

def create_ambulance_tracker() -> AmbulanceTracker:
    """Create an ambulance tracker instance"""
    return AmbulanceTracker()

def detect_ambulance_in_frame(frame: np.ndarray, detector: DedicatedAmbulanceDetector) -> Dict:
    """Simple function to detect ambulance in a single frame"""
    return detector.detect_ambulance(frame)

# Test function
def test_ambulance_detector():
    """Test the dedicated ambulance detector"""
    print("🚑 Testing Dedicated Ambulance Detector")
    print("=" * 50)
    
    try:
        detector = DedicatedAmbulanceDetector()
        print("✅ Detector initialized successfully")
        
        # Test with a dummy frame
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect_ambulance(test_frame)
        print(f"✅ Test detection completed: {result['ambulance_detected']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_ambulance_detector()