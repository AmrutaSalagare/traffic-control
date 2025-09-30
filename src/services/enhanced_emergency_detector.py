"""
Enhanced Emergency Vehicle Detection System with Multi-layered Architecture

This module implements a sophisticated emergency vehicle detection system designed
specifically for Indian traffic conditions with support for state-specific ambulance
designs and multi-model ensemble detection.
"""

import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from ultralytics import YOLO
import json
from pathlib import Path

try:
    from ..core.interfaces import IAmbulanceDetector
    from ..models.detection import Detection, BoundingBox, EmergencyVehicle
    from ..core.logging import get_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.interfaces import IAmbulanceDetector
    from models.detection import Detection, BoundingBox, EmergencyVehicle
    from core.logging import get_logger


class DetectionModel(Enum):
    """Available detection models"""
    PRIMARY_YOLO = "primary_yolo"
    SECONDARY_YOLO = "secondary_yolo"
    VISUAL_FEATURES = "visual_features"
    ENSEMBLE = "ensemble"


class IndianState(Enum):
    """Indian states with specific ambulance designs"""
    MAHARASHTRA = "maharashtra"
    DELHI = "delhi"
    KARNATAKA = "karnataka"
    TAMIL_NADU = "tamil_nadu"
    GUJARAT = "gujarat"
    RAJASTHAN = "rajasthan"
    UTTAR_PRADESH = "uttar_pradesh"
    WEST_BENGAL = "west_bengal"
    GENERIC = "generic"


@dataclass
class ModelConfig:
    """Configuration for detection models"""
    model_path: str
    confidence_threshold: float
    device: str
    weight: float  # Weight in ensemble voting
    enabled: bool = True
    classes: Optional[List[int]] = None  # Optional class filtering for YOLO


@dataclass
class StateSpecificConfig:
    """State-specific ambulance detection configuration"""
    state: IndianState
    color_schemes: List[Tuple[str, Tuple[int, int, int]]]  # (name, BGR color)
    text_patterns: List[str]  # Common text patterns
    design_features: Dict[str, Any]  # Specific design features
    priority_multiplier: float = 1.0


@dataclass
class DetectionResult:
    """Result from a single detection model"""
    model_type: DetectionModel
    detection: Optional[Detection]
    confidence: float
    processing_time: float
    features: Dict[str, float]


class EnhancedEmergencyDetector(IAmbulanceDetector):
    """
    Enhanced emergency vehicle detection system with multi-layered architecture
    and support for Indian state-specific ambulance designs.
    """

    def __init__(self, config_path: str = "config/emergency_detection.json",
                 camera_id: str = "default", state: IndianState = IndianState.GENERIC):
        """
        Initialize the enhanced emergency detector

        Args:
            config_path: Path to configuration file
            camera_id: Identifier for the camera source
            state: Indian state for state-specific detection
        """
        self.logger = get_logger(__name__)
        self.camera_id = camera_id
        self.state = state

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize models
        self.models: Dict[DetectionModel, Any] = {}
        self.model_configs: Dict[DetectionModel, ModelConfig] = {}

        # State-specific configuration
        self.state_config = self._get_state_config(state)

        # Visual feature detectors
        self.visual_detectors = {}

        # Performance tracking
        self.detection_history = []
        self.model_performance = {model: [] for model in DetectionModel}

        # Hysteresis state for stable detections
        self.last_detection_state = False
        self.detection_hysteresis_frames = 0
        self.hysteresis_threshold = 3  # frames to persist detection

        # Dataset collection
        self.dataset_collector = IndianEmergencyDatasetCollector()

        self.logger.info(
            f"EnhancedEmergencyDetector initialized for state: {state.value}")

        # Initialize all components
        self._initialize_models()
        self._initialize_visual_detectors()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(
                f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "models": {
                "primary_yolo": {
                    # Updated to new YOLOv11n Indian ambulance model (replaces indian_ambulance_yolov8.pt)
                    "model_path": "models/new_indian_ambulance_yolov11n.pt",
                    "confidence_threshold": 0.8,
                    "device": "cpu",
                    "weight": 0.4,
                    "enabled": True
                },
                "secondary_yolo": {
                    "model_path": "models/yolov8s.pt",
                    "confidence_threshold": 0.7,
                    "device": "cpu",
                    "weight": 0.3,
                    "enabled": True
                },
                "visual_features": {
                    "weight": 0.3,
                    "enabled": True
                }
            },
            "ensemble": {
                "min_models": 2,
                # Tighten threshold slightly for better precision
                "confidence_threshold": 0.67,
                "voting_strategy": "weighted"
            },
            "visual_features": {
                "red_cross_weight": 0.4,
                "siren_lights_weight": 0.3,
                "text_detection_weight": 0.2,
                "color_scheme_weight": 0.1
            }
        }

    def _get_state_config(self, state: IndianState) -> StateSpecificConfig:
        """Get state-specific configuration"""
        state_configs = {
            IndianState.MAHARASHTRA: StateSpecificConfig(
                state=state,
                color_schemes=[
                    ("white_red", (255, 255, 255)),
                    ("orange_white", (0, 165, 255)),
                ],
                text_patterns=["108", "AMBULANCE", "आपातकालीन", "महाराष्ट्र"],
                design_features={
                    "has_108_number": True,
                    "orange_stripe": True,
                    "state_emblem": True
                }
            ),
            IndianState.DELHI: StateSpecificConfig(
                state=state,
                color_schemes=[
                    ("white_green", (255, 255, 255)),
                    ("green_white", (0, 255, 0)),
                ],
                text_patterns=["102", "AMBULANCE", "दिल्ली", "DELHI"],
                design_features={
                    "has_102_number": True,
                    "green_stripe": True,
                    "delhi_govt_logo": True
                }
            ),
            IndianState.KARNATAKA: StateSpecificConfig(
                state=state,
                color_schemes=[
                    ("white_yellow", (255, 255, 255)),
                    ("yellow_red", (0, 255, 255)),
                ],
                text_patterns=["108", "AMBULANCE", "ಆಂಬುಲೆನ್ಸ್", "KARNATAKA"],
                design_features={
                    "has_108_number": True,
                    "yellow_stripe": True,
                    "kannada_text": True
                }
            ),
            IndianState.GENERIC: StateSpecificConfig(
                state=state,
                color_schemes=[
                    ("white_red", (255, 255, 255)),
                    ("white_blue", (255, 255, 255)),
                ],
                text_patterns=["AMBULANCE", "108", "102", "EMERGENCY"],
                design_features={
                    "red_cross": True,
                    "emergency_lights": True
                }
            )
        }

        return state_configs.get(state, state_configs[IndianState.GENERIC])

    def _initialize_models(self) -> None:
        """Initialize all detection models"""
        model_config = self.config.get("models", {})

        # Initialize primary YOLO model
        if model_config.get("primary_yolo", {}).get("enabled", True):
            try:
                primary_config = ModelConfig(**model_config["primary_yolo"])
                self.model_configs[DetectionModel.PRIMARY_YOLO] = primary_config
                model_path = primary_config.model_path
                if not Path(model_path).exists():
                    # Fallback to models/yolov8s.pt if custom file missing
                    fallback = Path("models/yolov8s.pt")
                    if fallback.exists():
                        self.logger.warning(
                            f"Primary model '{model_path}' not found. Falling back to '{fallback}'.")
                        model_path = str(fallback)
                    else:
                        self.logger.warning(
                            f"Primary model '{model_path}' not found and no fallback available. Attempting anyway.")
                        # If fallback doesn't exist and model_path has no directory, try models/ folder
                        if not os.path.dirname(model_path):
                            model_path = os.path.join("models", model_path)
                self.models[DetectionModel.PRIMARY_YOLO] = YOLO(model_path)
                # Move once to configured device
                try:
                    self.models[DetectionModel.PRIMARY_YOLO].to(
                        primary_config.device)
                    self.logger.debug(
                        f"Primary YOLO moved to {primary_config.device}")
                except Exception:
                    self.logger.debug(
                        "YOLO .to(device) not available; will rely on per-call device.")
                self.logger.info("Primary YOLO model initialized")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize primary YOLO: {str(e)}")

        # Initialize secondary YOLO model
        if model_config.get("secondary_yolo", {}).get("enabled", True):
            try:
                secondary_config = ModelConfig(
                    **model_config["secondary_yolo"])
                self.model_configs[DetectionModel.SECONDARY_YOLO] = secondary_config

                # Apply same path logic for secondary model
                model_path = secondary_config.model_path
                if not Path(model_path).exists():
                    # If model_path has no directory, try models/ folder
                    if not os.path.dirname(model_path):
                        model_path = os.path.join("models", model_path)

                self.models[DetectionModel.SECONDARY_YOLO] = YOLO(model_path)
                try:
                    self.models[DetectionModel.SECONDARY_YOLO].to(
                        secondary_config.device)
                    self.logger.debug(
                        f"Secondary YOLO moved to {secondary_config.device}")
                except Exception:
                    self.logger.debug(
                        "YOLO .to(device) not available for secondary model.")
                self.logger.info("Secondary YOLO model initialized")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize secondary YOLO: {str(e)}")

        # Warm up models
        self._warmup_models()

    def _warmup_models(self) -> None:
        """Warm up all models with dummy inference"""
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

        for model_type, model in self.models.items():
            try:
                _ = model(dummy_frame, verbose=False)
                self.logger.debug(f"Model {model_type.value} warmed up")
            except Exception as e:
                self.logger.warning(
                    f"Failed to warm up {model_type.value}: {str(e)}")

    def _initialize_visual_detectors(self) -> None:
        """Initialize visual feature detection components"""
        self.visual_detectors = {
            'red_cross': RedCrossDetector(self.state_config),
            'siren_lights': SirenLightDetector(self.state_config),
            'text_detector': EmergencyTextDetector(self.state_config),
            'color_scheme': ColorSchemeDetector(self.state_config)
        }
        self.logger.info("Visual feature detectors initialized")

    def detect_ambulance(self, frame: np.ndarray) -> Optional[EmergencyVehicle]:
        """
        Detect emergency vehicles using multi-layered ensemble approach

        Args:
            frame: Input video frame

        Returns:
            EmergencyVehicle if detected, None otherwise
        """
        if frame is None or frame.size == 0:
            return None

        start_time = time.time()

        # Collect detections from all models
        detection_results = []

        # Primary YOLO detection
        if DetectionModel.PRIMARY_YOLO in self.models:
            primary_result = self._detect_with_primary_model(frame)
            if primary_result:
                detection_results.append(primary_result)

        # Secondary YOLO detection
        if DetectionModel.SECONDARY_YOLO in self.models:
            secondary_result = self._detect_with_secondary_model(frame)
            if secondary_result:
                detection_results.append(secondary_result)

        # Visual feature detection
        visual_result = self._detect_with_visual_features(frame)
        if visual_result:
            detection_results.append(visual_result)

        # Ensemble decision making
        final_detection = self._ensemble_decision(detection_results, frame)

        # Record performance metrics
        processing_time = time.time() - start_time
        self._record_performance(detection_results, processing_time)

        # Collect data for dataset if enabled
        if self.dataset_collector.is_collecting:
            self.dataset_collector.collect_sample(
                frame, final_detection, detection_results)

        return final_detection

    def _detect_with_primary_model(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Detect using primary YOLO model"""
        model = self.models.get(DetectionModel.PRIMARY_YOLO)
        config = self.model_configs.get(DetectionModel.PRIMARY_YOLO)

        if not model or not config:
            return None

        start_time = time.time()

        try:
            results = model(
                frame,
                device=config.device,
                conf=config.confidence_threshold,
                verbose=False,
                classes=config.classes if getattr(
                    config, 'classes', None) else None
            )

            processing_time = time.time() - start_time

            if results[0].boxes is None or (hasattr(results[0].boxes, 'xyxy') and getattr(results[0].boxes.xyxy, 'shape', [0])[0] == 0):
                return DetectionResult(
                    model_type=DetectionModel.PRIMARY_YOLO,
                    detection=None,
                    confidence=0.0,
                    processing_time=processing_time,
                    features={}
                )

            # Get best detection
            best_detection = self._parse_yolo_results(
                results, DetectionModel.PRIMARY_YOLO)

            if best_detection:
                return DetectionResult(
                    model_type=DetectionModel.PRIMARY_YOLO,
                    detection=best_detection,
                    confidence=best_detection.confidence,
                    processing_time=processing_time,
                    features={"model_confidence": best_detection.confidence}
                )

            return None

        except Exception as e:
            self.logger.error(f"Primary model detection failed: {str(e)}")
            return None

    def _detect_with_secondary_model(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Detect using secondary YOLO model"""
        model = self.models.get(DetectionModel.SECONDARY_YOLO)
        config = self.model_configs.get(DetectionModel.SECONDARY_YOLO)

        if not model or not config:
            return None

        start_time = time.time()

        try:
            results = model(
                frame,
                device=config.device,
                conf=config.confidence_threshold,
                verbose=False,
                classes=config.classes if getattr(
                    config, 'classes', None) else None
            )

            processing_time = time.time() - start_time

            if results[0].boxes is None or (hasattr(results[0].boxes, 'xyxy') and getattr(results[0].boxes.xyxy, 'shape', [0])[0] == 0):
                return DetectionResult(
                    model_type=DetectionModel.SECONDARY_YOLO,
                    detection=None,
                    confidence=0.0,
                    processing_time=processing_time,
                    features={}
                )

            # Get best detection
            best_detection = self._parse_yolo_results(
                results, DetectionModel.SECONDARY_YOLO)

            if best_detection:
                return DetectionResult(
                    model_type=DetectionModel.SECONDARY_YOLO,
                    detection=best_detection,
                    confidence=best_detection.confidence,
                    processing_time=processing_time,
                    features={"model_confidence": best_detection.confidence}
                )

            return None

        except Exception as e:
            self.logger.error(f"Secondary model detection failed: {str(e)}")
            return None

    def _detect_with_visual_features(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Detect using visual feature analysis"""
        start_time = time.time()

        try:
            # Run all visual detectors
            features = {}

            for detector_name, detector in self.visual_detectors.items():
                feature_result = detector.detect(frame)
                features[detector_name] = feature_result

            processing_time = time.time() - start_time

            # Calculate overall confidence from features
            feature_weights = self.config.get("visual_features", {})

            # Extract confidence values properly
            red_cross_conf = features.get('red_cross', {}).get('confidence', 0.0) if isinstance(
                features.get('red_cross', {}), dict) else features.get('red_cross', 0.0)
            siren_lights_conf = features.get('siren_lights', {}).get('confidence', 0.0) if isinstance(
                features.get('siren_lights', {}), dict) else features.get('siren_lights', 0.0)
            text_conf = features.get('text_detector', {}).get('confidence', 0.0) if isinstance(
                features.get('text_detector', {}), dict) else features.get('text_detector', 0.0)
            color_conf = features.get('color_scheme', {}).get('confidence', 0.0) if isinstance(
                features.get('color_scheme', {}), dict) else features.get('color_scheme', 0.0)

            overall_confidence = (
                red_cross_conf * feature_weights.get('red_cross_weight', 0.4) +
                siren_lights_conf * feature_weights.get('siren_lights_weight', 0.3) +
                text_conf * feature_weights.get('text_detection_weight', 0.2) +
                color_conf * feature_weights.get('color_scheme_weight', 0.1)
            )

            # Create detection only if confidence is high enough and at least one strong cue is present
            strong_cue = (
                red_cross_conf > 0.15 or
                text_conf > 0.12 or
                siren_lights_conf > 0.20
            )
            if overall_confidence > 0.35 and strong_cue:
                # Find the best bounding box from feature detections
                best_bbox = self._get_best_bbox_from_features(
                    features, frame.shape)

                if best_bbox:
                    detection = Detection(
                        bbox=best_bbox,
                        class_id=0,  # ambulance
                        class_name='ambulance',
                        confidence=overall_confidence,
                        timestamp=datetime.now(),
                        camera_id=self.camera_id
                    )

                    return DetectionResult(
                        model_type=DetectionModel.VISUAL_FEATURES,
                        detection=detection,
                        confidence=overall_confidence,
                        processing_time=processing_time,
                        features=features
                    )

            return DetectionResult(
                model_type=DetectionModel.VISUAL_FEATURES,
                detection=None,
                confidence=overall_confidence,
                processing_time=processing_time,
                features=features
            )

        except Exception as e:
            self.logger.error(f"Visual feature detection failed: {str(e)}")
            return None

    def _parse_yolo_results(self, results, model_type: DetectionModel) -> Optional[Detection]:
        """Parse YOLO results and return best detection"""
        try:
            if results is None or results[0].boxes is None:
                return None
            boxes_tensor = results[0].boxes
            # Safeguard when no boxes present
            if not hasattr(boxes_tensor, 'xyxy'):
                return None
            boxes_np = boxes_tensor.xyxy.cpu().numpy()
            if boxes_np.size == 0 or boxes_np.shape[0] == 0:
                return None
            confidences = boxes_tensor.conf.cpu().numpy()
            if confidences.size == 0:
                return None
            class_ids = boxes_tensor.cls.cpu().numpy().astype(int)

            # Respect model class filtering; for secondary without classes, do not accept as ambulance
            cfg = self.model_configs.get(model_type)
            if model_type == DetectionModel.SECONDARY_YOLO and (cfg is None or not cfg.classes):
                return None

            # Select the best detection constrained to allowed classes if provided
            allowed = set(cfg.classes) if (cfg and cfg.classes) else None
            if allowed is not None:
                valid_indices = [i for i, cid in enumerate(
                    class_ids) if int(cid) in allowed]
                if not valid_indices:
                    return None
                best_idx = int(
                    max(valid_indices, key=lambda i: confidences[i]))
            else:
                best_idx = int(np.argmax(confidences))

            x1, y1, x2, y2 = boxes_np[best_idx]
            confidence = float(confidences[best_idx])
            class_id = int(class_ids[best_idx])

            bbox = BoundingBox(
                x=float(x1),
                y=float(y1),
                width=float(x2 - x1),
                height=float(y2 - y1)
            )

            return Detection(
                bbox=bbox,
                class_id=class_id,
                class_name='ambulance',
                confidence=confidence,
                timestamp=datetime.now(),
                camera_id=self.camera_id
            )

        except Exception as e:
            self.logger.error(f"Failed to parse YOLO results: {str(e)}")
            return None

    def _get_best_bbox_from_features(self, features: Dict[str, Any],
                                     frame_shape: Tuple[int, int, int]) -> Optional[BoundingBox]:
        """Extract best bounding box from visual feature detections"""
        # Look for bounding boxes in feature results
        bboxes = []

        for feature_name, feature_result in features.items():
            if isinstance(feature_result, dict) and 'bbox' in feature_result:
                bbox_data = feature_result['bbox']
                confidence = feature_result.get('confidence', 0.0)

                bbox = BoundingBox(
                    x=bbox_data['x'],
                    y=bbox_data['y'],
                    width=bbox_data['width'],
                    height=bbox_data['height']
                )
                bboxes.append((bbox, confidence))

        if not bboxes:
            # Create a default bbox in center of frame if no specific bbox found
            h, w = frame_shape[:2]
            return BoundingBox(
                x=w * 0.25,
                y=h * 0.25,
                width=w * 0.5,
                height=h * 0.5
            )

        # Return bbox with highest confidence
        best_bbox, _ = max(bboxes, key=lambda x: x[1])
        return best_bbox

    def _ensemble_decision(self, detection_results: List[DetectionResult],
                           frame: np.ndarray) -> Optional[EmergencyVehicle]:
        """Make ensemble decision from multiple detection results with hysteresis"""
        if not detection_results:
            # Apply hysteresis: if we were detecting before, continue for a few frames
            if self.last_detection_state and self.detection_hysteresis_frames > 0:
                self.detection_hysteresis_frames -= 1
                # Return a minimal detection to maintain continuity
                return self._create_minimal_emergency_vehicle(frame)
            else:
                self.last_detection_state = False
                self.detection_hysteresis_frames = 0
                return None

        ensemble_config = self.config.get("ensemble", {})
        min_models = ensemble_config.get("min_models", 2)
        confidence_threshold = ensemble_config.get(
            "confidence_threshold", 0.67)
        voting_strategy = ensemble_config.get("voting_strategy", "weighted")

        # Filter valid detections
        valid_detections = [
            r for r in detection_results if r.detection is not None]

        if len(valid_detections) < min_models:
            # Not enough models agree, but check if any single model has very high confidence
            for result in detection_results:
                if result.detection and result.confidence > 0.9:
                    self.last_detection_state = True
                    self.detection_hysteresis_frames = self.hysteresis_threshold
                    return self._create_emergency_vehicle(result, frame)

            # Apply hysteresis if we were detecting before
            if self.last_detection_state and self.detection_hysteresis_frames > 0:
                self.detection_hysteresis_frames -= 1
                return self._create_minimal_emergency_vehicle(frame)
            else:
                self.last_detection_state = False
                self.detection_hysteresis_frames = 0
                return None

        # Weighted voting
        if voting_strategy == "weighted":
            total_weight = 0.0
            weighted_confidence = 0.0

            for result in valid_detections:
                model_config = self.model_configs.get(result.model_type)
                weight = model_config.weight if model_config else 0.3

                total_weight += weight
                weighted_confidence += result.confidence * weight

            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
            else:
                final_confidence = np.mean(
                    [r.confidence for r in valid_detections])
        else:
            # Simple average
            final_confidence = np.mean(
                [r.confidence for r in valid_detections])

        if final_confidence >= confidence_threshold:
            # Use the detection with highest confidence as base
            best_result = max(valid_detections, key=lambda x: x.confidence)
            self.last_detection_state = True
            self.detection_hysteresis_frames = self.hysteresis_threshold
            return self._create_emergency_vehicle(best_result, frame, final_confidence)
        else:
            # Apply hysteresis if we were detecting before
            if self.last_detection_state and self.detection_hysteresis_frames > 0:
                self.detection_hysteresis_frames -= 1
                return self._create_minimal_emergency_vehicle(frame)
            else:
                self.last_detection_state = False
                self.detection_hysteresis_frames = 0
                return None

    def _create_emergency_vehicle(self, result: DetectionResult, frame: np.ndarray,
                                  override_confidence: Optional[float] = None) -> EmergencyVehicle:
        """Create EmergencyVehicle from detection result"""
        detection = result.detection

        if override_confidence is not None:
            # Create new detection with ensemble confidence
            detection = Detection(
                bbox=detection.bbox,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=override_confidence,
                timestamp=detection.timestamp,
                camera_id=detection.camera_id
            )

        # Extract ROI for additional feature analysis
        x, y, w, h = int(detection.bbox.x), int(detection.bbox.y), \
            int(detection.bbox.width), int(detection.bbox.height)

        # Ensure ROI is within bounds
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        roi = frame[y:y+h, x:x+w] if w > 0 and h > 0 else frame

        # Analyze emergency features in ROI
        emergency_features = {}
        for detector_name, detector in self.visual_detectors.items():
            feature_result = detector.detect(roi)
            if isinstance(feature_result, dict):
                emergency_features[detector_name] = feature_result.get(
                    'confidence', 0.0)
                # Also store detected text if available
                if 'detected_text' in feature_result:
                    emergency_features['detected_text'] = feature_result['detected_text']
            else:
                emergency_features[detector_name] = 0.0

        # Add model-specific features
        emergency_features.update(result.features)

        # Calculate priority score with state-specific multiplier
        priority_score = self._calculate_priority_score(
            detection, emergency_features)
        priority_score *= self.state_config.priority_multiplier

        return EmergencyVehicle(
            detection=detection,
            vehicle_type='ambulance',
            emergency_features=emergency_features,
            priority_score=min(1.0, priority_score)
        )

    def _calculate_priority_score(self, detection: Detection,
                                  emergency_features: Dict[str, float]) -> float:
        """Calculate priority score with state-specific considerations"""
        # Base score from detection confidence
        base_score = detection.confidence * 0.3

        # Feature-based scoring
        feature_score = 0.0
        feature_weights = {
            'red_cross': 0.25,
            'siren_lights': 0.20,
            'text_detector': 0.15,
            'color_scheme': 0.10
        }

        for feature, weight in feature_weights.items():
            feature_value = emergency_features.get(feature, 0.0)
            # Handle case where feature_value might be a dict
            if isinstance(feature_value, dict):
                feature_value = feature_value.get('confidence', 0.0)
            feature_score += feature_value * weight

        # State-specific bonus - make it proportional to text detector confidence
        state_bonus = 0.0
        if 'text_detector' in emergency_features:
            text_confidence = emergency_features.get('text_detector', 0.0)
            if isinstance(text_confidence, dict):
                text_confidence = text_confidence.get('confidence', 0.0)

            # Check for state-specific text patterns
            for pattern in self.state_config.text_patterns:
                if pattern.lower() in str(emergency_features.get('detected_text', '')).lower():
                    # Proportional bonus capped at 0.15, based on text confidence
                    state_bonus += min(0.15, text_confidence * 0.2)
                    break

        return min(1.0, base_score + feature_score + state_bonus)

    def _create_minimal_emergency_vehicle(self, frame: np.ndarray) -> EmergencyVehicle:
        """Create minimal emergency vehicle for hysteresis continuation"""
        # Create a basic detection with reduced confidence for hysteresis
        h, w = frame.shape[:2]
        detection = Detection(
            bbox=BoundingBox(x=w//4, y=h//4, width=w//2, height=h//2),
            class_id=0,
            class_name="ambulance",
            confidence=0.5,  # Reduced confidence for hysteresis
            timestamp=datetime.now(),
            camera_id=self.camera_id
        )

        return EmergencyVehicle(
            detection=detection,
            vehicle_type='ambulance',
            emergency_features={'hysteresis': True},
            priority_score=0.5
        )

    def _record_performance(self, results: List[DetectionResult], total_time: float) -> None:
        """Record performance metrics"""
        for result in results:
            self.model_performance[result.model_type].append({
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'timestamp': datetime.now()
            })

        self.detection_history.append({
            'total_time': total_time,
            'models_used': len(results),
            'timestamp': datetime.now()
        })

    def initialize_model(self, model_path: str, device: str = "cpu") -> None:
        """Initialize models (interface compliance)"""
        # This method is for interface compliance
        # Actual initialization is done in __init__
        pass

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for all models"""
        for model_type, config in self.model_configs.items():
            config.confidence_threshold = threshold

        # Update ensemble threshold
        self.config["ensemble"]["confidence_threshold"] = threshold

        self.logger.info(f"Confidence threshold updated to {threshold}")

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics"""
        stats = {
            'total_detections': len(self.detection_history),
            'model_performance': {},
            'state_config': {
                'state': self.state.value,
                'patterns': self.state_config.text_patterns,
                'priority_multiplier': self.state_config.priority_multiplier
            }
        }

        for model_type, performance_data in self.model_performance.items():
            if performance_data:
                stats['model_performance'][model_type.value] = {
                    'detections': len(performance_data),
                    'avg_confidence': np.mean([p['confidence'] for p in performance_data]),
                    'avg_processing_time': np.mean([p['processing_time'] for p in performance_data])
                }

        return stats

    def enable_dataset_collection(self, output_dir: str) -> None:
        """Enable dataset collection for training improvement"""
        self.dataset_collector.start_collection(output_dir)
        self.logger.info(f"Dataset collection enabled, output: {output_dir}")

    def disable_dataset_collection(self) -> None:
        """Disable dataset collection"""
        self.dataset_collector.stop_collection()
        self.logger.info("Dataset collection disabled")


class IndianEmergencyDatasetCollector:
    """
    Dataset collection and annotation framework for Indian emergency vehicles
    """

    def __init__(self):
        self.is_collecting = False
        self.output_dir = None
        self.sample_count = 0
        self.logger = get_logger(__name__)

    def start_collection(self, output_dir: str) -> None:
        """Start collecting dataset samples"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        self.is_collecting = True
        self.sample_count = 0

        self.logger.info(f"Dataset collection started: {output_dir}")

    def stop_collection(self) -> None:
        """Stop collecting dataset samples"""
        self.is_collecting = False
        self.logger.info(
            f"Dataset collection stopped. Collected {self.sample_count} samples")

    def collect_sample(self, frame: np.ndarray, detection: Optional[EmergencyVehicle],
                       model_results: List[DetectionResult]) -> None:
        """Collect a sample for the dataset"""
        if not self.is_collecting or self.output_dir is None:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sample_id = f"sample_{timestamp}_{self.sample_count:06d}"

            # Save image
            image_path = self.output_dir / "images" / f"{sample_id}.jpg"
            cv2.imwrite(str(image_path), frame)

            # Create annotation
            annotation = {
                'sample_id': sample_id,
                'timestamp': timestamp,
                'image_path': str(image_path),
                'detection': detection.to_dict() if detection else None,
                'model_results': [
                    {
                        'model_type': r.model_type.value,
                        'confidence': r.confidence,
                        'processing_time': r.processing_time,
                        'detection': r.detection.to_dict() if r.detection else None,
                        'features': r.features
                    }
                    for r in model_results
                ],
                'frame_shape': frame.shape
            }

            # Save annotation
            annotation_path = self.output_dir / \
                "annotations" / f"{sample_id}.json"
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=2, default=str)

            self.sample_count += 1

            if self.sample_count % 100 == 0:
                self.logger.info(
                    f"Collected {self.sample_count} dataset samples")

        except Exception as e:
            self.logger.error(f"Failed to collect dataset sample: {str(e)}")


# Visual Feature Detectors

class RedCrossDetector:
    """Detector for red cross symbols with state-specific variations"""

    def __init__(self, state_config: StateSpecificConfig):
        self.state_config = state_config
        self.templates = self._create_templates()

    def _create_templates(self) -> List[np.ndarray]:
        """Create red cross templates for different sizes and styles"""
        templates = []

        # Standard red cross
        for size in [20, 30, 40, 50]:
            template = np.zeros((size, size, 3), dtype=np.uint8)
            thickness = max(1, size // 10)

            # Vertical bar
            cv2.rectangle(template,
                          (size//2 - thickness, thickness),
                          (size//2 + thickness, size - thickness),
                          (0, 0, 255), -1)

            # Horizontal bar
            cv2.rectangle(template,
                          (thickness, size//2 - thickness),
                          (size - thickness, size//2 + thickness),
                          (0, 0, 255), -1)

            templates.append(template)

        return templates

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect red cross symbols in frame"""
        if frame.size == 0:
            return {'confidence': 0.0}

        max_confidence = 0.0
        best_bbox = None

        try:
            for template in self.templates:
                if template.shape[0] >= frame.shape[0] or template.shape[1] >= frame.shape[1]:
                    continue

                # Template matching
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                result = cv2.matchTemplate(
                    gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > max_confidence:
                    max_confidence = max_val
                    x, y = max_loc
                    w, h = template.shape[1], template.shape[0]
                    best_bbox = {'x': x, 'y': y, 'width': w, 'height': h}

            # Color-based validation
            red_mask = cv2.inRange(frame, (0, 0, 100), (50, 50, 255))
            red_ratio = np.sum(red_mask > 0) / \
                (frame.shape[0] * frame.shape[1])

            # Combine template matching and color analysis
            final_confidence = min(1.0, max_confidence * 0.7 + red_ratio * 2.0)

            result = {'confidence': final_confidence}
            if best_bbox:
                result['bbox'] = best_bbox

            return result

        except Exception:
            return {'confidence': 0.0}


class SirenLightDetector:
    """Detector for emergency vehicle siren lights"""

    def __init__(self, state_config: StateSpecificConfig):
        self.state_config = state_config

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect siren lights in frame"""
        if frame.size == 0:
            return {'confidence': 0.0}

        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Red light detection
            red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Blue light detection
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))

            # Bright light detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

            # Calculate confidence
            total_pixels = frame.shape[0] * frame.shape[1]
            red_ratio = np.sum(red_mask > 0) / total_pixels
            blue_ratio = np.sum(blue_mask > 0) / total_pixels
            bright_ratio = np.sum(bright_mask > 0) / total_pixels

            confidence = min(1.0, (red_ratio + blue_ratio)
                             * 3 + bright_ratio * 0.5)

            return {'confidence': confidence}

        except Exception:
            return {'confidence': 0.0}


class EmergencyTextDetector:
    """Detector for emergency-related text with state-specific patterns"""

    def __init__(self, state_config: StateSpecificConfig):
        self.state_config = state_config

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect emergency text patterns in frame"""
        if frame.size == 0:
            return {'confidence': 0.0}

        try:
            # Simple edge-based text detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Look for text-like patterns
            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, 25))

            horizontal_lines = cv2.morphologyEx(
                edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(
                edges, cv2.MORPH_OPEN, vertical_kernel)

            text_pattern = cv2.bitwise_or(horizontal_lines, vertical_lines)
            text_ratio = np.sum(text_pattern > 0) / \
                (frame.shape[0] * frame.shape[1])

            confidence = min(1.0, text_ratio * 5)

            return {
                'confidence': confidence,
                'detected_text': 'pattern_detected' if confidence > 0.3 else ''
            }

        except Exception:
            return {'confidence': 0.0}


class ColorSchemeDetector:
    """Detector for state-specific color schemes"""

    def __init__(self, state_config: StateSpecificConfig):
        self.state_config = state_config

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect state-specific color schemes"""
        if frame.size == 0:
            return {'confidence': 0.0}

        try:
            max_confidence = 0.0

            for color_name, color_bgr in self.state_config.color_schemes:
                # Create color mask
                lower = np.array([max(0, c - 30) for c in color_bgr])
                upper = np.array([min(255, c + 30) for c in color_bgr])

                mask = cv2.inRange(frame, lower, upper)
                color_ratio = np.sum(mask > 0) / \
                    (frame.shape[0] * frame.shape[1])

                confidence = min(1.0, color_ratio * 2)
                max_confidence = max(max_confidence, confidence)

            return {'confidence': max_confidence}

        except Exception:
            return {'confidence': 0.0}
