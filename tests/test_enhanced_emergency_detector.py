"""
Comprehensive tests for enhanced emergency vehicle detection system
"""

import unittest
import numpy as np
import cv2
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the classes to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from services.enhanced_emergency_detector import (
    EnhancedEmergencyDetector,
    IndianEmergencyDatasetCollector,
    RedCrossDetector,
    SirenLightDetector,
    EmergencyTextDetector,
    ColorSchemeDetector,
    DetectionModel,
    IndianState,
    StateSpecificConfig,
    DetectionResult
)
from models.detection import Detection, BoundingBox, EmergencyVehicle


class TestEnhancedEmergencyDetector(unittest.TestCase):
    """Test cases for EnhancedEmergencyDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            "models": {
                "primary_yolo": {
                    "model_path": "test_model.pt",
                    "confidence_threshold": 0.8,
                    "device": "cpu",
                    "weight": 0.4,
                    "enabled": False  # Disable for testing
                },
                "secondary_yolo": {
                    "model_path": "test_model2.pt",
                    "confidence_threshold": 0.7,
                    "device": "cpu",
                    "weight": 0.3,
                    "enabled": False  # Disable for testing
                },
                "visual_features": {
                    "weight": 0.3,
                    "enabled": True
                }
            },
            "ensemble": {
                "min_models": 1,  # Lower for testing
                "confidence_threshold": 0.6,
                "voting_strategy": "weighted"
            },
            "visual_features": {
                "red_cross_weight": 0.4,
                "siren_lights_weight": 0.3,
                "text_detection_weight": 0.2,
                "color_scheme_weight": 0.1
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()
        
        # Create test frame
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_config.name)
    
    def test_initialization_generic_state(self):
        """Test detector initialization with generic state"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        self.assertEqual(detector.state, IndianState.GENERIC)
        self.assertIsNotNone(detector.state_config)
        self.assertIn('red_cross', detector.visual_detectors)
        self.assertIn('siren_lights', detector.visual_detectors)
        self.assertIn('text_detector', detector.visual_detectors)
        self.assertIn('color_scheme', detector.visual_detectors)
    
    def test_initialization_maharashtra_state(self):
        """Test detector initialization with Maharashtra state"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.MAHARASHTRA
        )
        
        self.assertEqual(detector.state, IndianState.MAHARASHTRA)
        self.assertIn("108", detector.state_config.text_patterns)
        self.assertIn("महाराष्ट्र", detector.state_config.text_patterns)
        self.assertTrue(detector.state_config.design_features.get("has_108_number"))
    
    def test_initialization_delhi_state(self):
        """Test detector initialization with Delhi state"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.DELHI
        )
        
        self.assertEqual(detector.state, IndianState.DELHI)
        self.assertIn("102", detector.state_config.text_patterns)
        self.assertIn("दिल्ली", detector.state_config.text_patterns)
        self.assertTrue(detector.state_config.design_features.get("has_102_number"))
    
    def test_initialization_karnataka_state(self):
        """Test detector initialization with Karnataka state"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.KARNATAKA
        )
        
        self.assertEqual(detector.state, IndianState.KARNATAKA)
        self.assertIn("ಆಂಬುಲೆನ್ಸ್", detector.state_config.text_patterns)
        self.assertTrue(detector.state_config.design_features.get("kannada_text"))
    
    def test_config_loading_missing_file(self):
        """Test configuration loading with missing file"""
        detector = EnhancedEmergencyDetector(
            config_path="nonexistent_config.json",
            state=IndianState.GENERIC
        )
        
        # Should use default config
        self.assertIsNotNone(detector.config)
        self.assertIn("models", detector.config)
    
    def test_visual_feature_detection(self):
        """Test visual feature detection"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        # Mock visual detectors
        detector.visual_detectors['red_cross'].detect = Mock(return_value={'confidence': 0.8})
        detector.visual_detectors['siren_lights'].detect = Mock(return_value={'confidence': 0.6})
        detector.visual_detectors['text_detector'].detect = Mock(return_value={'confidence': 0.7})
        detector.visual_detectors['color_scheme'].detect = Mock(return_value={'confidence': 0.5})
        
        result = detector._detect_with_visual_features(self.test_frame)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.model_type, DetectionModel.VISUAL_FEATURES)
        self.assertGreater(result.confidence, 0.5)
    
    def test_ensemble_decision_single_high_confidence(self):
        """Test ensemble decision with single high confidence detection"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        # Create high confidence detection result
        detection = Detection(
            bbox=BoundingBox(x=100, y=100, width=200, height=150),
            class_id=0,
            class_name='ambulance',
            confidence=0.95,
            timestamp=datetime.now(),
            camera_id='test'
        )
        
        result = DetectionResult(
            model_type=DetectionModel.VISUAL_FEATURES,
            detection=detection,
            confidence=0.95,
            processing_time=0.1,
            features={}
        )
        
        emergency_vehicle = detector._ensemble_decision([result], self.test_frame)
        
        self.assertIsNotNone(emergency_vehicle)
        self.assertEqual(emergency_vehicle.vehicle_type, 'ambulance')
        self.assertGreater(emergency_vehicle.priority_score, 0.0)
    
    def test_ensemble_decision_multiple_models(self):
        """Test ensemble decision with multiple models"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        # Create multiple detection results
        detection1 = Detection(
            bbox=BoundingBox(x=100, y=100, width=200, height=150),
            class_id=0,
            class_name='ambulance',
            confidence=0.7,
            timestamp=datetime.now(),
            camera_id='test'
        )
        
        detection2 = Detection(
            bbox=BoundingBox(x=105, y=105, width=195, height=145),
            class_id=0,
            class_name='ambulance',
            confidence=0.8,
            timestamp=datetime.now(),
            camera_id='test'
        )
        
        results = [
            DetectionResult(
                model_type=DetectionModel.PRIMARY_YOLO,
                detection=detection1,
                confidence=0.7,
                processing_time=0.1,
                features={}
            ),
            DetectionResult(
                model_type=DetectionModel.VISUAL_FEATURES,
                detection=detection2,
                confidence=0.8,
                processing_time=0.05,
                features={}
            )
        ]
        
        # Mock model configs
        detector.model_configs[DetectionModel.PRIMARY_YOLO] = Mock()
        detector.model_configs[DetectionModel.PRIMARY_YOLO].weight = 0.4
        
        emergency_vehicle = detector._ensemble_decision(results, self.test_frame)
        
        self.assertIsNotNone(emergency_vehicle)
        self.assertEqual(emergency_vehicle.vehicle_type, 'ambulance')
    
    def test_priority_score_calculation(self):
        """Test priority score calculation with state-specific features"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.MAHARASHTRA
        )
        
        detection = Detection(
            bbox=BoundingBox(x=100, y=100, width=200, height=150),
            class_id=0,
            class_name='ambulance',
            confidence=0.8,
            timestamp=datetime.now(),
            camera_id='test'
        )
        
        emergency_features = {
            'red_cross': 0.9,
            'siren_lights': 0.7,
            'text_detector': 0.8,
            'color_scheme': 0.6,
            'detected_text': '108 AMBULANCE महाराष्ट्र'
        }
        
        priority_score = detector._calculate_priority_score(detection, emergency_features)
        
        self.assertGreater(priority_score, 0.0)
        self.assertLessEqual(priority_score, 1.0)
        
        # Should get state bonus for Maharashtra patterns
        self.assertGreater(priority_score, 0.5)
    
    def test_confidence_threshold_update(self):
        """Test confidence threshold update"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        new_threshold = 0.75
        detector.set_confidence_threshold(new_threshold)
        
        self.assertEqual(detector.config["ensemble"]["confidence_threshold"], new_threshold)
    
    def test_detection_stats(self):
        """Test detection statistics collection"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        # Add some mock performance data
        detector.model_performance[DetectionModel.VISUAL_FEATURES] = [
            {'confidence': 0.8, 'processing_time': 0.1, 'timestamp': datetime.now()},
            {'confidence': 0.7, 'processing_time': 0.12, 'timestamp': datetime.now()}
        ]
        
        stats = detector.get_detection_stats()
        
        self.assertIn('model_performance', stats)
        self.assertIn('state_config', stats)
        self.assertEqual(stats['state_config']['state'], 'generic')
    
    def test_dataset_collection_enable_disable(self):
        """Test dataset collection enable/disable"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            detector.enable_dataset_collection(temp_dir)
            self.assertTrue(detector.dataset_collector.is_collecting)
            
            detector.disable_dataset_collection()
            self.assertFalse(detector.dataset_collector.is_collecting)


class TestIndianEmergencyDatasetCollector(unittest.TestCase):
    """Test cases for IndianEmergencyDatasetCollector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.collector = IndianEmergencyDatasetCollector()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_start_stop_collection(self):
        """Test starting and stopping dataset collection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.collector.start_collection(temp_dir)
            
            self.assertTrue(self.collector.is_collecting)
            self.assertEqual(str(self.collector.output_dir), temp_dir)
            
            # Check directories created
            self.assertTrue((Path(temp_dir) / "images").exists())
            self.assertTrue((Path(temp_dir) / "annotations").exists())
            self.assertTrue((Path(temp_dir) / "metadata").exists())
            
            self.collector.stop_collection()
            self.assertFalse(self.collector.is_collecting)
    
    def test_collect_sample_with_detection(self):
        """Test collecting sample with emergency vehicle detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.collector.start_collection(temp_dir)
            
            # Create mock detection
            detection = Detection(
                bbox=BoundingBox(x=100, y=100, width=200, height=150),
                class_id=0,
                class_name='ambulance',
                confidence=0.8,
                timestamp=datetime.now(),
                camera_id='test'
            )
            
            emergency_vehicle = EmergencyVehicle(
                detection=detection,
                vehicle_type='ambulance',
                emergency_features={'red_cross': 0.8},
                priority_score=0.9
            )
            
            model_results = [
                DetectionResult(
                    model_type=DetectionModel.VISUAL_FEATURES,
                    detection=detection,
                    confidence=0.8,
                    processing_time=0.1,
                    features={'red_cross': 0.8}
                )
            ]
            
            self.collector.collect_sample(self.test_frame, emergency_vehicle, model_results)
            
            # Check files created
            images_dir = Path(temp_dir) / "images"
            annotations_dir = Path(temp_dir) / "annotations"
            
            self.assertTrue(len(list(images_dir.glob("*.jpg"))) > 0)
            self.assertTrue(len(list(annotations_dir.glob("*.json"))) > 0)
    
    def test_collect_sample_without_detection(self):
        """Test collecting sample without emergency vehicle detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.collector.start_collection(temp_dir)
            
            model_results = [
                DetectionResult(
                    model_type=DetectionModel.VISUAL_FEATURES,
                    detection=None,
                    confidence=0.3,
                    processing_time=0.1,
                    features={}
                )
            ]
            
            self.collector.collect_sample(self.test_frame, None, model_results)
            
            # Check files created
            images_dir = Path(temp_dir) / "images"
            annotations_dir = Path(temp_dir) / "annotations"
            
            self.assertTrue(len(list(images_dir.glob("*.jpg"))) > 0)
            self.assertTrue(len(list(annotations_dir.glob("*.json"))) > 0)


class TestVisualFeatureDetectors(unittest.TestCase):
    """Test cases for visual feature detectors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_config = StateSpecificConfig(
            state=IndianState.GENERIC,
            color_schemes=[("white_red", (255, 255, 255)), ("red_white", (0, 0, 255))],
            text_patterns=["AMBULANCE", "108", "EMERGENCY"],
            design_features={"red_cross": True, "emergency_lights": True}
        )
        
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_red_cross_detector_initialization(self):
        """Test RedCrossDetector initialization"""
        detector = RedCrossDetector(self.state_config)
        
        self.assertIsNotNone(detector.templates)
        self.assertGreater(len(detector.templates), 0)
    
    def test_red_cross_detector_empty_frame(self):
        """Test RedCrossDetector with empty frame"""
        detector = RedCrossDetector(self.state_config)
        
        empty_frame = np.array([])
        result = detector.detect(empty_frame)
        
        self.assertEqual(result['confidence'], 0.0)
    
    def test_red_cross_detector_with_red_cross(self):
        """Test RedCrossDetector with red cross pattern"""
        detector = RedCrossDetector(self.state_config)
        
        # Create frame with red cross pattern
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Vertical bar
        cv2.rectangle(frame, (45, 20), (55, 80), (0, 0, 255), -1)
        # Horizontal bar
        cv2.rectangle(frame, (20, 45), (80, 55), (0, 0, 255), -1)
        
        result = detector.detect(frame)
        
        self.assertGreater(result['confidence'], 0.0)
    
    def test_siren_light_detector_initialization(self):
        """Test SirenLightDetector initialization"""
        detector = SirenLightDetector(self.state_config)
        
        self.assertEqual(detector.state_config, self.state_config)
    
    def test_siren_light_detector_empty_frame(self):
        """Test SirenLightDetector with empty frame"""
        detector = SirenLightDetector(self.state_config)
        
        empty_frame = np.array([])
        result = detector.detect(empty_frame)
        
        self.assertEqual(result['confidence'], 0.0)
    
    def test_siren_light_detector_with_red_lights(self):
        """Test SirenLightDetector with red lights"""
        detector = SirenLightDetector(self.state_config)
        
        # Create frame with red areas
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(frame, (25, 25), 10, (0, 0, 255), -1)  # Red circle
        cv2.circle(frame, (75, 25), 10, (255, 0, 0), -1)  # Blue circle
        
        result = detector.detect(frame)
        
        self.assertGreater(result['confidence'], 0.0)
    
    def test_emergency_text_detector_initialization(self):
        """Test EmergencyTextDetector initialization"""
        detector = EmergencyTextDetector(self.state_config)
        
        self.assertEqual(detector.state_config, self.state_config)
    
    def test_emergency_text_detector_empty_frame(self):
        """Test EmergencyTextDetector with empty frame"""
        detector = EmergencyTextDetector(self.state_config)
        
        empty_frame = np.array([])
        result = detector.detect(empty_frame)
        
        self.assertEqual(result['confidence'], 0.0)
    
    def test_emergency_text_detector_with_text_pattern(self):
        """Test EmergencyTextDetector with text-like patterns"""
        detector = EmergencyTextDetector(self.state_config)
        
        # Create frame with text-like patterns (horizontal and vertical lines)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 40), (90, 45), (255, 255, 255), -1)  # Horizontal line
        cv2.rectangle(frame, (10, 55), (90, 60), (255, 255, 255), -1)  # Another horizontal line
        
        result = detector.detect(frame)
        
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertIn('detected_text', result)
    
    def test_color_scheme_detector_initialization(self):
        """Test ColorSchemeDetector initialization"""
        detector = ColorSchemeDetector(self.state_config)
        
        self.assertEqual(detector.state_config, self.state_config)
    
    def test_color_scheme_detector_empty_frame(self):
        """Test ColorSchemeDetector with empty frame"""
        detector = ColorSchemeDetector(self.state_config)
        
        empty_frame = np.array([])
        result = detector.detect(empty_frame)
        
        self.assertEqual(result['confidence'], 0.0)
    
    def test_color_scheme_detector_with_matching_colors(self):
        """Test ColorSchemeDetector with matching color scheme"""
        detector = ColorSchemeDetector(self.state_config)
        
        # Create frame with white color (matches state config)
        frame = np.full((100, 100, 3), (255, 255, 255), dtype=np.uint8)
        
        result = detector.detect(frame)
        
        self.assertGreater(result['confidence'], 0.0)


class TestMultiModelAccuracy(unittest.TestCase):
    """Test cases for multi-model detection accuracy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            "models": {
                "primary_yolo": {"enabled": False},
                "secondary_yolo": {"enabled": False},
                "visual_features": {"weight": 1.0, "enabled": True}
            },
            "ensemble": {
                "min_models": 1,
                "confidence_threshold": 0.5,
                "voting_strategy": "weighted"
            },
            "visual_features": {
                "red_cross_weight": 0.4,
                "siren_lights_weight": 0.3,
                "text_detection_weight": 0.2,
                "color_scheme_weight": 0.1
            }
        }
        
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_config, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_config.name)
    
    def test_detection_accuracy_with_clear_ambulance_features(self):
        """Test detection accuracy with clear ambulance features"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        # Create frame with ambulance-like features
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Add red cross
        cv2.rectangle(frame, (140, 80), (160, 120), (0, 0, 255), -1)  # Vertical
        cv2.rectangle(frame, (120, 95), (180, 105), (0, 0, 255), -1)  # Horizontal
        
        # Add some red areas (siren lights)
        cv2.circle(frame, (50, 50), 15, (0, 0, 255), -1)
        cv2.circle(frame, (250, 50), 15, (0, 0, 255), -1)
        
        # Add white background
        cv2.rectangle(frame, (50, 70), (250, 130), (255, 255, 255), -1)
        
        result = detector.detect_ambulance(frame)
        
        self.assertIsNotNone(result, "Should detect ambulance with clear features")
        self.assertEqual(result.vehicle_type, 'ambulance')
        self.assertGreater(result.priority_score, 0.3)
    
    def test_detection_accuracy_with_state_specific_features(self):
        """Test detection accuracy with state-specific features"""
        # Test Maharashtra-specific detection
        detector_mh = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.MAHARASHTRA
        )
        
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Mock text detection to return Maharashtra-specific text
        detector_mh.visual_detectors['text_detector'].detect = Mock(
            return_value={'confidence': 0.8, 'detected_text': '108 AMBULANCE महाराष्ट्र'}
        )
        detector_mh.visual_detectors['red_cross'].detect = Mock(
            return_value={'confidence': 0.7}
        )
        detector_mh.visual_detectors['siren_lights'].detect = Mock(
            return_value={'confidence': 0.6}
        )
        detector_mh.visual_detectors['color_scheme'].detect = Mock(
            return_value={'confidence': 0.5}
        )
        
        result = detector_mh.detect_ambulance(frame)
        
        self.assertIsNotNone(result)
        # Should get higher priority due to state-specific text
        self.assertGreater(result.priority_score, 0.5)
    
    def test_false_positive_handling(self):
        """Test handling of false positives"""
        detector = EnhancedEmergencyDetector(
            config_path=self.temp_config.name,
            state=IndianState.GENERIC
        )
        
        # Create frame with no ambulance features
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        
        result = detector.detect_ambulance(frame)
        
        # Should not detect ambulance in empty frame
        self.assertIsNone(result, "Should not detect ambulance in empty frame")
    
    def test_ensemble_voting_accuracy(self):
        """Test ensemble voting accuracy with multiple models"""
        # Enable multiple models for this test
        test_config = self.test_config.copy()
        test_config["ensemble"]["min_models"] = 2
        
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_config, temp_config)
        temp_config.close()
        
        try:
            detector = EnhancedEmergencyDetector(
                config_path=temp_config.name,
                state=IndianState.GENERIC
            )
            
            # Mock multiple detection results
            detection = Detection(
                bbox=BoundingBox(x=100, y=100, width=200, height=150),
                class_id=0,
                class_name='ambulance',
                confidence=0.8,
                timestamp=datetime.now(),
                camera_id='test'
            )
            
            results = [
                DetectionResult(
                    model_type=DetectionModel.VISUAL_FEATURES,
                    detection=detection,
                    confidence=0.7,
                    processing_time=0.1,
                    features={}
                ),
                DetectionResult(
                    model_type=DetectionModel.PRIMARY_YOLO,
                    detection=detection,
                    confidence=0.8,
                    processing_time=0.05,
                    features={}
                )
            ]
            
            # Mock model configs
            detector.model_configs[DetectionModel.PRIMARY_YOLO] = Mock()
            detector.model_configs[DetectionModel.PRIMARY_YOLO].weight = 0.4
            
            frame = np.zeros((200, 300, 3), dtype=np.uint8)
            emergency_vehicle = detector._ensemble_decision(results, frame)
            
            self.assertIsNotNone(emergency_vehicle)
            self.assertEqual(emergency_vehicle.vehicle_type, 'ambulance')
            
        finally:
            os.unlink(temp_config.name)


if __name__ == '__main__':
    unittest.main()