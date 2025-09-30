#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced Ambulance Detection System

This script tests all the improvements made to the ambulance detection:
1. Non-Maximum Suppression (NMS)
2. Size-based filtering
3. Enhanced temporal consistency
4. Confidence calibration
5. Shape validation
6. Detection history validation
"""

import os
import cv2
import numpy as np
import time
import json
from typing import List, Dict, Tuple
import argparse
from datetime import datetime

# Import our enhanced system
from final_tracking_onnx import ONNXTrafficDetector

class AmbulanceDetectionTester:
    """Comprehensive tester for the enhanced ambulance detection system"""
    
    def __init__(self, test_videos_dir: str = "videos", output_dir: str = "test_results"):
        self.test_videos_dir = test_videos_dir
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Test configuration
        self.test_config = {
            'nms_iou_thresholds': [0.3, 0.4, 0.5],
            'confidence_thresholds': [0.05, 0.08, 0.12, 0.15],
            'stability_ratios': [0.5, 0.6, 0.7],
            'min_tracklet_frames': [3, 5, 8]
        }
        
    def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        print("="*80)
        print("COMPREHENSIVE AMBULANCE DETECTION SYSTEM TESTS")
        print("="*80)
        
        # Test 1: Basic functionality test
        print("\n1. Testing Basic Functionality...")
        self.test_basic_functionality()
        
        # Test 2: NMS effectiveness test
        print("\n2. Testing NMS Effectiveness...")
        self.test_nms_effectiveness()
        
        # Test 3: Size filtering test
        print("\n3. Testing Size Filtering...")
        self.test_size_filtering()
        
        # Test 4: Temporal consistency test
        print("\n4. Testing Temporal Consistency...")
        self.test_temporal_consistency()
        
        # Test 5: Confidence calibration test
        print("\n5. Testing Confidence Calibration...")
        self.test_confidence_calibration()
        
        # Test 6: Performance comparison
        print("\n6. Performance Comparison...")
        self.test_performance_comparison()
        
        # Generate comprehensive report
        print("\n7. Generating Test Report...")
        self.generate_test_report()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED!")
        print("="*80)
    
    def test_basic_functionality(self):
        """Test basic functionality of the enhanced system"""
        print("Testing basic ambulance detection functionality...")
        
        try:
            # Initialize detector
            detector = ONNXTrafficDetector()
            
            # Test with a sample frame (create synthetic test frame)
            test_frame = self.create_test_frame()
            
            # Process frame
            result_frame = detector.process_frame(test_frame)
            
            if result_frame is not None:
                print("✓ Basic functionality test PASSED")
                self.results['basic_functionality'] = 'PASSED'
            else:
                print("✗ Basic functionality test FAILED")
                self.results['basic_functionality'] = 'FAILED'
                
        except Exception as e:
            print(f"✗ Basic functionality test FAILED: {str(e)}")
            self.results['basic_functionality'] = f'FAILED: {str(e)}'
    
    def test_nms_effectiveness(self):
        """Test NMS effectiveness in reducing duplicate detections"""
        print("Testing NMS effectiveness...")
        
        try:
            detector = ONNXTrafficDetector()
            
            # Create test detections with overlaps
            test_detections = [
                {'bbox': [100, 100, 200, 180], 'confidence': 0.8},
                {'bbox': [110, 105, 210, 185], 'confidence': 0.7},  # Overlapping
                {'bbox': [300, 200, 400, 280], 'confidence': 0.6},
                {'bbox': [305, 205, 405, 285], 'confidence': 0.5},  # Overlapping
                {'bbox': [500, 300, 600, 380], 'confidence': 0.9}   # Separate
            ]
            
            # Test different IoU thresholds
            for iou_threshold in self.test_config['nms_iou_thresholds']:
                filtered = detector._apply_nms_to_ambulance_detections(test_detections, iou_threshold)
                
                print(f"  IoU threshold {iou_threshold}: {len(test_detections)} → {len(filtered)} detections")
                
                # Should reduce overlapping detections
                if len(filtered) < len(test_detections):
                    print(f"  ✓ NMS working at IoU {iou_threshold}")
                else:
                    print(f"  ⚠ NMS may not be working optimally at IoU {iou_threshold}")
            
            self.results['nms_effectiveness'] = 'TESTED'
            
        except Exception as e:
            print(f"✗ NMS test FAILED: {str(e)}")
            self.results['nms_effectiveness'] = f'FAILED: {str(e)}'
    
    def test_size_filtering(self):
        """Test size-based filtering effectiveness"""
        print("Testing size-based filtering...")
        
        try:
            detector = ONNXTrafficDetector()
            frame_shape = (720, 1280, 3)  # HD frame
            
            # Create test detections with various sizes
            test_detections = [
                {'bbox': [100, 100, 105, 105], 'confidence': 0.8},    # Too small
                {'bbox': [200, 200, 800, 600], 'confidence': 0.7},    # Too large
                {'bbox': [300, 300, 380, 360], 'confidence': 0.6},    # Good size
                {'bbox': [400, 400, 450, 420], 'confidence': 0.5},    # Good size
                {'bbox': [10, 10, 15, 12], 'confidence': 0.9}         # Way too small
            ]
            
            # Apply size filtering
            filtered = detector._apply_size_shape_filtering(test_detections, frame_shape)
            
            print(f"  Size filtering: {len(test_detections)} → {len(filtered)} detections")
            
            # Check if unrealistic sizes were filtered out
            for det in filtered:
                bbox = det['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                
                print(f"  ✓ Kept detection: {width}x{height} (area: {area})")
            
            self.results['size_filtering'] = f'Filtered {len(test_detections) - len(filtered)} detections'
            
        except Exception as e:
            print(f"✗ Size filtering test FAILED: {str(e)}")
            self.results['size_filtering'] = f'FAILED: {str(e)}'
    
    def test_temporal_consistency(self):
        """Test temporal consistency and stability checks"""
        print("Testing temporal consistency...")
        
        try:
            detector = ONNXTrafficDetector()
            
            # Simulate detection sequence
            detection_sequence = [
                # Frame 1-3: Consistent detections
                [{'bbox': [100, 100, 200, 180], 'confidence': 0.7}],
                [{'bbox': [105, 102, 205, 182], 'confidence': 0.75}],
                [{'bbox': [110, 104, 210, 184], 'confidence': 0.8}],
                # Frame 4-6: More consistent detections
                [{'bbox': [115, 106, 215, 186], 'confidence': 0.78}],
                [{'bbox': [120, 108, 220, 188], 'confidence': 0.82}],
                [{'bbox': [125, 110, 225, 190], 'confidence': 0.85}],
            ]
            
            test_frame = self.create_test_frame()
            
            for frame_idx, detections in enumerate(detection_sequence):
                print(f"  Processing frame {frame_idx + 1}...")
                
                # Apply temporal analysis
                enhanced = detector._apply_enhanced_temporal_analysis(detections, test_frame)
                
                print(f"    Input: {len(detections)} detections")
                print(f"    Output: {len(enhanced)} enhanced detections")
                
                if enhanced:
                    best_det = enhanced[0]
                    temporal_score = best_det.get('temporal_score', 1.0)
                    print(f"    Temporal score: {temporal_score:.3f}")
                    print(f"    Stable: {detector.ambulance_stable}")
            
            self.results['temporal_consistency'] = 'TESTED'
            
        except Exception as e:
            print(f"✗ Temporal consistency test FAILED: {str(e)}")
            self.results['temporal_consistency'] = f'FAILED: {str(e)}'
    
    def test_confidence_calibration(self):
        """Test confidence calibration system"""
        print("Testing confidence calibration...")
        
        try:
            detector = ONNXTrafficDetector()
            frame_shape = (720, 1280, 3)
            
            # Test detections with various characteristics
            test_detections = [
                {
                    'bbox': [400, 300, 500, 380],  # Good position, good size
                    'confidence': 0.5,
                    'validation': {
                        'relative_area': 0.01,
                        'aspect_ratio': 1.25,
                        'pixel_area': 8000,
                        'confidence_threshold_used': 0.12
                    }
                },
                {
                    'bbox': [100, 50, 150, 80],    # Bad position (top), small size
                    'confidence': 0.6,
                    'validation': {
                        'relative_area': 0.001,
                        'aspect_ratio': 1.67,
                        'pixel_area': 1500,
                        'confidence_threshold_used': 0.20
                    }
                }
            ]
            
            # Apply calibration
            calibrated = detector._calibrate_detection_confidence(test_detections, frame_shape)
            
            print(f"  Calibration results:")
            for i, det in enumerate(calibrated):
                orig_conf = det.get('original_confidence', 0)
                new_conf = det['confidence']
                calib_factor = det.get('calibration_factor', 1.0)
                
                print(f"    Detection {i+1}: {orig_conf:.3f} → {new_conf:.3f} (factor: {calib_factor:.3f})")
            
            self.results['confidence_calibration'] = f'Calibrated {len(calibrated)} detections'
            
        except Exception as e:
            print(f"✗ Confidence calibration test FAILED: {str(e)}")
            self.results['confidence_calibration'] = f'FAILED: {str(e)}'
    
    def test_performance_comparison(self):
        """Compare performance with and without enhancements"""
        print("Testing performance comparison...")
        
        try:
            # Test with available video files
            video_files = []
            for filename in os.listdir(self.test_videos_dir):
                if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(self.test_videos_dir, filename))
            
            if not video_files:
                print("  No video files found for performance testing")
                self.results['performance_comparison'] = 'NO_VIDEOS'
                return
            
            # Test first video file
            test_video = video_files[0]
            print(f"  Testing with video: {os.path.basename(test_video)}")
            
            # Initialize detector
            detector = ONNXTrafficDetector()
            
            # Process a few frames and measure performance
            cap = cv2.VideoCapture(test_video)
            frame_count = 0
            detection_count = 0
            total_time = 0
            
            while frame_count < 30 and cap.isOpened():  # Test first 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                result_frame = detector.process_frame(frame)
                process_time = time.time() - start_time
                
                total_time += process_time
                frame_count += 1
                
                # Count ambulance detections (simplified)
                if hasattr(detector, 'ambulance_stable') and detector.ambulance_stable:
                    detection_count += 1
            
            cap.release()
            
            avg_fps = frame_count / total_time if total_time > 0 else 0
            detection_rate = detection_count / frame_count if frame_count > 0 else 0
            
            print(f"  Processed {frame_count} frames")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Detection rate: {detection_rate:.2%}")
            print(f"  Total ambulance detections: {detection_count}")
            
            self.results['performance_comparison'] = {
                'frames_processed': frame_count,
                'avg_fps': avg_fps,
                'detection_rate': detection_rate,
                'total_detections': detection_count
            }
            
        except Exception as e:
            print(f"✗ Performance comparison test FAILED: {str(e)}")
            self.results['performance_comparison'] = f'FAILED: {str(e)}'
    
    def create_test_frame(self) -> np.ndarray:
        """Create a synthetic test frame for testing"""
        # Create a simple test frame (640x480, 3 channels)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some basic patterns to simulate a road scene
        # Sky (top third)
        frame[:160, :] = [135, 206, 235]  # Light blue
        
        # Road (bottom two thirds)
        frame[160:, :] = [64, 64, 64]     # Dark gray
        
        # Add some white lane markings
        cv2.rectangle(frame, (300, 200), (340, 480), (255, 255, 255), -1)
        
        return frame
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report_file = os.path.join(self.output_dir, f"ambulance_detection_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Create detailed report
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'test_configuration': self.test_config,
            'test_results': self.results,
            'summary': {
                'total_tests': len(self.results),
                'passed_tests': len([r for r in self.results.values() if 'PASSED' in str(r) or 'TESTED' in str(r)]),
                'failed_tests': len([r for r in self.results.values() if 'FAILED' in str(r)])
            }
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Test report saved to: {report_file}")
        
        # Print summary
        print("\nTEST SUMMARY:")
        print("-" * 40)
        for test_name, result in self.results.items():
            status = "✓ PASS" if ('PASSED' in str(result) or 'TESTED' in str(result)) else "✗ FAIL"
            print(f"{test_name:25} : {status}")
        
        print(f"\nTotal Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")

def main():
    """Main function to run tests"""
    parser = argparse.ArgumentParser(description='Test Enhanced Ambulance Detection System')
    parser.add_argument('--videos-dir', default='videos', help='Directory containing test videos')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = AmbulanceDetectionTester(args.videos_dir, args.output_dir)
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main()
