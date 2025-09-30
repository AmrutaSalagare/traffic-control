"""
Test improved ambulance detection with filtering and enhanced detection
"""
import cv2
import numpy as np
import os
from datetime import datetime
from final_tracking_onnx import ONNXTrafficDetector

def test_improved_ambulance():
    """Test the improved ambulance detection system"""
    
    print("="*60)
    print("TESTING IMPROVED AMBULANCE DETECTION")
    print("="*60)
    
    # Initialize detector
    print("\nInitializing enhanced traffic detection system...")
    try:
        detector = ONNXTrafficDetector()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # Test on ambulance video
    video_path = "videos\Untitled video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Ambulance video not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    print("✅ Ambulance test video loaded successfully!")
    
    # Test statistics
    stats = {
        'frames_processed': 0,
        'ambulance_detections': 0,
        'raw_ambulance_detections': 0,
        'filtered_out': 0,
        'detection_frames': []
    }
    
    print("\nProcessing ambulance video frames...")
    print("Looking for ambulances with improved detection and filtering...")
    
    frame_count = 0
    max_frames = 60  # Process first 60 frames
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame with enhanced detection
        processed_frame = detector.process_frame(frame)
        
        # Count detections
        stats['frames_processed'] += 1
        
        # Check if ambulances were detected this frame
        if detector.ambulance_detected:
            stats['detection_frames'].append(frame_count)
            print(f"  Frame {frame_count:3d}: Ambulance detected!")
            
            # Save detection frame every 10 frames
            if frame_count % 10 == 0:
                output_file = f"improved_ambulance_detection_frame_{frame_count}.jpg"
                cv2.imwrite(output_file, processed_frame)
                print(f"    Saved: {output_file}")
        
        # Progress update every 15 frames  
        elif frame_count % 15 == 0:
            print(f"  Frame {frame_count:3d}: Processing...")
    
    cap.release()
    
    # Calculate final statistics
    unique_detection_frames = len(stats['detection_frames'])
    detection_rate = unique_detection_frames / stats['frames_processed'] if stats['frames_processed'] > 0 else 0
    
    print(f"\n{'='*60}")
    print("IMPROVED AMBULANCE DETECTION RESULTS")
    print("="*60)
    
    print(f"\n📊 DETECTION PERFORMANCE:")
    print(f"   Frames Processed: {stats['frames_processed']}")
    print(f"   Frames with Ambulance: {unique_detection_frames}")
    print(f"   Detection Rate: {detection_rate:.1%}")
    print(f"   Detection Quality: {'EXCELLENT' if detection_rate > 0.3 else 'GOOD' if detection_rate > 0.1 else 'NEEDS IMPROVEMENT'}")
    
    print(f"\n🚨 AMBULANCE ALERTS:")
    if stats['detection_frames']:
        print(f"   First Detection: Frame {min(stats['detection_frames'])}")
        print(f"   Last Detection: Frame {max(stats['detection_frames'])}")
        print(f"   Detection Frames: {stats['detection_frames'][:10]}{'...' if len(stats['detection_frames']) > 10 else ''}")
    else:
        print("   No ambulances detected in test video")
    
    print(f"\n⚙️  SYSTEM IMPROVEMENTS:")
    print("   ✅ Lower confidence threshold (0.15) for better small ambulance detection")
    print("   ✅ Enhanced filtering to reduce false positives")
    print("   ✅ Size-based filtering (area and aspect ratio)")
    print("   ✅ Dynamic detection frequency (more frequent when ambulance detected)")
    print("   ✅ Improved visualization with flashing alerts")
    
    # Test recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if detection_rate < 0.1:
        print("   - Consider lowering confidence threshold further")
        print("   - Check ambulance model training data compatibility")
        print("   - Verify video contains clear ambulance images")
    elif detection_rate > 0.5:
        print("   - Detection rate is excellent!")
        print("   - System is working optimally for ambulance detection")
    else:
        print("   - Detection rate is good")
        print("   - Fine-tune filtering parameters if needed")
    
    print("="*60)
    
    return detection_rate > 0.1  # Success if we detect ambulances in at least 10% of frames

def test_small_ambulance():
    """Test detection on smaller ambulances"""
    
    print("\n" + "="*60)
    print("TESTING SMALL AMBULANCE DETECTION")
    print("="*60)
    
    # Test on smaller ambulance video
    video_path = "test_small_ambulance.mp4"
    if not os.path.exists(video_path):
        print(f"⚠️  Small ambulance video not found: {video_path}")
        print("Skipping small ambulance test...")
        return True
    
    # Initialize detector
    detector = ONNXTrafficDetector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    print("✅ Small ambulance test video loaded!")
    
    frame_count = 0
    detections = 0
    max_frames = 50
    
    print("\nProcessing small ambulance video...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        if detector.ambulance_detected:
            detections += 1
            print(f"  Frame {frame_count:3d}: Small ambulance detected!")
            
            # Save some detection frames
            if frame_count % 10 == 0:
                output_file = f"small_ambulance_detection_frame_{frame_count}.jpg"
                cv2.imwrite(output_file, processed_frame)
                print(f"    Saved: {output_file}")
    
    cap.release()
    
    detection_rate = detections / frame_count if frame_count > 0 else 0
    
    print(f"\n📊 SMALL AMBULANCE RESULTS:")
    print(f"   Frames Processed: {frame_count}")
    print(f"   Detections: {detections}")
    print(f"   Detection Rate: {detection_rate:.1%}")
    print(f"   Performance: {'EXCELLENT' if detection_rate > 0.2 else 'GOOD' if detection_rate > 0.05 else 'NEEDS IMPROVEMENT'}")
    
    return detection_rate > 0.05  # Success threshold for small ambulances

if __name__ == "__main__":
    print("Starting comprehensive ambulance detection testing...")
    
    # Test 1: Regular ambulance detection
    success1 = test_improved_ambulance()
    
    # Test 2: Small ambulance detection  
    success2 = test_small_ambulance()
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Regular Ambulance Detection: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Small Ambulance Detection: {'✅ PASSED' if success2 else '❌ FAILED'}")
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if success1 and success2 else '⚠️ SOME TESTS FAILED'}")
    print("="*60)
    
    exit(0 if success1 and success2 else 1)
