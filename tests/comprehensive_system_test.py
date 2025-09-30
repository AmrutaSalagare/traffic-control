"""
Comprehensive system test - validates all components working together
"""
import cv2
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
from final_tracking_onnx import ONNXTrafficDetector

def comprehensive_system_test():
    """Test all system components together"""
    
    print("="*60)
    print("COMPREHENSIVE TRAFFIC DETECTION SYSTEM TEST")
    print("="*60)
    
    # Initialize detector
    print("\n1. INITIALIZING SYSTEM...")
    try:
        detector = ONNXTrafficDetector()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # Test video loading
    print("\n2. TESTING VIDEO INPUT...")
    video_path = "videos/Untitled video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    print("✅ Video loaded successfully!")
    
    # Process test frames
    print("\n3. PROCESSING TEST FRAMES...")
    
    test_results = {
        'frames_processed': 0,
        'vehicle_detections': 0,
        'ambulance_detections': 0,
        'vehicles_tracked': 0,
        'vehicles_counted': 0,
        'processing_times': [],
        'errors': []
    }
    
    # Track unique vehicles seen
    unique_vehicles = set()
    class_counts = defaultdict(int)
    
    # Process first 100 frames for comprehensive test
    max_frames = 100
    
    for frame_num in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        try:
            # Process frame
            processed_frame = detector.process_frame(frame)
            processing_time = time.time() - start_time
            test_results['processing_times'].append(processing_time)
            
            # Count active tracked objects
            active_objects = len(detector.tracker.objects)
            unique_vehicles.update(detector.tracker.objects.keys())
            
            # Count detections by class
            for obj_id, obj in detector.tracker.objects.items():
                class_name = obj['class']
                if class_name == 'ambulance':
                    test_results['ambulance_detections'] += 1
                else:
                    test_results['vehicle_detections'] += 1
                    class_counts[class_name] += 1
            
            test_results['frames_processed'] += 1
            test_results['vehicles_tracked'] = len(unique_vehicles)
            test_results['vehicles_counted'] = detector.vehicle_count
            
            # Print progress every 20 frames
            if frame_num % 20 == 0:
                avg_time = np.mean(test_results['processing_times'][-20:]) if test_results['processing_times'] else 0
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"  Frame {frame_num:3d}: {active_objects:2d} objects, {detector.vehicle_count:2d} counted, {fps:5.1f} FPS")
            
        except Exception as e:
            test_results['errors'].append(f"Frame {frame_num}: {str(e)}")
            print(f"❌ Error processing frame {frame_num}: {e}")
    
    cap.release()
    
    # Analyze results
    print("\n4. ANALYZING RESULTS...")
    
    # Calculate performance metrics
    avg_processing_time = np.mean(test_results['processing_times']) if test_results['processing_times'] else 0
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    max_fps = 1.0 / np.min(test_results['processing_times']) if test_results['processing_times'] else 0
    min_fps = 1.0 / np.max(test_results['processing_times']) if test_results['processing_times'] else 0
    
    # Generate report
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*60)
    
    # System Performance
    print("\n📊 SYSTEM PERFORMANCE:")
    print(f"   Frames Processed: {test_results['frames_processed']}")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Max FPS: {max_fps:.1f}")
    print(f"   Min FPS: {min_fps:.1f}")
    print(f"   Avg Processing Time: {avg_processing_time*1000:.1f}ms")
    
    # Detection Performance
    print("\n🎯 DETECTION PERFORMANCE:")
    print(f"   Total Vehicle Detections: {test_results['vehicle_detections']}")
    print(f"   Total Ambulance Detections: {test_results['ambulance_detections']}")
    print(f"   Unique Vehicles Tracked: {test_results['vehicles_tracked']}")
    
    # Class Distribution
    if class_counts:
        print("\n🚗 VEHICLE CLASS DISTRIBUTION:")
        class_names = {
            '0': 'Cars', '1': 'Motorcycles', '2': 'Buses', 
            '3': 'Trucks', '4': 'Auto-rickshaws'
        }
        for class_id, count in sorted(class_counts.items()):
            class_name = class_names.get(str(class_id), f'Class {class_id}')
            percentage = (count / sum(class_counts.values())) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    # Tracking Performance
    print("\n📍 TRACKING PERFORMANCE:")
    print(f"   Vehicles Counted (Line Crossings): {test_results['vehicles_counted']}")
    print(f"   Active Objects (Final Frame): {len(detector.tracker.objects)}")
    print(f"   Tracking Success Rate: {(test_results['vehicles_tracked'] / max(test_results['frames_processed'], 1)):.2f} vehicles/frame")
    
    # Error Analysis
    print("\n⚠️  ERROR ANALYSIS:")
    if test_results['errors']:
        print(f"   Total Errors: {len(test_results['errors'])}")
        print(f"   Error Rate: {(len(test_results['errors']) / test_results['frames_processed']):.1%}")
        print("   Recent Errors:")
        for error in test_results['errors'][-3:]:  # Show last 3 errors
            print(f"     - {error}")
    else:
        print("   No errors detected! ✅")
    
    # System Health Assessment
    print("\n🏥 SYSTEM HEALTH ASSESSMENT:")
    
    health_score = 100
    issues = []
    
    # Performance checks
    if avg_fps < 10:
        health_score -= 20
        issues.append("Low FPS performance")
    elif avg_fps < 20:
        health_score -= 10
        issues.append("Moderate FPS performance")
    
    # Detection checks
    if test_results['vehicle_detections'] == 0:
        health_score -= 30
        issues.append("No vehicle detections")
    
    # Tracking checks
    if test_results['vehicles_tracked'] == 0:
        health_score -= 25
        issues.append("No vehicles tracked")
    
    # Error checks
    error_rate = len(test_results['errors']) / max(test_results['frames_processed'], 1)
    if error_rate > 0.1:
        health_score -= 25
        issues.append("High error rate")
    elif error_rate > 0.05:
        health_score -= 10
        issues.append("Moderate error rate")
    
    # Final assessment
    if health_score >= 90:
        status = "EXCELLENT ✅"
        color = "Green"
    elif health_score >= 70:
        status = "GOOD ✅"
        color = "Yellow"
    elif health_score >= 50:
        status = "FAIR ⚠️"
        color = "Orange"
    else:
        status = "POOR ❌"
        color = "Red"
    
    print(f"   Overall Health Score: {health_score}/100")
    print(f"   System Status: {status}")
    
    if issues:
        print(f"   Issues Found:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"   No issues found! System is operating optimally.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if avg_fps < 20:
        print("   - Consider reducing input resolution for better performance")
        print("   - Optimize confidence thresholds to reduce processing load")
    if test_results['vehicles_counted'] == 0:
        print("   - Check counting line position and vehicle trajectories")
        print("   - Verify tracking parameters are appropriate for video content")
    if len(test_results['errors']) > 0:
        print("   - Review error logs and fix underlying issues")
        print("   - Add additional error handling for robustness")
    
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST COMPLETED")
    print("="*60)
    
    return health_score >= 70  # Return success if health score is good or better

if __name__ == "__main__":
    success = comprehensive_system_test()
    exit(0 if success else 1)
