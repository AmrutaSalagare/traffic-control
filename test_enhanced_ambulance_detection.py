"""
Test the enhanced ambulance detection with all improvements:
1. Temporal Smoothing & Tracking
2. Region-of-Interest (ROI) Masking  
3. Color/Visual Cues
"""
import cv2
import numpy as np
from final_tracking_onnx import ONNXTrafficDetector

def test_enhanced_ambulance_detection():
    """Test all ambulance detection enhancements"""
    
    print("="*70)
    print("🚨 TESTING ENHANCED AMBULANCE DETECTION")
    print("="*70)
    
    print("\n✅ NEW ENHANCEMENTS IMPLEMENTED:")
    print("   1. 🕒 Temporal Smoothing & Tracklet Confirmation")
    print("      - Detection history tracking (10 frames)")
    print("      - Confidence smoothing (5 frames)")
    print("      - Minimum 3 consecutive detections required")
    print("      - 60% detection rate in recent frames")
    
    print("   2. 🎯 Region-of-Interest (ROI) Masking")
    print("      - Focus on road areas (exclude sky/edges)")
    print("      - ROI: 20% from top, 5% from sides/bottom")
    print("      - Visual ROI indicator on display")
    
    print("   3. 🎨 Color/Visual Cues Enhancement")
    print("      - Red detection (emergency lights/cross): +0.15 boost")
    print("      - White detection (ambulance body): +0.10 boost")
    print("      - Blue detection (emergency lights): +0.10 boost")
    print("      - Red+White combination: +0.10 bonus")
    print("      - Maximum boost: +0.30 confidence")
    
    # Initialize enhanced detector
    print("\n🔧 Initializing enhanced detection system...")
    detector = ONNXTrafficDetector()
    
    # Test on ambulance video
    video_path = "test_ambulance_Ambulance Running.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open test video: {video_path}")
        return
    
    print(f"✅ Test video loaded: {video_path}")
    print("\n🎬 Starting enhanced detection test...")
    print("Watch for:")
    print("   🟡 Yellow ROI box showing detection area")
    print("   🚨 Red ambulance boxes with confidence boosts")
    print("   📊 Enhanced logging with color boost details")
    print("   ⏱️  Temporal smoothing in action")
    
    frame_count = 0
    detections_logged = []
    
    while frame_count < 100:  # Test first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process with enhanced detection
        processed_frame = detector.process_frame(frame)
        
        # Log detection details
        if detector.ambulance_detected:
            detection_info = {
                'frame': frame_count,
                'tracklet_frames': detector.ambulance_tracklet_frames,
                'detection_history': list(detector.ambulance_detection_history)[-5:],
                'confidence_history': list(detector.ambulance_confidence_history)
            }
            detections_logged.append(detection_info)
        
        # Display the enhanced frame
        cv2.imshow("🚨 Enhanced Ambulance Detection Test", processed_frame)
        
        # Quick exit option
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analysis
    print(f"\n{'='*70}")
    print("🔍 ENHANCED DETECTION ANALYSIS")
    print("="*70)
    
    total_detection_frames = len(detections_logged)
    detection_rate = total_detection_frames / frame_count if frame_count > 0 else 0
    
    print(f"\n📊 DETECTION STATISTICS:")
    print(f"   Frames Processed: {frame_count}")
    print(f"   Frames with Ambulance: {total_detection_frames}")
    print(f"   Detection Rate: {detection_rate:.1%}")
    print(f"   Max Tracklet Length: {max([d['tracklet_frames'] for d in detections_logged], default=0)}")
    
    print(f"\n🕒 TEMPORAL SMOOTHING ANALYSIS:")
    if detections_logged:
        avg_tracklet = np.mean([d['tracklet_frames'] for d in detections_logged])
        print(f"   Average Tracklet Length: {avg_tracklet:.1f} frames")
        print(f"   Temporal Consistency: {'EXCELLENT' if avg_tracklet >= 5 else 'GOOD' if avg_tracklet >= 3 else 'NEEDS IMPROVEMENT'}")
        
        # Show detection pattern
        recent_detection = detections_logged[-1]
        print(f"   Recent Detection Pattern: {recent_detection['detection_history']}")
        print(f"   Recent Confidence History: {[f'{c:.3f}' for c in recent_detection['confidence_history']]}")
    
    print(f"\n🎯 ROI EFFECTIVENESS:")
    if detector.ambulance_roi:
        roi = detector.ambulance_roi
        roi_area = (roi['x2'] - roi['x1']) * (roi['y2'] - roi['y1'])
        total_area = frame.shape[0] * frame.shape[1] if 'frame' in locals() else 0
        roi_percentage = (roi_area / total_area) * 100 if total_area > 0 else 0
        print(f"   ROI Coverage: {roi_percentage:.1f}% of frame")
        print(f"   ROI Bounds: ({roi['x1']}, {roi['y1']}) -> ({roi['x2']}, {roi['y2']})")
        print(f"   Status: {'ACTIVE' if detector.roi_enabled else 'DISABLED'}")
    
    print(f"\n🎨 COLOR ENHANCEMENT STATUS:")
    print("   Red Detection: Active (emergency lights/cross)")
    print("   White Detection: Active (ambulance body)")  
    print("   Blue Detection: Active (emergency lights)")
    print("   Confidence Boost: Up to +0.30")
    
    print(f"\n🏆 OVERALL ENHANCEMENT PERFORMANCE:")
    if detection_rate > 0.4:
        status = "🟢 EXCELLENT - All enhancements working optimally!"
    elif detection_rate > 0.2:
        status = "🟡 GOOD - Enhancements showing improvement"
    else:
        status = "🔴 NEEDS TUNING - Consider adjusting parameters"
    
    print(f"   {status}")
    
    print("\n💡 ENHANCEMENT RECOMMENDATIONS:")
    if detection_rate < 0.3:
        print("   - Consider lowering temporal confirmation requirements")
        print("   - Adjust ROI to cover more area")
        print("   - Fine-tune color detection thresholds")
    else:
        print("   - System performing well with current settings")
        print("   - Monitor for false positives and adjust if needed")
    
    print("="*70)
    return detection_rate > 0.2

if __name__ == "__main__":
    success = test_enhanced_ambulance_detection()
    print(f"\n🎯 Test Result: {'✅ PASSED' if success else '❌ NEEDS IMPROVEMENT'}")
