"""
Test Advanced Ambulance Visual Cues Detection:
- Flashing lights detection
- Plus/cross mark detection  
- Ambulance text detection
- Emergency color patterns
- Light bar patterns
"""
import cv2
import numpy as np
from datetime import datetime
from final_tracking_onnx import ONNXTrafficDetector

def test_advanced_ambulance_features():
    """Test all advanced ambulance visual cues"""
    
    print("="*80)
    print("🚨🔍 ADVANCED AMBULANCE VISUAL CUES DETECTION TEST")
    print("="*80)
    
    print("\n🎯 ADVANCED FEATURES BEING TESTED:")
    print("   1. 🚦 FLASHING LIGHTS DETECTION")
    print("      - Brightness variation analysis")
    print("      - Periodic flashing pattern detection")
    print("      - Emergency light color detection")
    print("      - Confidence boost: up to +0.35")
    
    print("   2. ➕ PLUS/CROSS MARK DETECTION")
    print("      - Medical red cross symbol detection")
    print("      - Cross shape template matching")
    print("      - Red color pattern analysis")
    print("      - Confidence boost: up to +0.25")
    
    print("   3. 📝 AMBULANCE TEXT DETECTION")
    print("      - Text-like pattern recognition")
    print("      - High contrast region analysis")
    print("      - Rectangular text region detection")
    print("      - Confidence boost: up to +0.15")
    
    print("   4. 🎨 EMERGENCY COLOR PATTERNS")
    print("      - Red + White classic ambulance colors")
    print("      - Red + Blue emergency light combinations")
    print("      - High visibility white + bright colors")
    print("      - Confidence boost: up to +0.15")
    
    print("   5. 💡 LIGHT BAR PATTERNS")
    print("      - Horizontal light arrangement detection")
    print("      - Bright area pattern analysis")
    print("      - Emergency vehicle light bar recognition")
    print("      - Confidence boost: up to +0.10")
    
    print("\n⚡ TOTAL POSSIBLE BOOST: up to +0.40 confidence")
    
    # Initialize enhanced detector
    print("\n🔧 Initializing advanced feature detection system...")
    detector = ONNXTrafficDetector()
    
    # Test on ambulance video
    video_path = "test_ambulance_Ambulance Running.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open test video: {video_path}")
        return False
    
    print(f"✅ Test video loaded: {video_path}")
    print("\n🎬 Starting advanced feature detection test...")
    print("\nWatch for:")
    print("   🚨 Red ambulance boxes with feature indicators")
    print("   🚦 Flashing light detection (brightness analysis)")
    print("   ➕ Plus/cross mark detection (red cross symbols)")
    print("   📝 Text pattern detection (ambulance labels)")
    print("   🎨 Color pattern analysis (emergency colors)")
    print("   💡 Light bar detection (horizontal light patterns)")
    print("   📊 Detailed feature logging in console")
    
    # Test statistics
    stats = {
        'frames_processed': 0,
        'ambulance_detections': 0,
        'feature_breakdown': {
            'flashing_lights': 0,
            'plus_cross_mark': 0,
            'ambulance_text': 0,
            'emergency_colors': 0,
            'light_patterns': 0
        },
        'avg_boosts': {
            'flashing_lights': [],
            'plus_cross_mark': [],
            'ambulance_text': [],
            'emergency_colors': [],
            'light_patterns': []
        },
        'total_boost_history': [],
        'detection_frames': []
    }
    
    frame_count = 0
    max_frames = 120  # Test more frames for flashing detection
    
    print("\n🔍 Processing frames for advanced feature analysis...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        stats['frames_processed'] += 1
        
        # Process with advanced detection
        processed_frame = detector.process_frame(frame)
        
        # Analyze detection results
        if detector.ambulance_detected:
            stats['ambulance_detections'] += 1
            stats['detection_frames'].append(frame_count)
            
            # Analyze tracked ambulances
            for obj_id, obj in detector.tracker.objects.items():
                if obj.get('class') == 'ambulance':
                    features = obj.get('features', {})
                    
                    # Count feature detections
                    for feature_name, value in features.items():
                        if feature_name != 'total_boost' and value > 0:
                            stats['feature_breakdown'][feature_name] += 1
                            stats['avg_boosts'][feature_name].append(value)
                    
                    # Track total boost
                    total_boost = features.get('total_boost', 0)
                    if total_boost > 0:
                        stats['total_boost_history'].append(total_boost)
        
        # Display enhanced frame
        cv2.imshow("🚨 Advanced Ambulance Feature Detection", processed_frame)
        
        # Progress update
        if frame_count % 20 == 0:
            print(f"  Processed {frame_count}/{max_frames} frames...")
        
        # Quick exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Comprehensive analysis
    print(f"\n{'='*80}")
    print("🔍 ADVANCED FEATURE DETECTION ANALYSIS")
    print("="*80)
    
    detection_rate = stats['ambulance_detections'] / stats['frames_processed'] if stats['frames_processed'] > 0 else 0
    
    print(f"\n📊 OVERALL DETECTION PERFORMANCE:")
    print(f"   Frames Processed: {stats['frames_processed']}")
    print(f"   Ambulance Detections: {stats['ambulance_detections']}")
    print(f"   Detection Rate: {detection_rate:.1%}")
    print(f"   Detection Quality: {'🟢 EXCELLENT' if detection_rate > 0.4 else '🟡 GOOD' if detection_rate > 0.2 else '🔴 NEEDS IMPROVEMENT'}")
    
    print(f"\n🎯 FEATURE DETECTION BREAKDOWN:")
    total_feature_detections = sum(stats['feature_breakdown'].values())
    
    for feature_name, count in stats['feature_breakdown'].items():
        percentage = (count / total_feature_detections) * 100 if total_feature_detections > 0 else 0
        avg_boost = np.mean(stats['avg_boosts'][feature_name]) if stats['avg_boosts'][feature_name] else 0
        
        feature_icons = {
            'flashing_lights': '🚦',
            'plus_cross_mark': '➕',
            'ambulance_text': '📝',
            'emergency_colors': '🎨',
            'light_patterns': '💡'
        }
        
        icon = feature_icons.get(feature_name, '🔧')
        feature_display = feature_name.replace('_', ' ').title()
        
        print(f"   {icon} {feature_display}: {count} detections ({percentage:.1f}%) | Avg boost: +{avg_boost:.3f}")
    
    print(f"\n⚡ CONFIDENCE BOOST ANALYSIS:")
    if stats['total_boost_history']:
        avg_total_boost = np.mean(stats['total_boost_history'])
        max_total_boost = max(stats['total_boost_history'])
        min_total_boost = min(stats['total_boost_history'])
        
        print(f"   Average Total Boost: +{avg_total_boost:.3f}")
        print(f"   Maximum Boost Achieved: +{max_total_boost:.3f}")
        print(f"   Minimum Boost: +{min_total_boost:.3f}")
        print(f"   Boost Effectiveness: {'🟢 HIGH' if avg_total_boost > 0.2 else '🟡 MODERATE' if avg_total_boost > 0.1 else '🔴 LOW'}")
    else:
        print("   No confidence boosts detected")
    
    print(f"\n🚦 FLASHING LIGHTS ANALYSIS:")
    flashing_detections = stats['feature_breakdown']['flashing_lights']
    if flashing_detections > 0:
        avg_flashing_boost = np.mean(stats['avg_boosts']['flashing_lights'])
        print(f"   Flashing Detected: {flashing_detections} times")
        print(f"   Average Flashing Boost: +{avg_flashing_boost:.3f}")
        print(f"   Flashing Effectiveness: {'🟢 EXCELLENT' if avg_flashing_boost > 0.2 else '🟡 GOOD' if avg_flashing_boost > 0.1 else '🔴 WEAK'}")
    else:
        print("   No flashing lights detected (may need longer observation)")
    
    print(f"\n➕ MEDICAL SYMBOL ANALYSIS:")
    cross_detections = stats['feature_breakdown']['plus_cross_mark']
    if cross_detections > 0:
        avg_cross_boost = np.mean(stats['avg_boosts']['plus_cross_mark'])
        print(f"   Cross/Plus Marks: {cross_detections} times")
        print(f"   Average Symbol Boost: +{avg_cross_boost:.3f}")
        print(f"   Symbol Recognition: {'🟢 EXCELLENT' if avg_cross_boost > 0.15 else '🟡 GOOD' if avg_cross_boost > 0.08 else '🔴 WEAK'}")
    else:
        print("   No medical symbols detected")
    
    print(f"\n🎨 COLOR PATTERN ANALYSIS:")
    color_detections = stats['feature_breakdown']['emergency_colors']
    if color_detections > 0:
        avg_color_boost = np.mean(stats['avg_boosts']['emergency_colors'])
        print(f"   Emergency Colors: {color_detections} times")
        print(f"   Average Color Boost: +{avg_color_boost:.3f}")
        print(f"   Color Recognition: {'🟢 EXCELLENT' if avg_color_boost > 0.1 else '🟡 GOOD' if avg_color_boost > 0.05 else '🔴 WEAK'}")
    else:
        print("   No emergency color patterns detected")
    
    print(f"\n🏆 OVERALL FEATURE SYSTEM PERFORMANCE:")
    if total_feature_detections > 0 and detection_rate > 0.3:
        print("   🟢 EXCELLENT - Advanced features significantly improving detection!")
        recommendation = "System performing optimally with advanced visual cues"
    elif total_feature_detections > 0 and detection_rate > 0.1:
        print("   🟡 GOOD - Features providing measurable improvement")
        recommendation = "Consider fine-tuning feature thresholds for better performance"
    else:
        print("   🔴 NEEDS IMPROVEMENT - Features not providing significant boost")
        recommendation = "Review feature detection parameters and test on different videos"
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   - {recommendation}")
    if flashing_detections == 0:
        print("   - Test on videos with visible flashing emergency lights")
    if cross_detections == 0:
        print("   - Ensure test videos contain visible medical symbols")
    if avg_total_boost < 0.15:
        print("   - Consider adjusting feature detection sensitivity")
    
    print("="*80)
    
    return detection_rate > 0.2 and total_feature_detections > 0

def test_feature_visualization():
    """Quick test to show feature detection in action"""
    print("\n🎨 Running feature visualization test...")
    
    detector = ONNXTrafficDetector()
    cap = cv2.VideoCapture("test_ambulance_Ambulance Running.mp4")
    
    if not cap.isOpened():
        return
    
    print("Press SPACE to save feature analysis frame, Q to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue
        
        processed_frame = detector.process_frame(frame)
        cv2.imshow("🔍 Ambulance Feature Analysis", processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ambulance_features_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"📸 Saved feature analysis: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚀 Starting Advanced Ambulance Feature Detection Tests...")
    
    # Main feature test
    success = test_advanced_ambulance_features()
    
    # Interactive visualization test
    test_feature_visualization()
    
    print(f"\n🎯 Advanced Feature Test Result: {'✅ SUCCESS' if success else '⚠️ NEEDS TUNING'}")
    print("🎉 Advanced ambulance detection with visual cues is now active!")
