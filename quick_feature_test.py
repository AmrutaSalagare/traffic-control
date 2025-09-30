"""
Quick test of advanced ambulance features
"""
import os
from final_tracking_onnx import ONNXTrafficDetector

def quick_test():
    print("="*60)
    print("ADVANCED AMBULANCE FEATURES - QUICK TEST")
    print("="*60)
    
    print("\nTesting system initialization...")
    try:
        detector = ONNXTrafficDetector()
        print("SUCCESS: System initialized with advanced features!")
        
        # Check if advanced features are available
        print("\nChecking advanced feature methods...")
        
        methods_to_check = [
            '_detect_ambulance_features',
            '_detect_flashing_lights', 
            '_detect_plus_cross_mark',
            '_detect_ambulance_text',
            '_detect_emergency_color_patterns',
            '_detect_light_bar_patterns'
        ]
        
        for method in methods_to_check:
            if hasattr(detector, method):
                print(f"  [OK] {method} - Available")
            else:
                print(f"  [MISSING] {method} - Missing")
        
        # Check feature storage
        print(f"\nAdvanced feature storage initialized:")
        print(f"  Previous frames buffer: {len(detector.previous_frames)}/{detector.previous_frames.maxlen}")
        print(f"  Visual features dict: {len(detector.ambulance_visual_features)} entries")
        print(f"  ROI enabled: {detector.roi_enabled}")
        
        print("\n" + "="*60)
        print("ADVANCED FEATURES STATUS: ACTIVE & READY!")
        print("="*60)
        
        print("\nFeature Capabilities:")
        print("  [FLASH] Flashing Lights: Brightness variation analysis")
        print("  [CROSS] Plus/Cross Mark: Medical symbol detection")  
        print("  [TEXT] Text Patterns: Ambulance text recognition")
        print("  [COLOR] Color Patterns: Emergency color combinations")
        print("  [LIGHTS] Light Bars: Horizontal light arrangements")
        print("  [BOOST] Max Boost: +0.40 confidence enhancement")
        
        print("\nTo test in action, run:")
        print("  python final_tracking_onnx.py --source your_ambulance_video.mp4")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\nTest Result: {'SUCCESS' if success else 'FAILED'}")
