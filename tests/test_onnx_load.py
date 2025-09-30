import os
import sys
import onnxruntime as ort
import numpy as np

# Redirect stdout to a file
sys.stdout = open('onnx_test_output.txt', 'w')
print("Starting ONNX model test...")

def test_onnx_model(model_path):
    print(f"\n{'='*50}")
    print(f"Testing model: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        # Print available providers
        print("\nAvailable ONNX Runtime providers:", ort.get_available_providers())
        
        # Try to create a session with different providers
        providers = [
            'CUDAExecutionProvider', 
            'CPUExecutionProvider'
        ]
        
        # Filter to only available providers
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        print(f"Trying with providers: {providers}")
        
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 0  # 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
        
        # Try to load the model
        print("\nAttempting to load model...")
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Print model input details
        print("\nModel loaded successfully!")
        print("Input details:")
        for i, input in enumerate(session.get_inputs()):
            print(f"  Input {i}: {input.name}, shape: {input.shape}, type: {input.type}")
            
        # Print output details
        print("\nOutput details:")
        for i, output in enumerate(session.get_outputs()):
            print(f"  Output {i}: {output.name}, shape: {output.shape}, type: {output.type}")
            
        return True
        
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        # Test with both models
        project_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Project directory: {project_dir}")
        
        # List files in optimized_models
        models_dir = os.path.join(project_dir, "optimized_models")
        print(f"\nContents of {models_dir}:")
        for f in os.listdir(models_dir):
            if f.endswith('.onnx'):
                print(f"- {f}")
        
        # Test vehicle model
        print("\n" + "="*50)
        vehicle_model = os.path.join(models_dir, "yolo11n_optimized.onnx")
        print(f"\nTesting vehicle model: {vehicle_model}")
        if os.path.exists(vehicle_model):
            test_onnx_model(vehicle_model)
        else:
            print(f"Error: Vehicle model not found at {vehicle_model}")
        
        # Test ambulance model
        print("\n" + "="*50)
        ambulance_model = os.path.join(models_dir, "indian_ambulance_yolov11n_best_optimized.onnx")
        print(f"\nTesting ambulance model: {ambulance_model}")
        if os.path.exists(ambulance_model):
            test_onnx_model(ambulance_model)
        else:
            print(f"Error: Ambulance model not found at {ambulance_model}")
            
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print("Test completed. Check onnx_test_output.txt for details.")
