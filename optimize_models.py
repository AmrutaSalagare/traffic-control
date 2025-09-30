import os
import torch
import torch.quantization
from ultralytics import YOLO
import numpy as np
from torch.quantization import quantize_dynamic
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType
from pathlib import Path
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='torch.quantization')

# Configuration
MODEL_DIR = "models"
OUTPUT_DIR = "optimized_models"
oshow = True  # Set to False to see model outputs during optimization

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_size_of_model(model, model_name):
    """Print model size in MB"""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6  # in MB
    print(f"Size of {model_name}: {size:.2f} MB")
    os.remove('temp.p')
    return size

def convert_to_fp16(model_path: str, model_name: str):
    """Convert a YOLO model to FP16 ONNX format"""
    print(f"\n{'='*50}")
    print(f"Converting {model_name} to FP16 ONNX...")
    
    try:
        # Load the model
        print("Loading model...")
        model = YOLO(model_path)
        
        # Define output path
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(model_path))[0]}_fp16.onnx")
        
        # Export with FP16
        model.export(
            format="onnx",
            imgsz=640,
            half=True,  # Enable FP16
            device='cpu',
            simplify=True,
            dynamic=True,
            opset=12,
            verbose=False
        )
        
        # The export saves the file in the current directory, so we need to move it
        exported_model = os.path.basename(model_path).replace('.pt', '.onnx')
        if os.path.exists(exported_model):
            os.rename(exported_model, output_path)
            print(f"FP16 ONNX model saved to: {output_path}")
            
            # Optimize the ONNX model
            optimized_path = optimize_onnx(output_path)
            return model, optimized_path
        else:
            print("Error: Failed to export FP16 model")
            return None, None
            
    except Exception as e:
        print(f"Error during FP16 conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def quantize_to_int8(model_path: str, model_name: str):
    """Quantize a YOLO model to INT8 ONNX format"""
    print(f"\n{'='*50}")
    print(f"Quantizing {model_name} to INT8...")
    
    try:
        # First convert to ONNX if not already
        onnx_path = convert_to_onnx(model_path, "", model_name)
        if not onnx_path:
            print("Error: Failed to convert to ONNX")
            return None, None
            
        # Define output path
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(model_path))[0]}_int8.onnx")
        
        # Use ONNX Runtime to quantize the model
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print(f"Quantizing {onnx_path} to INT8...")
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QUInt8,
            optimize_model=True,
            use_external_data_format=False,
            extra_options={
                'DisableShapeInference': True,
                'EnableSubgraph': True
            }
        )
        
        print(f"INT8 quantized model saved to: {output_path}")
        return None, output_path
        
    except Exception as e:
        print(f"Error during INT8 quantization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def convert_to_onnx(model_path, output_path, model_name, input_shape=(1, 3, 640, 640)):
    """Convert a YOLO model to ONNX format"""
    print(f"\n{'='*50}")
    print(f"Converting {model_name} to ONNX...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Define ONNX output path
        onnx_filename = f"{os.path.splitext(os.path.basename(model_path))[0]}.onnx"
        onnx_path = os.path.join(OUTPUT_DIR, onnx_filename)
        
        # Check if ONNX file already exists
        if os.path.exists(onnx_path):
            print(f"ONNX file already exists at {onnx_path}, skipping export.")
            return onnx_path
            
        print(f"Exporting to ONNX format: {onnx_path}")
        
        try:
            # Load the model with verbose output
            print(f"Loading model from {model_path}...")
            model = YOLO(model_path)
            
            # First try with default settings
            print("Attempting export with default settings...")
            success = model.export(
                format="onnx",
                imgsz=640,  # Standard YOLO input size
                dynamic=True,
                simplify=True,
                opset=12,
                half=False,
                int8=False,
                device='cpu',
                verbose=True
            )
            
            # Check for the exported file in various possible locations
            possible_paths = [
                os.path.basename(model_path).replace('.pt', '.onnx'),  # Current directory
                onnx_path,  # Target path
                os.path.join('runs', 'detect', 'train', 'weights', 'best.onnx'),  # Common Ultralytics export location
                os.path.join('runs', 'detect', 'predict', 'weights', 'best.onnx'),
                os.path.join('runs', 'detect', 'export', os.path.basename(model_path).replace('.pt', '.onnx'))
            ]
            
            exported = False
            for path in possible_paths:
                if os.path.exists(path):
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                    
                    # Copy the file to our target location
                    import shutil
                    shutil.copy2(path, onnx_path)
                    print(f"Found and copied ONNX model to: {onnx_path}")
                    exported = True
                    break
            
            if not exported:
                print("Error: Could not find exported ONNX file in any expected location.")
                return None
                
            # Verify the ONNX model
            try:
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print("ONNX model is valid!")
                
                # Print ONNX model size
                onnx_size = os.path.getsize(onnx_path) / 1e6  # in MB
                print(f"ONNX model size: {onnx_size:.2f} MB")
                
                return onnx_path
                
            except Exception as e:
                print(f"Warning: ONNX model validation failed: {str(e)}")
                print("The model was still exported, but there might be compatibility issues.")
                return onnx_path
                
        except Exception as e:
            print(f"Error during model export: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Error during ONNX conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def optimize_onnx(onnx_path):
    """Optimize ONNX model with ONNX Runtime"""
    print(f"\n{'='*50}")
    print(f"Optimizing ONNX model: {onnx_path}")
    
    try:
        # Set optimization level
        optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Define output path for optimized model
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        # Create session options for optimization
        options = ort.SessionOptions()
        options.graph_optimization_level = optimization_level
        
        # Enable all optimizations
        options.optimized_model_filepath = optimized_path
        
        # Configure session for performance
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Create session to trigger optimization
        print("Creating optimized ONNX Runtime session...")
        providers = ['CPUExecutionProvider']
        _ = ort.InferenceSession(onnx_path, options, providers=providers)
        
        if os.path.exists(optimized_path):
            print(f"Optimized ONNX model saved to: {optimized_path}")
            
            # Print optimized model size
            optimized_size = os.path.getsize(optimized_path) / 1e6  # in MB
            original_size = os.path.getsize(onnx_path) / 1e6
            print(f"Optimized model size: {optimized_size:.2f} MB")
            print(f"Size reduction: {original_size/optimized_size:.2f}x")
            
            return optimized_path
        else:
            print("Warning: Optimized model file was not created. Using original ONNX model.")
            return onnx_path
            
    except Exception as e:
        print(f"Warning: ONNX optimization failed: {str(e)}")
        print("Falling back to using the original ONNX model.")
        return onnx_path

def test_model_performance(model_path, model_type="pytorch"):
    """Test the performance of a model"""
    print(f"\n{'='*50}")
    print(f"Testing performance of {model_path}")
    
    if model_type == "pytorch":
        # Load the model
        model = YOLO(model_path)
        
        # Create a dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Warm up
        for _ in range(3):
            _ = model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            _ = model(dummy_input)
        
        avg_time = (time.time() - start_time) / num_runs
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        
    elif model_type == "onnx":
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Create a dummy input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Warm up
        for _ in range(3):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            _ = session.run(None, {input_name: dummy_input})
        
        avg_time = (time.time() - start_time) / num_runs
        print(f"Average inference time: {avg_time*1000:.2f} ms")

def main():
    # Models to optimize with their respective precision targets
    # We'll use INT8 for vehicle detection (faster) and FP16 for ambulance detection (more accurate)
    model_configs = [
        {
            "path": os.path.join(MODEL_DIR, "yolo11n.pt"),
            "name": "YOLOv11n Vehicle Detection",
            "precision": "int8",  # Use INT8 for vehicle detection (speed focus)
            "output_name": "yolo11n_optimized_int8.onnx"
        },
        {
            "path": os.path.join(MODEL_DIR, "indian_ambulance_yolov11n_best.pt"),
            "name": "YOLOv11n Ambulance Detection",
            "precision": "fp16",  # Use FP16 for ambulance detection (accuracy focus)
            "output_name": "indian_ambulance_yolov11n_best_optimized_fp16.onnx"
        }
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each model with its specific precision target
    for config in model_configs:
        model_path = config["path"]
        model_name = config["name"]
        precision = config["precision"]
        output_name = config["output_name"]
        
        if not os.path.exists(model_path):
            print(f"\n{'='*50}")
            print(f"Error: Model not found at {model_path}")
            print("Available models in the models directory:")
            for f in os.listdir(MODEL_DIR):
                if f.endswith('.pt'):
                    print(f"- {f}")
            print("Skipping to next model...")
            print("="*50 + "\n")
            continue
            
        try:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {model_name} with {precision.upper()} precision")
            print(f"Model path: {model_path}")
            print("="*80)
            
            # 1. First convert to base ONNX format
            print("\nStep 1: Converting to base ONNX format...")
            base_onnx_path = convert_to_onnx(
                model_path,
                OUTPUT_DIR,
                model_name
            )
            
            if not base_onnx_path or not os.path.exists(base_onnx_path):
                print("Failed to convert model to base ONNX. Skipping to next model...")
                continue
            
            # 2. Apply precision-specific optimization
            optimized_path = None
            if precision == "int8":
                print("\nStep 2: Applying INT8 quantization...")
                _, optimized_path = quantize_to_int8(model_path, model_name)
            elif precision == "fp16":
                print("\nStep 2: Converting to FP16...")
                _, optimized_path = convert_to_fp16(model_path, model_name)
            
            # 3. If we have an optimized model, test its performance
            if optimized_path and os.path.exists(optimized_path):
                print("\n" + "="*50)
                print("Performance Testing")
                print("="*50)
                
                try:
                    # Test the optimized model
                    print(f"\nTesting {precision.upper()} optimized model performance:")
                    test_model_performance(optimized_path, "onnx")
                    
                    # Also test the base ONNX model for comparison
                    print("\nTesting base ONNX model performance (for comparison):")
                    test_model_performance(base_onnx_path, "onnx")
                    
                except Exception as e:
                    print(f"Error during performance testing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Rename the optimized model to the desired output name
                final_output_path = os.path.join(OUTPUT_DIR, output_name)
                if os.path.exists(optimized_path) and optimized_path != final_output_path:
                    if os.path.exists(final_output_path):
                        os.remove(final_output_path)
                    os.rename(optimized_path, final_output_path)
                    print(f"\nFinal optimized model saved to: {final_output_path}")
            
            print("\n" + "="*80)
            print(f"COMPLETED OPTIMIZATION FOR: {model_name} ({precision.upper()})")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\nError processing {model_name}: {str(e)}")
            print("Skipping to next model...\n")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
