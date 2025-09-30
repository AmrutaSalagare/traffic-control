#!/usr/bin/env python3
"""
Environment setup script for Intelligent Traffic Management System
Automatically downloads required models and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True,
                                check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def download_yolo_models():
    """Download required YOLO models"""
    print("\n📥 Downloading YOLO models...")

    try:
        # Import ultralytics to download models
        from ultralytics import YOLO

        models = [
            ("yolov8n.pt", "YOLOv8 Nano (fastest)"),
            ("yolov8s.pt", "YOLOv8 Small (balanced)")
        ]

        for model_file, description in models:
            if not Path(model_file).exists():
                print(f"📥 Downloading {description}...")
                try:
                    model = YOLO(model_file)  # This downloads the model
                    print(f"✅ {description} downloaded")
                except Exception as e:
                    print(
                        f"⚠️ {model_file} will be downloaded on first use: {e}")
            else:
                print(f"✅ {description} already exists")

        return True

    except ImportError:
        print("⚠️ Ultralytics not installed yet, models will be downloaded after dependency installation")
        return True


def verify_ambulance_model():
    """Verify ambulance model availability and provide guidance"""
    print("\n🚑 Checking ambulance detection model...")

    # Updated custom ambulance model path (new YOLOv11n)
    new_model_path = Path("models/new_indian_ambulance_yolov11n.pt")
    # legacy location (root)
    legacy_root_path = Path("indian_ambulance_yolov8.pt")
    legacy_models_path = Path(
        "models/indian_ambulance_yolov8.pt")  # legacy in models/

    if new_model_path.exists():
        print("✅ New Indian ambulance model found: new_indian_ambulance_yolov11n.pt")
        try:
            import os
            size = os.path.getsize(new_model_path)
            if size > 1000:  # Basic size check
                print(f"   Model size: {size // 1024} KB")
            else:
                print("⚠️ Model file seems too small, may be corrupted")
        except Exception as e:
            print(f"⚠️ Could not validate model: {e}")
    else:
        # Backward compatibility: check legacy names
        if legacy_models_path.exists() or legacy_root_path.exists():
            found_legacy = legacy_models_path if legacy_models_path.exists() else legacy_root_path
            print(f"ℹ️ Found legacy ambulance model: {found_legacy.name}")
            print(
                "   For consistency rename or copy it to: models/new_indian_ambulance_yolov11n.pt")
        else:
            print("⚠️ No ambulance-specific model found")
            print("   Emergency detection will fall back to generic YOLO model")
            print(
                "   Provide your trained weight as: models/new_indian_ambulance_yolov11n.pt")

    return True


def setup_environment():
    """Set up the development environment"""
    print("🚀 Setting up Intelligent Traffic Management System")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        return False

    # Install requirements
    requirements_file = "requirements_optimized.txt" if Path(
        "requirements_optimized.txt").exists() else "requirements.txt"
    if not run_command(f"pip install -r {requirements_file}", f"Installing dependencies from {requirements_file}"):
        return False

    # Download YOLO models
    download_yolo_models()

    # Verify ambulance model availability
    verify_ambulance_model()

    # Create necessary directories
    directories = ["logs", "data", "models", "output", "screenshots"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        return False

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False

    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO installed")
    except ImportError:
        print("❌ Ultralytics not installed")
        return False

    print("\n🎉 Setup completed successfully!")
    print("\n🚀 Ready to use!")
    print("=" * 40)
    print("Run traffic detection:")
    print("  python final_tracking_detection.py")
    print("\nWith video file:")
    print("  python final_tracking_detection.py --source path/to/video.mp4")
    print("\nWith custom settings:")
    print("  python final_tracking_detection.py --source video.mp4 --conf 0.15")

    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
