# 🚦 Intelligent Traffic Management System - Setup Guide

This guide will help you set up the development environment and test the vehicle detection system.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**Windows:**

```bash
setup.bat
```

**Linux/Mac:**

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

1. **Create Virtual Environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment:**

   **Windows:**

   ```bash
   venv\Scripts\activate
   ```

   **Linux/Mac:**

   ```bash
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🧪 Testing Vehicle Detection

### Basic Test (Synthetic Frame)

```bash
python test_vehicle_detection.py --source synthetic
```

### Webcam Test

```bash
python test_vehicle_detection.py --source webcam
```

### Video File Test

```bash
python test_vehicle_detection.py --source video --video path/to/your/video.mp4
```

### Complete Test Suite

```bash
python test_vehicle_detection.py --source all
```

## 🎛️ Advanced Configuration

### Custom Model

```bash
python test_vehicle_detection.py --model yolov8s.pt --confidence 0.6
```

### GPU Acceleration (if available)

```bash
python test_vehicle_detection.py --device cuda
```

### Test Specific Components

```bash
# Test camera manager only
python -m pytest tests/test_camera_manager.py -v

# Test integration
python -m pytest tests/test_camera_integration.py -v

# Test vehicle detection
python test_basic_vehicle_detector.py
```

## 📋 System Requirements

### Minimum Requirements

- Python 3.8+
- 4GB RAM
- CPU with AVX support
- Webcam (optional, for live testing)

### Recommended Requirements

- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with CUDA support
- Multiple cameras/video sources

### Dependencies

- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **Deep Learning:** PyTorch, ONNX Runtime
- **Data Processing:** NumPy, Pandas
- **Testing:** Pytest
- **Hardware Interface:** PyModbus, PySerial (for production)

## 🎯 What Gets Tested

### 1. Vehicle Detection System

- ✅ YOLOv8 model loading and initialization
- ✅ Real-time vehicle detection and classification
- ✅ Performance metrics (FPS, inference time)
- ✅ Multiple vehicle types (cars, motorcycles, trucks, buses)

### 2. Camera Stream Management

- ✅ RTSP stream handling
- ✅ Multi-camera support
- ✅ Health monitoring and auto-reconnection
- ✅ Frame preprocessing pipeline

### 3. Integration Testing

- ✅ Camera manager + vehicle detector integration
- ✅ Concurrent multi-camera processing
- ✅ Error handling and recovery
- ✅ Resource management

## 🔧 Troubleshooting

### Common Issues

**1. OpenCV Installation Issues**

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

**2. CUDA/GPU Issues**

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. YOLOv8 Model Download Issues**

```bash
# Pre-download models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**4. Webcam Access Issues**

- Check if webcam is being used by another application
- Try different camera indices (0, 1, 2...)
- On Linux, check permissions: `sudo usermod -a -G video $USER`

### Performance Optimization

**1. For CPU-only systems:**

- Use smaller models: `yolov8n.pt` (nano) instead of `yolov8s.pt` (small)
- Reduce input resolution
- Lower confidence threshold

**2. For GPU systems:**

- Install CUDA-enabled PyTorch
- Use larger models for better accuracy
- Enable TensorRT optimization (on supported hardware)

**3. Memory optimization:**

- Reduce batch size
- Use frame skipping for video processing
- Enable garbage collection

## 📊 Expected Results

### Synthetic Frame Test

- Should detect 3-5 vehicle-like objects
- Inference time: 50-200ms (CPU), 10-50ms (GPU)
- Confidence scores: 0.3-0.9

### Webcam Test

- Real-time processing at 10-30 FPS
- Accurate detection of vehicles in view
- Smooth bounding box tracking

### Video Test

- Consistent detection across frames
- Performance metrics logging
- Proper resource cleanup

## 🎮 Interactive Controls

During testing:

- **'q'**: Quit video/webcam test
- **Any key**: Continue from synthetic frame test
- **Ctrl+C**: Interrupt test gracefully

## 📈 Performance Monitoring

The system tracks:

- **FPS**: Frames processed per second
- **Inference Time**: Time per detection
- **Detection Count**: Vehicles detected per frame
- **Memory Usage**: RAM and GPU memory
- **Error Rates**: Failed detections and recoveries

## 🔄 Next Steps

After successful testing:

1. **Integrate with Traffic Control:**

   ```bash
   python -m src.main  # Run full system
   ```

2. **Deploy to Edge Device:**

   - Copy to Raspberry Pi/Jetson Nano
   - Install ARM-compatible dependencies
   - Configure camera connections

3. **Production Setup:**
   - Configure RTSP camera URLs
   - Set up traffic signal interfaces
   - Enable cloud synchronization

## 📞 Support

If you encounter issues:

1. Check the logs in `logs/` directory
2. Run tests with verbose output: `pytest -v -s`
3. Verify all dependencies: `pip list`
4. Check system resources: `python -c "import psutil; print(psutil.virtual_memory())"`

---

**Happy Testing! 🚗🚦**
