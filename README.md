# Production-ready vehicle + ambulance detection & tracking pipeline optimized for | Balanced (default) | yolo11s (GPU) / yolo11n (CPU) | 640              | Good trade-off             |
| `--fast`           | yolo11n                       | 480              | Lower latency, single scale |
| `--accuracy`       | yolo11s                       | 832 (GPU) / 640 (CPU) | Larger size, more recall    |ian traffic. The system now uses a **single simplified CLI**, **smart defaults**, and a **custom YOLOv11m ambulance model** with optional fallback. Legacy experimental flags and scripts have been removed for clarity.ntelligent Traffic Management System

Production‑ready vehicle + ambulance detection & tracking pipeline optimized for Indian traffic. The system now uses a **single simplified CLI**, **smart defaults**, and a **custom YOLOv11m ambulance model** with optional fallback. Legacy experimental flags and scripts have been removed for clarity.

## 🚀 Core Features

| Capability                                  | Status                            |
| ------------------------------------------- | --------------------------------- |
| Vehicle detection (YOLOv11 n/s)            | ✅                          |
| ByteTrack tracking (auto if deps available) | ✅                          |
| Ambulance detection (raw YOLOv11m custom)   | ✅                          |
| Fallback ambulance YOLOv8 model            | ✅                          |
| Adaptive ROI (road focus)                  | ✅ (always on unless disabled)    |
| Line crossing logic                        | ✅ (internal)                   |
| Unified vehicle labeling option            | ✅ (`--unified-vehicles`)         |
| Ensemble / visual feature ambulance mode    | Optional (`--ambulance-ensemble`) |
| Verbose ambulance debug                    | Optional (`--ambulance-debug`)    |

Removed / deprecated (old docs may mention): `--super-fast`, `--use-bytetrack`, homography calibration UI, manual TTA flags, dense traffic flags, many ambulance tuning knobs.

## 🎯 Quick Start

### 1) Clone

```bash
git clone https://github.com/AmrutaSalagare/traffic-control.git
cd traffic-control
```

### 2) Install dependencies

```bash
python -m venv venv
venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

Models folder (`models/`) should contain:

- `yolo11n.pt` (fast)
- `yolo11s.pt` (balanced/accuracy baseline)
- `indian_ambulance_yolov11m_best.pt` (PRIMARY ambulance model)
- `indian_ambulance_yolov8.pt` (fallback - optional)

### 3) Run

```bash
# Webcam (balanced)
python final_tracking_detection.py

# Video file
python final_tracking_detection.py --source videos/rushing.mp4

# Fast / Accuracy profiles
python final_tracking_detection.py --fast
python final_tracking_detection.py --accuracy

# Use a specific vehicle model
python final_tracking_detection.py --model yolo11s.pt

# Disable ambulance detection
python final_tracking_detection.py --no-ambulance

# Ambulance ensemble mode (slower, uses visual fusion)
python final_tracking_detection.py --ambulance-ensemble

# Adjust ambulance sensitivity
python final_tracking_detection.py --ambulance-conf 0.08
```

### 4) Controls

- **Q**: Quit
- **S**: Save screenshot
- **R**: Toggle ROI display
- **T**: Toggle tracking display
- **C**: Clear tracking history
- **Space**: Pause/Resume

## 💡 Runtime Profiles

| Mode               | Default Vehicle Model         | Inference Size        | Notes                       |
| ------------------ | ----------------------------- | --------------------- | --------------------------- |
| Balanced (default) | yolo11s (GPU) / yolo11n (CPU) | 640                   | Good trade‑off              |
| `--fast`           | yolo11n                       | 480                   | Lower latency, single scale |
| `--accuracy`       | yolo11s                       | 832 (GPU) / 640 (CPU) | Larger size, more recall    |

## 📊 Typical Performance (illustrative)

- CPU: ~3-5 FPS standard, up to ~10-12 FPS with `--super-fast` (content/hardware dependent)
- GPU: higher throughput with the same flags
- Accurate counting via ByteTrack + line-crossing

## 🛠 Key CLI Flags

General:

- `--source` (index/path/RTSP) | `--model yolo11n.pt|yolo11s.pt|yolo11m.pt`
- `--conf <float>` vehicle confidence (default 0.25)
- `--imgsz <int>` YOLO inference size (default 640)
- `--fast` / `--accuracy`
- `--no-roi` disable adaptive ROI
- `--unified-vehicles` collapse classes visually (car/bus/truck/motorcycle... → Vehicle)

Ambulance:

- `--ambulance-conf <float>` (default 0.15, adaptive internally)
- `--ambulance-interval <int>` (default 3)
- `--ambulance-ensemble` slower fused mode
- `--ambulance-model <path>` override primary
- `--ambulance-fallback-model <path>` override fallback
- `--ambulance-debug` verbose diagnostics
- `--no-ambulance` disable entirely

### Vehicle Classes Detected

- Cars (Green boxes)
- Motorcycles (Orange boxes)
- Bicycles (Cyan boxes)
- Buses (Magenta boxes)
- Trucks (Blue boxes)

## 🏗 Simplified Flow

```
```
Frame → Vehicle YOLO → (ByteTrack) → ROI Filter → Stats/Overlay → Display
            └─ every N frames → Ambulance YOLO (primary + optional fallback)
```
```

## 📁 Project Structure (post‑cleanup)

```
```
├── final_tracking_detection.py   # Main entry
├── src/services/
│   ├── bytetrack_counter.py
│   └── enhanced_emergency_detector.py
├── models/                     # YOLO weights
├── config/                     # emergency_detection.json + env yamls
├── docs/                       # Consolidated docs
├── tests/                      # Remaining tests
├── videos/                     # Sample inputs
└── requirements.txt
```
```

## 🎛 Examples

Higher recall (lower vehicle confidence) in accuracy mode:

```
python final_tracking_detection.py --conf 0.18 --accuracy --source videos/rushing.mp4
```

Fast mode + tighter ambulance interval (every frame):

```
python final_tracking_detection.py --fast --ambulance-interval 1 --source videos/rushing.mp4
```

Override ambulance model:

```
python final_tracking_detection.py --ambulance-model models/custom_amb.pt
```

### Performance Optimization

- CPU: prefer `--super-fast` (YOLO11n)
- GPU: larger `--imgsz` + `--accuracy` for recall
- Memory: tune processing size and ROI

## � Ambulance Detection Logic

Default: raw custom YOLOv11m weights at adaptive confidence (starts ~0.07, decays slightly if starved, gently rises after hits, capped to avoid missing).

Fallback: YOLOv8 custom model may produce higher confidences; used when primary misses and fallback is enabled.

Ensemble: `--ambulance-ensemble` invokes the enhanced detector (slower; includes visual feature heuristics) for evaluation scenarios.

## 📈 Runtime Stats Overlay

Displayed:

- FPS / frame latency
- Active / confirmed vehicles
- Recent ambulance event flag
- (Optional) snapshot saving to `logs/` when enabled

## 🚦 Integration Potential

Foundation for adaptive signals, emergency corridor automation, and density analytics. JSON/event streaming layer can be added externally (not bundled to keep core lean).

## 📦 Deployment Notes

- Keep model weights outside Git history (use release assets or manual placement)
- GPU recommended for accuracy mode at 832 resolution
- For edge devices drop to `--fast` and `yolo11n.pt`

## 📋 Requirements

- Python 3.10+
- PyTorch + Ultralytics
- OpenCV, NumPy
- (Optional) filterpy (for full ByteTrack implementation)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - Production ready for commercial use

---

Built for Indian traffic conditions 🇮🇳 - production simplified & maintained.
