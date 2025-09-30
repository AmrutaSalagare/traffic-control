# Documentation Overview

This folder previously contained many granular progress and tuning journals. The system has since been streamlined. This consolidated index lists what remains relevant and what was deprecated.

## ✅ Active Concepts

- Core usage: See project root `README.md` (authoritative)
- Ambulance detection config: `config/emergency_detection.json`
- Models in use: `indian_ambulance_yolov11m_best.pt` (primary), fallback `indian_ambulance_yolov8.pt`

## 🗑 Deprecated / Removed Docs (superseded)

The following older deep-dive or transitional documents were removed to reduce noise:

- ACCURACY_UPGRADE_SUMMARY.md (folded into current README sections)
- PRODUCTION_SUMMARY.md (outdated structure & features)
- STREAMLINED_FEATURES.md (merged into simplified feature list)
- SPEED_MODE_FIX.md (speed modes refactored; `--super-fast` removed)
- DEPLOYMENT_CHECKLIST.md (general deployment now straightforward)
- ROAD_TRACKING_RESULTS.md (empty)

## 📄 Kept As Context

- `context.md` – High-level problem statement & future vision (kept for stakeholders)
- `SETUP_GUIDE.md` – Retained temporarily; could be merged later (contains more verbose environment steps and troubleshooting)

## 🔍 If You Need Legacy Details

Retrieve them from git history (use `git log` / `git show <commit>:docs/<file>`).

## ✨ Next Possible Improvements

- Merge `SETUP_GUIDE.md` sections (troubleshooting) into root README appendix
- Add lightweight JSON event output option for downstream integration
- Provide benchmark script for automated FPS + detection quality reporting

---

Documentation consolidated on: 2025-09-20
