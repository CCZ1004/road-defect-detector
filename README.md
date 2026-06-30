# RoadSafe AI: Road Defect Detection System

A real-time road defect monitoring system that uses computer vision (YOLOv8) and geolocation to detect, map, and log road damage.

## 🚀 Features
- **Real-time Detection:** Uses YOLOv8 to identify road defects via webcam.
- **Geolocation Logging:** Captures live GPS coordinates for every detected defect.
- **Database Integration:** Automatically logs all detections into a local MongoDB instance.
- **Interactive Dashboard:** A web interface to view statistics and maps of detected defects.

## 🛠 Prerequisites
- **Python 3.10+**
- **MongoDB:** Ensure it is running locally on `localhost:27017`.
- **NVIDIA GPU (Recommended):** For optimal frame rates, an NVIDIA GPU with CUDA drivers is recommended.

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/road-defect-detector.git](https://github.com/yourusername/road-defect-detector.git)
   cd road-defect-detector