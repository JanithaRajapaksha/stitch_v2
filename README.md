# Stitch V2 Fabric Inspection System

This project is a real-time fabric inspection system that uses computer vision to detect defects in stitches and fabric. It measures stitch length, distance to edge, and identifies consecutive defects.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd stitch_v2
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Configuration

1.  **`config.yaml`:** This file contains the main configuration for the application, including camera settings, serial port, database credentials, and defect thresholds.

2.  **Camera Calibration:**
    *   `camera_calibration.json`: Contains the camera intrinsic matrix and distortion coefficients.
    *   `camera_extrinsics.json`: Contains the camera extrinsics (rotation and translation vectors) relative to the scene.

    These files are essential for accurate measurements.

## Usage

To run the fabric inspection system, execute the main script:

```bash
python linear_machine_v2.py
```

The system will start, initialize the camera and serial communication, and begin processing the fabric according to the settings in `config.yaml`.
