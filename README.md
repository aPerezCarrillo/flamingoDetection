# flamingoDetection
Detect and Track flamingoes using YOLOv5 and DeepSORT

## Overview
This project implements flamingo detection and tracking in video streams using YOLOv5 for object detection and DeepSORT for object tracking.

## Features
- Detects flamingos in video frames using YOLOv5
- Tracks detected flamingos across frames using DeepSORT
- Multiple fallback options if YOLOv5 is not available:
  - Torch Hub YOLOv5
  - OpenCV DNN with MobileNet SSD
  - Simulated detections

## Requirements
Install the required dependencies:
```bash
pip install -r pre_requirements.txt
```

## Usage
1. Place your video file in the `data` directory
2. Update the `video_path` variable in `scripts/object_detection.py` to point to your video file
3. Run the script:
```bash
cd scripts
python object_detection.py
```

## Implementation Details
- YOLOv5 is used for object detection, focusing on the 'bird' class as a proxy for flamingos
- DeepSORT is used for tracking the detected flamingos across frames
- The output video will be saved in the `data` directory

## Customization
- To use a custom YOLOv5 model trained specifically for flamingos, update the `weights_path` variable
- Adjust detection thresholds in the code to fine-tune detection sensitivity
