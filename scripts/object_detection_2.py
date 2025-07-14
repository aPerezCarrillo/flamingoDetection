from collections import defaultdict

import cv2
import numpy as np
import time

# ... (imports and video setup) ...
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt') # 'n' for nano, a small and fast model. Use 'yolov8s.pt' for slightly better accuracy.

# Identify the class ID for 'bird' in COCO dataset (usually 14 for YOLOv8/COCO)
# You can check model.names to confirm class IDs
# e.g., print(model.names)
BIRD_CLASS_ID = 14 # Confirm this from model.names

# --- Placeholder for Object Detection Model ---
# In a real scenario, you'd load your pre-trained Mask R-CNN, YOLO, etc. here.
# For simplicity, we'll simulate detections or use a very generic pre-trained model if available.

# Example: If using a simple pre-trained OpenCV DNN model (e.g., MobileNet SSD)
# This model might detect 'bird' or 'animal' but not specifically 'flamingo'.
# Replace these paths with your actual model files if you have them.
# net = cv2.dnn.readNetFromCaffe("path/to/MobileNetSSD_deploy.prototxt", "path/to/MobileNetSSD_deploy.caffemodel")
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# DESIRED_CLASSES = ["bird"] # Or "flamingo" if your model supports it

# --- Placeholder for DeepSORT Tracker ---
# We'll use the 'deep_sort_realtime' library.
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Video Stream Configuration ---
video_path = '../data/Flamingo_30sec_excerpt.avi'  # <<< IMPORTANT: Change this to your video file path
output_video_path = '../data/flamingo_tracking_output.mp4'  # Using .mp4 extension for QuickTime compatibility

# Initialize video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
# XVID is a common codec, you might need 'mp4v' or 'MJPG' depending on your system
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize DeepSORT tracker
# The `max_iou_distance` and `n_init` parameters can be tuned.
# `max_iou_distance`: maximum cosine distance or IOU distance between track and detection.
# `n_init`: number of consecutive detections before a track is confirmed.
tracker = DeepSort(max_iou_distance=0.7, n_init=3,
                   nms_max_overlap=1.0)  # nms_max_overlap=1.0 to disable NMS within DeepSORT

# --- Trajectory Storage ---
# Stores {track_id: [(x1, y1), (x2, y2), ...]}
# Using defaultdict will automatically create an empty list for new track_ids
trajectories = defaultdict(list)
max_trajectory_length = 100 # How many past points to store for drawing the tail

frame_count = 0
start_time = time.time()


frame_count = 0
start_time = time.time()

print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_video_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break

    frame_count += 1

    # Perform detection with YOLOv8
    results = model(frame, verbose=False)[0] # verbose=False to suppress output for each frame

    detections_for_tracker = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        if class_id == BIRD_CLASS_ID and confidence > 0.5: # Adjust confidence threshold
            w = x2 - x1
            h = y2 - y1
            detections_for_tracker.append(([x1, y1, w, h], confidence, model.names[class_id]))


    # --- Update DeepSORT Tracker ---
    # `update_tracks` takes a list of detections (as specified above)
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # --- Draw Detections and Tracks on Frame ---
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box in (left, top, right, bottom) format

        # Convert to integer coordinates
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Calculate center point for trajectory
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Store current center point in trajectory
        trajectories[track_id].append((center_x, center_y))

        # Limit trajectory length to keep it from getting too long
        if len(trajectories[track_id]) > max_trajectory_length:
            trajectories[track_id].pop(0)  # Remove the oldest point



        # Draw bounding box
        color = (0, 255, 0)  # Green for tracked objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Put track ID text
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- Draw Trajectory ---
        # Draw lines connecting the past points
        for i in range(1, len(trajectories[track_id])):
            pt1 = trajectories[track_id][i - 1]
            pt2 = trajectories[track_id][i]
            # Use a slightly different color or transparency for trajectory
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue line for trajectory

    # Display frame
    cv2.imshow('Flamingo Tracking', frame)

    # Write the frame to the output video
    out.write(frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
print(f"Average FPS: {frame_count / (end_time - start_time):.2f}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing finished.")