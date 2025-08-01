import cv2
import json
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
MODEL_PATH = '../runs/detect/train2/weights/best.pt'
INPUT_VIDEO_PATH = '../data/Flamingo_30sec_excerpt.avi'
OUTPUT_VIDEO_PATH = '../data/tracked_video.mp4'  # Will be saved in the same folder
JSON_OUTPUT_PATH = '../data/timeline.json'
# ---------------------

CLASS_COLORS = {
    0: (255, 105, 180),  # Flamingo (Pink)
    1: (135, 206, 250)  # Human (Sky Blue)
}
CLASS_COLORS = {
    0: (139, 0, 139),   # Flamingo (Dark Magenta)
    1: (0, 0, 205)      # Human (Medium Blue)
}
# ---------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {INPUT_VIDEO_PATH}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Data structures for new tracking logic ---
tracked_objects = defaultdict(lambda: {'first_seen': -1, 'last_seen': -1, 'class_id': -1, 'display_id': ''})
class_counters = defaultdict(int)
track_id_to_display_id = {}

# --- PASS 1: Discover tracks and assign class-specific IDs ---
print("Pass 1/2: Discovering all unique tracks...")
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_number += 1

    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for i, track_id in enumerate(track_ids):
            # If it's a new track ID, assign it a class-specific ID
            if track_id not in track_id_to_display_id:
                class_id = class_ids[i]
                class_name = model.names[class_id]
                class_counters[class_name] += 1
                display_id = f"{class_name}-{class_counters[class_name]}"
                track_id_to_display_id[track_id] = display_id

                tracked_objects[track_id]['display_id'] = display_id
                tracked_objects[track_id]['class_id'] = class_id
                tracked_objects[track_id]['first_seen'] = frame_number

            tracked_objects[track_id]['last_seen'] = frame_number

    print(f"Scanning frame {frame_number}/{total_frames}", end='\r')

print(f"\nDiscovered {len(tracked_objects)} unique objects.")
cap.release()

# --- Re-initialize for Pass 2 ---
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))
print("Pass 2/2: Drawing video with track IDs...")
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_number += 1

    results = model.track(frame, persist=True, verbose=False)
    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            track_id = track_ids[i]
            class_id = class_ids[i]

            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # --- Draw the class-specific ID on the box ---
            display_id = track_id_to_display_id.get(track_id, f"ID-{track_id}")
            label_position = (x1, y1 - 10)
            cv2.putText(annotated_frame, display_id, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    out.write(annotated_frame)
    print(f"Writing frame {frame_number}/{total_frames}", end='\r')

# --- Finalize ---
final_output = {"video_metadata": {"total_frames": total_frames, "fps": fps}, "timeline": dict(tracked_objects)}
with open(JSON_OUTPUT_PATH, 'w') as f: json.dump(final_output, f, indent=4)

print(f"\nProcessing complete.")
print(f"Tracked video saved to: {OUTPUT_VIDEO_PATH}")
print(f"Timeline data saved to: {JSON_OUTPUT_PATH}")

cap.release()
out.release()