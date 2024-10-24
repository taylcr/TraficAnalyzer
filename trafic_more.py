import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort import Sort
import random
from datetime import datetime

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Initialize video capture from video file
cap = cv2.VideoCapture("cars.mp4")

# Initialize the SORT tracker
tracker = Sort()

# Class names for YOLO (COCO dataset classes)
tracked_classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light"]

# Variables to store the start and end points of lines
line_pairs = []
current_line_start = None
current_line_end = None
drawing = False
line_set = False

# Colors for each pair of lines
line_colors = []

# Initialize a list to count the objects that complete each pair of lines
object_counters = {cls: [0 for _ in range(10)] for cls in tracked_classes}

# Store completed object IDs to avoid double counting
completed_objects = {cls: set() for cls in tracked_classes}

# Mouse callback function to draw lines
def draw_lines(event, x, y, flags, param):
    global current_line_start, current_line_end, drawing, line_set
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_line_start = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_line_end = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_line_end = (x, y)
        if len(line_pairs) % 2 == 0:
            # Add start line
            line_pairs.append((current_line_start, current_line_end))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
            line_colors.append(color)
        else:
            # Add end line with the same color as the start line
            line_pairs.append((current_line_start, current_line_end))
        current_line_start = None
        current_line_end = None
        line_set = True

# Capture the first frame for line setup
success, first_frame = cap.read()

if success:
    # Display the first frame and allow user to draw lines
    cv2.namedWindow("Setup Lines")
    cv2.setMouseCallback("Setup Lines", draw_lines)

    while True:
        img_copy = first_frame.copy()
        
        # Draw lines as they are being drawn and finalized
        if drawing and current_line_start and current_line_end:
            cv2.line(img_copy, current_line_start, current_line_end, (0, 255, 0), 2)
        for i in range(0, len(line_pairs), 2):
            cv2.line(img_copy, line_pairs[i][0], line_pairs[i][1], line_colors[i//2], 2)  # Start line
            if i + 1 < len(line_pairs):
                cv2.line(img_copy, line_pairs[i + 1][0], line_pairs[i + 1][1], line_colors[i//2], 2)  # End line

        cv2.imshow("Setup Lines", img_copy)

        # Press 'Enter' key to finalize and proceed
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and line_set:  # 'Enter' key
            break
        elif key == ord('q'):  # Exit on 'q' key press
            line_set = False
            break

    cv2.destroyWindow("Setup Lines")

if not line_set:
    print("No lines were set. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Function to check if an object crosses a line by comparing its previous and current positions
def check_line_crossing(prev_center, curr_center, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    
    # Check if the object is moving from one side of the line to the other
    prev_side = (prev_center[0] - x1) * (y2 - y1) - (prev_center[1] - y1) * (x2 - x1)
    curr_side = (curr_center[0] - x1) * (y2 - y1) - (curr_center[1] - y1) * (x2 - x1)
    
    if prev_side * curr_side < 0:  # If the sign changes, the object has crossed the line
        return True
    return False

# Process the video frames
start_time = time.time()  # Record the start time

while True:
    success, img = cap.read()
    if not success:
        break

    # YOLO detection
    results = model(img, stream=True)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Ensure the class index is within the range of classNames and is part of the tracked classes
            if cls < len(classNames) and classNames[cls] in tracked_classes:
                detections.append([x1, y1, x2, y2, conf])

    # Convert detections to np.array format required for SORT
    detections = np.array(detections)

    # Update SORT tracker with detections
    tracked_objects = tracker.update(detections)

    # Ensure line_colors has enough colors for the number of line pairs
    while len(line_colors) < len(line_pairs) // 2:
        line_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # Draw tracking results
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        center_y = (y1 + y2) // 2  # Calculate the center y position of the object
        center_x = (x1 + x2) // 2  # Calculate the center x position of the object
        obj_center = (center_x, center_y)

        cls = None
        # Iterate over the results to get the class label
        for result in results:
            for box in result.boxes:
                if int(track_id) == int(box.id):
                    cls = classNames[int(box.cls[0])]

        # Skip if class is not found or the object class is not in tracked_classes
        if cls not in tracked_classes:
            continue

        # Draw the bounding box and the ID on the frame
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{cls.capitalize()} {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Skip objects that have already completed the crossing
        if track_id in completed_objects[cls]:
            continue

        # Track objects across line pairs
        if track_id not in completed_objects[cls]:
            prev_positions = obj_center  # Initialize previous position

        # Check each pair of start and end lines
        for idx in range(0, len(line_pairs), 2):
            start_line = line_pairs[idx]
            end_line = line_pairs[idx + 1] if idx + 1 < len(line_pairs) else None

            if not completed_objects[cls]:
                # Check if object crosses the start line
                if check_line_crossing(prev_positions, obj_center, start_line):
                    completed_objects[cls].add(track_id)  # Mark the object as having crossed the start line

            elif end_line and check_line_crossing(prev_positions, obj_center, end_line):
                # Object crossed both start and end lines
                cv2.putText(img, f'{cls.capitalize()} {int(track_id)} completed {idx//2+1}', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                object_counters[cls][idx // 2] += 1  # Increment the counter for this pair

    # Draw the lines on the screen
    for i in range(0, len(line_pairs), 2):
        cv2.line(img, line_pairs[i][0], line_pairs[i][1], line_colors[i//2], 2)  # Start line
        if i + 1 < len(line_pairs):
            cv2.line(img, line_pairs[i + 1][0], line_pairs[i + 1][1], line_colors[i//2], 2)  # End line

    # Display line-based pair counters for each object class
    y_offset = 50
    for cls in tracked_classes:
        for i, count in enumerate(object_counters[cls]):
            if i < len(line_colors):  # Ensure we don't exceed available colors
                color = line_colors[i]
                cv2.putText(img, f'{cls.capitalize()} Pair {i+1} Count: {count}', 
                            (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30

    # Display the updated frame
    cv2.imshow("Traffic Analysis", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After video processing, log the data
stop_time = time.time()  # Record the stop time
start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
stop_time_str = datetime.fromtimestamp(stop_time).strftime('%Y-%m-%d %H:%M:%S')

log_data = f"Log Start Time: {start_time_str}\nLog Stop Time: {stop_time_str}\n"
log_data += "Pair Crossing Counters (by object type):\n"
for cls in tracked_classes:
    log_data += f"\n{cls.capitalize()}:\n"
    for i, count in enumerate(object_counters[cls]):
        log_data += f"  Pair {i+1}: {count} crossed\n"

# Write the log to a file
with open("log.txt", "w") as log_file:
    log_file.write(log_data)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
