import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import random
import time
from datetime import datetime

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")  # Using a smaller YOLO model for faster inference

# Initialize video capture from video file
cap = cv2.VideoCapture("cars.mp4")

# Initialize the SORT tracker
tracker = Sort()

# Class names for YOLO (COCO dataset classes)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light"]

# We will now track more types of objects
tracked_classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

# Variables to store the start and end points of lines
line_pairs = []
current_line_start = None
current_line_end = None
drawing = False
line_set = False

# Colors for each pair of lines
line_colors = []

# Initialize a list to count the objects that complete each pair of lines
object_counters = []

# Store completed object IDs to avoid double counting
completed_objects = set()

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
            line_colors.append(color)  # Assign the same color to the pair
            object_counters.append({"person": 0, "bicycle": 0, "car": 0, "motorbike": 0, "bus": 0, "truck": 0})  # Initialize counters for this pair
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

# Initialize variables for tracking
tracked_object_states = {}  # Dictionary to store object states
prev_positions = {}  # Store previous positions of objects to check direction of movement

# Start time for logging
start_time = datetime.now()

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
while True:
    success, img = cap.read()
    if not success:
        break

    # YOLO detection
    results = model(img, stream=True)

    detections = []
    detection_classes = {}  # To store class type by track ID
    for r in results:
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Ensure the class index is within the range of classNames and track only specified objects
            if cls < len(classNames) and classNames[cls] in tracked_classes:
                detections.append([x1, y1, x2, y2, conf])  # Only pass the coordinates and confidence
                detection_classes[len(detections) - 1] = classNames[cls]  # Store the class for each detection
    
    # Convert detections to np.array format required for SORT
    detections = np.array(detections)

    # Update SORT tracker with detections
    tracked_objects = tracker.update(detections)

    # Draw tracking results
    for i, track in enumerate(tracked_objects):
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        center_y = (y1 + y2) // 2  # Calculate the center y position of the object
        center_x = (x1 + x2) // 2  # Calculate the center x position of the object
        object_center = (center_x, center_y)

        # Skip objects that have already completed the crossing
        if track_id in completed_objects:
            continue

        # Retrieve the class from the detection_classes
        object_type = detection_classes.get(i, "Unknown")

        # Draw bounding box and track ID with object type
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{object_type.capitalize()} {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Track objects across line pairs
        if track_id not in tracked_object_states:
            tracked_object_states[track_id] = [False] * (len(line_pairs) // 2)  # Track if the object has crossed each start line
            prev_positions[track_id] = object_center  # Initialize previous position
        
        # Check each pair of start and end lines
        for idx in range(0, len(line_pairs), 2):
            start_line = line_pairs[idx]
            end_line = line_pairs[idx + 1] if idx + 1 < len(line_pairs) else None

            if not tracked_object_states[track_id][idx // 2]:
                # Check if object crosses the start line
                if check_line_crossing(prev_positions[track_id], object_center, start_line):
                    tracked_object_states[track_id][idx // 2] = True  # Mark the object as having crossed the start line
            elif end_line and check_line_crossing(prev_positions[track_id], object_center, end_line):
                # Object crossed both start and end lines
                cv2.putText(img, f'{object_type.capitalize()} {int(track_id)} completed {idx//2+1}', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                object_counters[idx // 2][object_type] += 1  # Increment the counter for this pair
                completed_objects.add(track_id)  # Add object to completed set to avoid double counting

        # Update the previous position of the object
        prev_positions[track_id] = object_center

    # Draw the lines on the screen
    for i in range(0, len(line_pairs), 2):
        cv2.line(img, line_pairs[i][0], line_pairs[i][1], line_colors[i//2], 2)  # Start line
        if i + 1 < len(line_pairs):
            cv2.line(img, line_pairs[i + 1][0], line_pairs[i + 1][1], line_colors[i//2], 2)  # End line

    # Display the counters for each pair with a smaller font size
    for i, counts in enumerate(object_counters):
        color = line_colors[i]
        person_count = counts["person"]
        bicycle_count = counts["bicycle"]
        car_count = counts["car"]
        motorbike_count = counts["motorbike"]
        bus_count = counts["bus"]
        truck_count = counts["truck"]
        cv2.putText(img, f'Pair {i+1} - Person: {person_count}, Bicycle: {bicycle_count}, Car: {car_count}, Motorbike: {motorbike_count}, Bus: {bus_count}, Truck: {truck_count}', 
                    (50, 50 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Display the updated frame
    cv2.imshow("Traffic Analysis", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# End time for logging
end_time = datetime.now()

# Write log file
with open("log.txt", "w") as log_file:
    log_file.write(f"Log Start Time: {start_time}\n")
    log_file.write(f"Log Stop Time: {end_time}\n")
    log_file.write(f"Pair Crossing Counters (by object type):\n\n")

    for i, counts in enumerate(object_counters):
        log_file.write(f"Pair {i+1}:\n")
        log_file.write(f"  Person: {counts['person']} crossed\n")
        log_file.write(f"  Bicycle: {counts['bicycle']} crossed\n")
        log_file.write(f"  Car: {counts['car']} crossed\n")
        log_file.write(f"  Motorbike: {counts['motorbike']} crossed\n")
        log_file.write(f"  Bus: {counts['bus']} crossed\n")
        log_file.write(f"  Truck: {counts['truck']} crossed\n\n")
