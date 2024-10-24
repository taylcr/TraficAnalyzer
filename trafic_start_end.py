import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import random

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")  # Using a smaller YOLO model for faster inference

# Initialize video capture from video file
cap = cv2.VideoCapture("cars.mp4")

# Initialize the SORT tracker
tracker = Sort()

# Class names for YOLO (COCO dataset classes)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light"]

# Variables to store the start and end points of lines
line_pairs = []
current_line_start = None
current_line_end = None
drawing = False
line_set = False

# Colors for each pair of lines
line_colors = []

# Initialize a list to count the cars that complete each pair of lines
car_counters = []

# Store completed car IDs to avoid double counting
completed_cars = set()

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
            car_counters.append(0)  # Initialize car counter for this pair
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
tracked_cars = {}
prev_positions = {}  # Store previous positions of cars to check direction of movement

# Function to check if a car crosses a line by comparing its previous and current positions
def check_line_crossing(prev_center, curr_center, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    
    # Check if the car is moving from one side of the line to the other
    prev_side = (prev_center[0] - x1) * (y2 - y1) - (prev_center[1] - y1) * (x2 - x1)
    curr_side = (curr_center[0] - x1) * (y2 - y1) - (curr_center[1] - y1) * (x2 - x1)
    
    if prev_side * curr_side < 0:  # If the sign changes, the car has crossed the line
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
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Ensure the class index is within the range of classNames
            if cls < len(classNames) and classNames[cls] == "car":  # Only track cars
                detections.append([x1, y1, x2, y2, conf])
    
    # Convert detections to np.array format required for SORT
    detections = np.array(detections)

    # Update SORT tracker with detections
    tracked_objects = tracker.update(detections)

    # Draw tracking results
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        center_y = (y1 + y2) // 2  # Calculate the center y position of the car
        center_x = (x1 + x2) // 2  # Calculate the center x position of the car
        car_center = (center_x, center_y)

        # Skip cars that have already completed the crossing
        if track_id in completed_cars:
            continue

        # Draw bounding box and track ID
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'Car {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Track cars across line pairs
        if track_id not in tracked_cars:
            tracked_cars[track_id] = [False] * (len(line_pairs) // 2)  # Track if the car has crossed each start line
            prev_positions[track_id] = car_center  # Initialize previous position
        
        # Check each pair of start and end lines
        for idx in range(0, len(line_pairs), 2):
            start_line = line_pairs[idx]
            end_line = line_pairs[idx + 1] if idx + 1 < len(line_pairs) else None

            if not tracked_cars[track_id][idx // 2]:
                # Check if car crosses the start line
                if check_line_crossing(prev_positions[track_id], car_center, start_line):
                    tracked_cars[track_id][idx // 2] = True  # Mark the car as having crossed the start line
            elif end_line and check_line_crossing(prev_positions[track_id], car_center, end_line):
                # Car crossed both start and end lines
                cv2.putText(img, f'Car {int(track_id)} completed {idx//2+1}', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                car_counters[idx // 2] += 1  # Increment the counter for this pair
                completed_cars.add(track_id)  # Add car to completed set to avoid double counting

        # Update the previous position of the car
        prev_positions[track_id] = car_center

    # Draw the lines on the screen
    for i in range(0, len(line_pairs), 2):
        cv2.line(img, line_pairs[i][0], line_pairs[i][1], line_colors[i//2], 2)  # Start line
        if i + 1 < len(line_pairs):
            cv2.line(img, line_pairs[i + 1][0], line_pairs[i + 1][1], line_colors[i//2], 2)  # End line

    # Display the car counters for each pair
    for i, count in enumerate(car_counters):
        color = line_colors[i]
        cv2.putText(img, f'Pair {i+1} Count: {count}', (50, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the updated frame
    cv2.imshow("Traffic Analysis", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
