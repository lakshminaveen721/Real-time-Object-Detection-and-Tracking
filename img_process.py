import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path

# Initialize tracker
if hasattr(cv2, 'TrackerCSRT_create'):
    tracker = cv2.TrackerCSRT_create()
else:
    raise AttributeError("CSRT tracker is not available in this version of OpenCV")


# Function to detect objects
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, class_ids, confidences

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, class_ids, confidences = detect_objects(frame)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Initialize tracker with first detected object
            tracker.init(frame, (x, y, w, h))
            break

    # Update tracker and draw tracked object
    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    else:
        print("Tracker update failed")
    
    cv2.imshow("Real-time Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
