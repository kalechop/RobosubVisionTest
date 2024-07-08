from ultralytics import YOLO
import cv2
import cvzone
import math

img = "counterclockwise.png"
model = YOLO("zed_right_24_06_15.pt")
results = model(img)[0]

classNames = ['Counterclockwise Banner ', 'Clockwise Banner ', 'Buoy ', 'Torpedo Banner', 'Torpedo Holes', 'Full Gate']

detections = results.boxes

# get a tensor of the location of detections: size = (# of detections, 4)
detection_loacations = detections.xywh 

# get a tensor of the confidence scores of detections: size = (# of detections)
detection_confidence = detections.conf

# get a tensor of class IDs of the detection: size = (# of detections)
detection_IDs = detections.id

# detection_loacations[i], detection_confidence[i] 
# and detection_IDs[i] all corespond to the same detection
results = model(source = img, show = True, conf = 0.5, save = True)