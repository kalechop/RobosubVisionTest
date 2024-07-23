from ultralytics import YOLO
import cv2
import cvzone
import math

img = "gate.png"
model = YOLO("best.pt")
#frame = cv2.resize(img, (640, 640))

cap = cv2.VideoCapture(0)

#Horizontal Field of View for Blue Robotics Camera
horizontal_fov_degrees = 80
vertical_fov_degrees = 64

results = model(source=img, show=True, conf=0.3, save=True)

for r in results: 
    for box in r.boxes.xyxy: 
        centroid_x = int((box[0] + box[2]) / 2)
        centroid_y = int((box[1] + box[3]) / 2)
            
        print((centroid_x, centroid_y))
        
        x_degrees = (centroid_x / 640) * horizontal_fov_degrees  # Use resized width
        y_degrees = (centroid_y / 640) * vertical_fov_degrees    # Use resized height
     
        #cv2.circle(img, (centroid_x, centroid_y), 20, (36,255,12), -1)
       # print('Center: ({}, {})'.format(cx,cy))
       # cv2.putText(img, ())

"""
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

"""
