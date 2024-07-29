
# Hobosub vision 2024 (we're cooked lol)

import cv2                       # input (webcam, video file, image file, etc)
from ultralytics import YOLO     # does the inferencing yay
import numpy as np               # does meth
import rospy                     # for publishing to topic

model = YOLO("models/best50.pt")   # import weights as .pt
cap = cv2.VideoCapture(0)          # setup video stream (filepath or 0 for webcam)

# open the ROS topic
pub = rospy.Publisher('vision_front', std_msgs.msg.String, queue_size=10)

classes = {0: 'bin', 1: 'bin_blue', 2: 'bin_red', 3: 'buoy', 4: 'gate', 5: 'gate_blue', 6: 'gate_red', 7: 'path', 8: 'torpedo', 9: 'torpedo_target'}

frame_size = (640, 480)
frame_center = (frame_size[0] / 2, frame_size[1] / 2)

def find_angle_to_object(x1,y1,x2,y2):
    object_center = ((x1 + x2) // 2), ((y1 + y2 )// 2)
    angle = np.degrees(np.arctan2(object_center[1] - frame_center[1], 
                                  object_center[0] - frame_center[0]))
    return(angle)

while cap.isOpened():
    # read one input frame
    success, frame = cap.read()

    # make sure the frame loaded correctly
    if success:
        # run inference on one frame
        results = model(source=frame, conf=0.3)

        # get data from inference in ultralytics.engine.results.boxes object
        # contains ordered lists with detection data
        boxes = results[0].boxes
        
        out = str(len(boxes.cls))

        # for each detected object in the current frame
        for i in range(len(boxes.cls)):
            # data will be outputted through ros topic
            left = float(boxes.xyxy[i][0])
            top = float(boxes.xyxy[i][1])
            right = float(boxes.xyxy[i][2])
            bottom = float(boxes.xyxy[i][3])
            
            # format
            # numberofobjects {object1class object1top object1left object1bottom object1right} {object1class object1top object1left object1bottom object1right}
            out += "{" + classes[int(boxes.cls[i])] + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + "} "
        
        pub.publish(std_msgs.msg.String(out))
        print(out)

        # shows the stuff
        cv2.imshow("YOLOv8 Inference (live)", results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
