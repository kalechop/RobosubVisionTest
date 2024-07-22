import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('best.pt')

def find_center(image): 
  input = cv2.imread(image)
  gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

  canny_edges = cv2.Canny(gray, 30, 200)

  M = cv2.moments(canny_edges)

  cx = int(M['m10']/M['m00'])
  cy = int(M['m01']/M['m00'])

  circle = cv2.circle(input, (cx, cy), 10, (255, 255, 255), -1)
  cv2.putText(input, "C", (cx-25, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 10, 2)

  fig = plt.figure(figsize = (5,5))
  plt.imshow(input)
  plt.xticks([])
  plt.yticks([])

def plot_bboxes(results): 
    xyxys = []
    confidences = []
    class_ids = []
    
    for result in results: 
        
        boxes = results.boxes.cpu().numpy()
        xyxy = boxes.xyxy
        
        print(xyxy)
        
        #for xyxy in xyxys: 
         #   cv2.rectangle(frame, (int(xyxy[0])), )
    
def main(): 
    results = model(source=0, show=True, conf=0.3, save=True)
    detections = results.boxes
    detection_locations = detections.xyxy
    
    print(detection_locations)
    #find_center(detections)

    # get a tensor of the location of detections: size = (# of detections, 4)
    #detection_locations = detections.xywh 

    #results = model(source=0, show=True, conf=0.3, save=True)
    #find_center('counterclockwise.png')
    
   # while (True): 
    #    results = model(source=0, conf=0.3)
     #   print(results.boxes.xyxy)
    
        #for result in results: 
        #    boxes = result.boxes.cpu().numpy()
        #   xyxy = boxes.xyxy
        #  
        # print(xyxy)

        #find_center(boxes.xyxy)
        
def test():
    img = "gate.png"
    model = YOLO("best.pt")
   # results = model(img)[0]
   # detections = results.boxes

    # get a tensor of the location of detections: size = (# of detections, 4)
   # detection_locations = detections.xyxy
    ###center_x = (detection_locations[0]+detection_locations[2])/2
    #center_y = (detection_locations[1] + detection_locations[3])/2
    #cv2.putText(img, "C", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 10, 2)
    
    #print(center_x)
    #print(center_y)
    
    
    #print(detection_locations) 
    #print(detection_locations[1])
    
    results = model(img)[0]
    
    #detection_boxes = results.boxes
        
    #detections = detection_boxes.xyxy[0]  # Detections with [xmin, ymin, xmax, ymax, confidence, class]
    
    for r in results: 
        for box in r.boxes.xyxy: 
            centroid_x = int((box[0] + box[2]) / 2)
            centroid_y = int((box[1] + box[3]) / 2)
            
            print(centroid_x)
            print(centroid_y)

  
    #xmin = detections[0]
    #ymin = detections[1]
    #xmax = detections[2]
    #ymax = detections[3]
    # Calculate centroid
    #cx = (xmin + xmax) / 2
    #cy = (ymin + ymax) / 2
    ## Convert to real-world coordinates (assuming simple scale factor for demo)
    #real_x = 1 * cx
    #real_y = 1 * cy
    # Generate robot arm commands based on converted coordinates
    #print(real_x)
    #print(real_y)
        
    # prints center on image
    #cv2.putText(img, "C", (real_x, real_y), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 10, 2)
    #results = model(source = img, show = True, conf = 0.5, save = True)
    
test()
