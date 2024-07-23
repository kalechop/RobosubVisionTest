from ultralytics import YOLO
import numpy as np 
import cv2


model = YOLO('best.pt')

def find_angle_to_object(x1,y1,x2,y2):
    # Replace this with your actual calculation based on object coordinates
    # For example, you might calculate the angle based on the center of the object
    # and the center of the frame.
    frame_center = (320, 320)
    object_center = ((x1 + x2) // 2), ((y1 + y2 )// 2)
                     
    
    # Calculate the angle
    angle = np.degrees(np.arctan2(object_center[1] - frame_center[1], 
                                  object_center[0] - frame_center[0]))
    
    print(angle)

vid = cv2.VideoCapture(0)   

#results = model(source=0, show=True, conf=0.3, save=True)

while (True): 
    ret, frame = vid.read() 
    cv2.imshow('Inference', frame)
    results = model(source=frame, show=True, conf=0.3, save=True)
    
    
    """
    for r in results: 
        for box in r.boxes.xyxy: 
            centroid_x = int((box[0] + box[2]) / 2)
            centroid_y = int((box[1] + box[3]) / 2)
        
            print("Centroid: ", (centroid_x, centroid_y))        
            find_angle_to_object(box[0], box[1], box[2], box[3])
    
    """
    
    
    
    
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
cv2.destroyAllWindows() 

    
"""
for r in results: 
    for box in r.boxes.xyxy: 
        centroid_x = int((box[0] + box[2]) / 2)
        centroid_y = int((box[1] + box[3]) / 2)
        
        cv2.putText(img, "Hello", (0,0), 40, 40, (0,0,0), 8, 1, True)
        
        print("Centroid: ", (centroid_x, centroid_y))        
        find_angle_to_object(box[0], box[1], box[2], box[3])
        
      #  x_degrees = (centroid_x / 640) * horizontal_fov_degrees  # Use resized width
      #  y_degrees = (centroid_y / 640) * vertical_fov_degrees    # Use resized height
      """

"""
while cap.isOpened():
    #img = cap.read()
    results = model(source=0, show=True, conf=0.3, save=True)
    for r in results: 
        for box in r.boxes.xyxy: 
            centroid_x = int((box[0] + box[2]) / 2)
            centroid_y = int((box[1] + box[3]) / 2)
            
            centroid_value = "({centroid_x:.2f},{centroid_y:2f})" 
                        
            #print((centroid_x, centroid_y))
        
            #find_angle_to_object(box[0], box[1], box[2], box[3])
            
            frame_center = (320, 320)
            object_center = ((box[0] + box[2]) // 2), ((box[1] + box[3] )// 2)
                     
            # Calculate the angle
            angle = np.degrees(np.arctan2(object_center[1] - frame_center[1], 
                                  object_center[0] - frame_center[0]))
            
            cv2.putText(img, centroid_value, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 100, (0, 255, 255))
            cv2.putText(img, angle, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 50, (0, 244, 255))

    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2  = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            print(find_angle_to_object(x1,y1,x2,y2))

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(source=frame, conf=0.7, imgsz=640)

        boxes = results[0].boxes
        for i in range(len(boxes.cls)):
            print("")
            print(classes[int(boxes.cls[i])])
            print("confidence:", float(boxes.conf[i]))
            print("left:", float(boxes.xyxy[i][0]))
            print("right:", float(boxes.xyxy[i][2]))
            print("top:", float(boxes.xyxy[i][1]))
            print("bottom:", float(boxes.xyxy[i][3]))
            print("centroid: (" + str((float(boxes.xyxy[i][0]) + float(boxes.xyxy[i][2])) * 0.5) + ", " + str((float(boxes.xyxy[i][1]) + float(boxes.xyxy[i][3])) * 0.5) + ")")
            print("offset from center: " + str(320 - (float(boxes.xyxy[i][0]) + float(boxes.xyxy[i][2])) * 0.5) + ", " + str(320 - (float(boxes.xyxy[i][1]) + float(boxes.xyxy[i][3])) * 0.5) + ")")

        cv2.imshow("YOLOv8 Inference (live)", results[0].plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

"""