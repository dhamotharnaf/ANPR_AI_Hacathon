import numpy as np 
import pandas as pd 
import cv2 
import math 
import csv
import collections
import time
from sqlalchemy import create_engine
from skimage.filters import threshold_local
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from skimage import measure
import imutils



# Vehicle Tracker, Capture, and Speed Calculation class with member functions respectively
y_a = 80
y_b = 100
y_c = 250
y_d = 270

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
        self.et=0
        self.s1 = np.zeros((1,1000))
        self.s2 = np.zeros((1,1000))
        self.s = np.zeros((1,1000))
        self.f = np.zeros(1000)
        self.capf = np.zeros(1000)
        self.count = 0
        self.exceeded = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 40: 
                    self.center_points[id] = (cx, cy)
                    
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True
                    
                    # start timer s1
                    if (y>=y_c and y<=y_d): 
                        self.s1[0,id] = time.time() 
                        
                    # stop timer and find difference 
                    if (y>=y_a and y<=y_b): 
                        self.s2[0, id] = time.time() 
                        self.s[0,id] = self.s2[0,id] - self.s1[0,id]

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1
                self.s[0,self.id_count] = 0 
                self.s1[0,self.id_count] = 0 
                self.s2[0,self.id_count] = 0

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, index = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
            

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    def getsp(self,id): 
        if (self.s[0,id] != 0): 
            #s = 214.15/ self.s[0, id]
            s = 10000000000/ self.s[0, id]
            s = s * 3
        else: 
            s = 0
        return int(s)

    def capture(self,img,x,y,h,w,sp,id):
        if(self.capf[id]==0):
            self.capf[id] = 1
            self.f[id]=0
            crop_img = img[y-5:y + h+5, x-5:x + w+5]
            n = str(id)+"_speed_"+str(sp)
            file =  './'+n+'.jpg'
            cv2.imwrite(file, crop_img)
            self.count += 1
 
    
    def limit(self): 
        return limit
print("Try to save the video file in the current directory ")
vidpath = input("Enter the video path here: ")

# Initialize Tracker
tracker = EuclideanDistTracker()
limit = 70

# Initialize the videocapture object
cap = cv2.VideoCapture(vidpath)
#input_size = 415

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 225   
up_line_position = 90
down_line_position = 260


# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')


# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicle
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id
            
   
        

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here


# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames , detection, boxes_ids, s, name, confidence_scores
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)

        # Draw classname and confidence score 
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])
        
    # Update the tracker for each object


def realTime():
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(0,0),None,0.5,0.5)
        ih, iw, channels = img.shape
        input_size = iw - ih - 5 
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        
        # Feed data to the network
        outputs = net.forward(outputNames)
      
    
        # Find the objects from the network output
        postProcess(outputs,img)
        boxes_ids = tracker.update(detection)
        speed_lst = []
        for box_id in boxes_ids:
            count_vehicle(box_id, img)
            x,y,w,h,id,index = box_id
        
            s = tracker.getsp(id)
            speed_lst.append(s)
            
            text= "       id:"+str(id)+' speed:'+str(s)+'Km/h'



            if(tracker.getsp(id)<limit):
                cv2.putText(img,text,(x+30,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

            else:
                cv2.putText(img,text,(x+30, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255),1)

            s = tracker.getsp(id)
            if (tracker.f[id] == 1):
                tracker.capture(img, x, y, h, w, s, id)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    # Write the vehicle counting information in a file and save it

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        speed_lst.insert(0, "Speed")
        boxes_ids.insert(0,"ID")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
        cwriter.writerow(speed_lst)
        cwriter.writerow(boxes_ids)
        f1.close()
    
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    realTime()
    
