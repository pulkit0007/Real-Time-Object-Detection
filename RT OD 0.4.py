from tkinter import *
import tkinter as tk
import cv2
import numpy as np
from datetime import datetime
import time
import os
from tkinter.filedialog import askopenfilename
import tkinter.filedialog

def img_obj_detcn():

    
    net = cv2.dnn.readNet("weights/PM_od_fast.weights", "cfg/PM_od_fast.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

   
    filename = askopenfilename() 
    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    
    class_ids =[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x=int(center_x-w/2)
                y=int(center_y-h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_ids)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            color=colors[i]
            cv2.rectangle(img,(x, y),(x + w, y + h),color, 2)
            cv2.putText(img, label,(x, y + 30),font,3,color, 3)

    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    filename.close()
    cv2.destroyAllWindows()

def rt_obj_detcn():

    net = cv2.dnn.readNet("weights/PM_od_fast.weights", "cfg/PM_od_fast.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    cap=cv2.VideoCapture(0)
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter('Output.avi',fourcc,20.0,(640,480))
    starting_time = time.time()
    frame_id = 0
    while True:
        _, frame = cap.read()
        frame_id += 1
        height, width, channels = frame.shape

        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0, 0, 0), True, crop=False)

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
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h=boxes[i]
                label=str(classes[class_ids[i]])
                color=colors[i]
                cv2.rectangle(frame,(x, y),(x + w, y + h),color, 2)
                cv2.putText(frame,label,(x,y+30),font,3,color,3)

        elapsed_time = time.time() - starting_time
        fps=frame_id / elapsed_time
        cv2.putText(frame,str(datetime.now()),(20,30),font,1,(255,255,255),2,cv2.LINE_AA) # Print current time
        cv2.putText(frame,"FPS:"+str(round(fps, 2)),(410, 455),font,1.5,(230, 230, 230), 3) # Print FPS
        cv2.imshow("Live Feed",frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break;
    cap.release()
    out.release()
    cv2.destroyAllWindows()

v=tk.Tk()
v.iconbitmap('cam.ico') 
frame=tk.Frame(v)
frame.pack()
v.title("Cam Object Detector") 
T=tk.Text(v, height=5, width=60)
T.pack()

T.insert(tk.END,"Project By :- Pulkit Mehta \n ")
 

button_1=tk.Button(frame,text="Detect from Image",font=36,height=8,width=36,command=img_obj_detcn)
button_2=tk.Button(frame,text="Detect from WebCam",font=36,height=8,width=36,command=rt_obj_detcn)
button_3=tk.Button(frame,text="Exit",font=36,height=8,width=36,command=quit)

button_1.pack()
button_2.pack()
button_3.pack()

v.mainloop()
