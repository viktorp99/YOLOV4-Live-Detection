# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# setup video capture
cap = cv2.VideoCapture(0)

'''
Setup the detector 
'''
# load class names
with open('coco.names', 'r') as f:
    classes = f.read().splitlines() 

# load detector config and weights    
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(512,1024), swapRB=True)

with tf.device('/gpu:0'):
    '''
    Process the whole video
    '''
    print(tf.test.gpu_device_name())
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)
            # flip frame this is a MPEG Bug !!!
            frame = frame[::-1,:,:]
    
    
            # do the inference 
            if True:
                classIds, scores, boxes = model.detect(frame, confThreshold=0.2, nmsThreshold=0.1) 
            else:
                classIds = []
                scores = []
                boxes = []
    
            # do the visualisation
            frame_vis = frame.copy()
            # loop over all detections
            for (classId, score, box) in zip(classIds, scores, boxes):
                # color from class id 
                hsv = np.uint8([[[classId*30+30,255,255 ]]])
                box_color = tuple( cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR).squeeze().astype('float') )
                #box_color = (0, 255, 0)
                # draw rectangels
                cv2.rectangle(frame_vis, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=box_color, thickness=1)
                # add text with class and score information
                text = '%s: %.2f' % (classes[classId], score)
                cv2.putText(frame_vis, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,color=box_color, thickness=1)
    
    
            cv2.imshow('frame',frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()