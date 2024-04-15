#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 02:41:24 2023

@author: iamlokki
"""

import cv2
import numpy as np
import os

#Data
dataset_path = "./data/"
faceData = []
labels = []
nameMap = {}
offset = 30
skip = 0

classID = 0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        
        nameMap[classID] = f[:-4]
        # X-value
        dataItem = np.load(dataset_path + f)
        #print(dataItem.shape)
        m = dataItem.shape[0]
        faceData.append(dataItem)
        
        # Y-value
        target = classID * np.ones((m,))
        classID+=1
        labels.append(target)

print(faceData)
print(labels)

#Adding the name labels together in the list of image data
Xt = np.concatenate(faceData, axis=0)
yt = np.concatenate(labels, axis=0).reshape((-1,1))

print(Xt.shape)
print(yt.shape)
print(nameMap)

#Algorithm
def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d, y[i]))

    dlist = sorted(dlist)
    labels = np.array([label for _, label in dlist[:k]], dtype=int)

    unique_labels, counts = np.unique(labels, return_counts=True)
    idx = counts.argmax()
    pred = unique_labels[idx]

    return int(pred)


#Prediction
#Creating a camera video
cam = cv2.VideoCapture(0)
#model
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    success, img = cam.read()
    if not success:
        print("Reading camera failed")

    
    faces = model.detectMultiScale(img,1.3,5)
    
    #Render a box around each face and prdict it's name
    for f in faces:
        x,y,w,h = f
    
        #Crop and save the largest face
        cropped_face = img[y-offset:y+h+offset, x-offset:x+w+offset]
        cropped_face = cv2.resize(cropped_face, (100,100))
        skip += 1
        
        cv2.imshow("Image Window", img)
        
        #Predict the name using KNN
        classPredicted = knn(Xt,yt, cropped_face.flatten())
        
        #Name predicted
        namePredicted = nameMap[classPredicted]
        
        #Display name and the box
        cv2.putText(img,namePredicted,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,200,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    #cv2.imshow("Prediction Window", img)
        
        
    #cv2.imshow("Cropped Face", cropped_face)
    key = cv2.waitKey(1) #Pause for 1 ms before taking the next image
    if key == ord('q'):
        break

#Releae camera and destroy window
cam.release()
cv2.destroyAllWindows()