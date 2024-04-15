#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 01:24:37 2023

@author: iamlokki
"""
#Read a video from web cam using OpenCV
#Face Detection in a Video
#CLick pitcures of the person and save them as a numpy
import cv2
import numpy as np

#Creating a camera video

cam = cv2.VideoCapture(0)

#Ask the name of the person

fileName = input("Enter the name of the person:")
dataset_path = "./data/"
offset = 30

#model

model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Create a list to save face data
faceData = []
skip = 0

#Reading image from camera object
while True:
    success, img = cam.read()
    if not success:
        print("Reading camera failed")
    
    #Store the gray images
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = model.detectMultiScale(img,1.3,5)
    
    #Sorting the face with the largest bounding box
    faces = sorted(faces,key=lambda f:f[2]*f[3])
    
    #Pick the largest face
    if len(faces)>0:
        f = faces[-1]
    
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
        #Crop and save the largest face
        cropped_face = img[y-offset:y+h+offset, x-offset:x+w+offset]
        cropped_face = cv2.resize(cropped_face, (100,100))
        skip += 1
        if skip %10 == 0:
            faceData.append(cropped_face)
            print("Face Saved:" + str(len(faceData)))
        
    cv2.imshow("Image Window", img)
    #cv2.imshow("Cropped Face", cropped_face)
    key = cv2.waitKey(1) #Pause for 1 ms before taking the next image
    if key == ord('q'):
        break

#Write the face data n the disk
faceData = np.asarray(faceData)
print(faceData.shape)

#Reshaping the data
m =faceData.shape[0]
faceData = faceData.reshape((m,-1))
print(faceData.shape)

#Save on the disk
filepath = dataset_path + fileName + ".npy"
np.save(filepath, faceData)
print("Data saved successfully at :" + filepath)

#Releae camera and destroy window
cam.release()
cv2.destroyAllWindows()
