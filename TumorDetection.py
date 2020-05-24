#!/usr/bin/env python
# coding: utf-8

import cv2 
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import h5py
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn import mixture
from sklearn import datasets, svm, metrics
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import mysql.connector
import base64
import io

currentdir = os.getcwd()
img_dir = currentdir + "\data"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
global_features = []
labels = []
train_path = img_dir
h5_data = currentdir + "\data.h5"
h5_labels = currentdir + "\labels.h5"
train_labels = os.listdir(train_path)
train_labels.sort()
connection = mysql.connector.connect(host='127.0.0.1', database='tumors', user='root', port='3307', password='password')

def trainTest():
    for i in range(len(files)):
        path = files[i]
        detectTumor(path)
    connection.close()    
    
def detectTumor(path):
    image = readImage(path)
    preprocessed = preprocessImage(image)
    segmented = segmentImage(preprocessed)
    postprocessed = postprocessImage(segmented)
    contour = drawContour(segmented, postprocessed)
    result = findTumor(postprocessed)
    writeDB(image,result)
    createFeatureVector(image)
        
def readImage(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (400, 400))
    return image
    
def preprocessImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    ret, markers = cv2.connectedComponents(thresh)
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    largest_component = np.argmax(marker_area)+1                    
    brain_mask = markers==largest_component
    brain_out = image.copy()
    brain_out[brain_mask==False] = (0,0,0)
    gray = cv2.cvtColor(brain_out, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(brain_out, 3)
    return median
    
def segmentImage(image):
    data = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(data, 3, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)    
    res = center[label.flatten()]
    segmented = res.reshape(image.shape[0], image.shape[1], image.shape[2])
    return segmented
    
def postprocessImage(segmented):
    area = 0
    rows, cols, dims = segmented.shape
    pixels = list()
    reduced_pixels = list()
    for j in range(rows):
        for k in range(cols):
            pixels.append(segmented[j, k][0])
            pixels.append(segmented[j, k][1])
            pixels.append(segmented[j, k][2])
            
    pix_array = np.array(pixels) 
    reduced_pixels = np.unique(pix_array)
    pixel = int(sum(reduced_pixels)/(len(reduced_pixels) - 1))
    
    if(pixel >= 120):
        ret, thresh = cv2.threshold(segmented, pixel, 255, cv2.THRESH_BINARY)  
    else:
        ret, thresh = cv2.threshold(segmented, 120, 255, cv2.THRESH_BINARY)     
    
    kernel = np.ones((5, 5), np.uint8)  
    erode = cv2.erode(thresh, kernel)
    erode = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)
    return erode
    
def drawContour(segmented, erode):
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(segmented, contours, -1, (0, 0, 255), 3)
        
    return segmented
    
def findTumor(erode):
    area = 0
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, erode.shape[1]//2, erode.shape[0]
    left = erode[y:y+h, x:x+w]
    right = erode[y:y+h, x+w:x+w+w]
    left_pixels = cv2.countNonZero(left)
    right_pixels = cv2.countNonZero(right)
    ratio = -1 
    status = []

    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(contour)
            
        if area > 0:
            if left_pixels == 0 and right_pixels > 0:
                labels.append("yes")
                status.append("Yes")
                status.append("Right")
            elif right_pixels == 0 and left_pixels > 0:
                labels.append("yes")
                status.append("Yes")
                status.append("Left")
            elif right_pixels > left_pixels:
                ratio = float(right_pixels/left_pixels)
                if ratio >= 1.5:
                    labels.append("yes")
                    status.append("Yes")
                    status.append("Right")
                else:
                    labels.append("no")
                    status.append("No")
            else:
                ratio = float(left_pixels/right_pixels)
                if ratio >= 1.5:
                    labels.append("yes")
                    status.append("Yes")
                    status.append("Left")
                else:
                    labels.append("no")
                    status.append("No")
        else:
            labels.append("no")
            status.append("No")
    else:
        labels.append("no") 
        status.append("No")
        
    return status
   
def writeDB(image, status):
 
    a=''

    if status[0] == "No":
        b = "No"
        
    else:
        b = "Yes"
        a=status[1]
            
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    sql_insert_blob_query = """ INSERT INTO new_table
                            (image, isTumor, location) VALUES (%s,%s,%s)"""
    insert_blob_tuple = (img_str, b, a)
    cursor = connection.cursor()
    cursor.execute(sql_insert_blob_query,insert_blob_tuple )
    connection.commit()
    
    
def createFeatureVector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    global_features.append(feature)

def classifyTumor():
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)   
    h5f_data = h5py.File(h5_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
    h5f_label = h5py.File(h5_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))
    h5f_data.close()
    h5f_label.close()
    scoring = "accuracy"
    h5f_data = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')
    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']
    global_features_arr = np.array(global_features_string)
    global_labels = np.array(global_labels_string)
    h5f_data.close()
    h5f_label.close()
    x_train, x_test, y_train, y_test = train_test_split(global_features_arr, global_labels, test_size=0.25, shuffle=True)
    svm = SVC(random_state=9)
    kfold = KFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(svm, x_train, y_train, cv=kfold, scoring=scoring)
    accuracy =  cv_results.mean() * 100
    return accuracy

trainTest() 
classifyTumor()
