#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import mysql.connector
import base64
import io

currentdir = os.getcwd()
img_dir = currentdir + "\data"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
global_features = []
labels  = []
train_path = img_dir
h5_data = currentdir + "\data.h5"
h5_labels = currentdir + "\labels.h5"
train_labels = os.listdir(train_path)
train_labels.sort()
connection = mysql.connector.connect(host='127.0.0.1', database='tumordetection', user='root', port='3307', password='password')
for i in range(len(files)):
    path = files[i]
    folder = path.strip(img_dir)
    folder = folder.strip(".jpg")
    directory = currentdir + "\\output\\" + folder
    if not os.path.exists(directory):
        os.mkdir(directory)
    image = cv2.imread(path) 
    img = cv2.resize(image, (400, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(directory + "\original.jpg", img) 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    ret, markers = cv2.connectedComponents(thresh)
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    largest_component = np.argmax(marker_area)+1                    
    brain_mask = markers==largest_component
    brain_out = img.copy()
    brain_out[brain_mask==False] = (0,0,0)
    cv2.imwrite(directory + "\skullmasking.jpg", brain_out) 
    
    gray = cv2.cvtColor(brain_out, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(brain_out, 3)
  
    data = median.reshape(median.shape[0]*median.shape[1], median.shape[2])
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(data, 3, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)    
    res = center[label.flatten()]
    segmented = res.reshape(median.shape[0], median.shape[1], median.shape[2])
    cv2.imwrite(directory + "\segmented.jpg", segmented) 

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
    cv2.imwrite(directory + "\\threshold.jpg", thresh)
   
    kernel = np.ones((5, 5), np.uint8)  
    erode = cv2.erode(thresh, kernel)
    erode = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite(directory + "\erode.jpg", erode)
    
    x, y, w, h = 0, 0, erode.shape[1]//2, erode.shape[0]
    left = erode[y:y+h, x:x+w]
    right = erode[y:y+h, x+w:x+w+w]
    left_pixels = cv2.countNonZero(left)
    right_pixels = cv2.countNonZero(right)
    ratio = -1       
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    global_features.append(feature)
    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(contour)
        cv2.drawContours(segmented, contours, -1, (0, 255, 0), 3)
        cv2.imwrite(directory + "\contour.jpg", segmented)
        perimeter = cv2.arcLength(contour, True)
        if area > 0:
            if left_pixels == 0 and right_pixels > 0:
                labels.append("yes")
            elif right_pixels == 0 and left_pixels > 0:
                labels.append("yes")
            elif right_pixels > left_pixels:
                ratio = float(right_pixels/left_pixels)
                if ratio >= 1.5:
                    labels.append("yes")
                else:
                    labels.append("no")
            else:
                ratio = float(left_pixels/right_pixels)
                if ratio >= 1.5:
                    labels.append("yes")
                else:
                    labels.append("no")
        else:
            labels.append("no")
            break
    else:
        labels.append("no")
    if(labels[i]=="no"):
        label_bool = 0
    else:
        label_bool = 1
    
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    img_str2=cv2.imencode('.jpg', segmented)[1].tostring()
    img_str3=cv2.imencode('.jpg', erode)[1].tostring()
    sql_insert_blob_query = """ INSERT INTO detection
                             (original_image,segmented_image,classified_image, isTumor, location) VALUES (%s,%s,%s,%s,%s)"""
    insert_blob_tuple = (img_str, img_str2,img_str3,label_bool, ?loction)
    cursor = connection.cursor()
    cursor.execute(sql_insert_blob_query,insert_blob_tuple )
    connection.commit()

connection.close()    

print("Feature vector size {}".format(np.array(global_features).shape))
print("Training labels {}".format(np.array(labels).shape))
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
scaler  = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)   
print("Target labels shape: {}".format(target.shape))
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))
h5f_data.close()
h5f_label.close()
scoring = "accuracy"
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

x_train, x_test, y_train, y_test = train_test_split(np.array(global_features), np.array(global_labels), test_size=0.25, shuffle=True)
svm = SVC(random_state=9)
kfold = KFold(n_splits=10, shuffle=True)
cv_results = cross_val_score(svm, x_train, y_train, cv=kfold, scoring=scoring)
accuracy =  cv_results.mean() * 100
print("SVM classifier accuracy:\n{}".format(accuracy))

