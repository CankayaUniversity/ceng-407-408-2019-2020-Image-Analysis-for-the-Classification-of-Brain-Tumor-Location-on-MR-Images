import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image
from PIL import Image
l = 10
a = []
for i in range(500):
    a.append([120, 10, 50])
b = []
for i in range(500):
    b.append(a)

class HPF(object):
    def __init__(self, kernel, image):
        self.kernel = np.array(kernel)
        self.image = image

    def process(self):
        return ndimage.convolve(self.image, self.kernel)

img = cv2.imread('Y1.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.imread('Y1.jpg',0)
median = cv2.medianBlur(gray,5)
cv2.imshow('blurred image', median)
cv2.waitKey(0)  
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]]) 
convoleOutput = ndimage.convolve(gray, kernel)
lowpass = ndimage.gaussian_filter(median, 3)
gauss_highpass = median - lowpass
cv2.imshow('highpass2', gauss_highpass)
cv2.waitKey(0)
enhanced = cv2.addWeighted(median,1.0,gauss_highpass,0.1,0)
cv2.imshow('highpass2', enhanced)
cv2.waitKey(0)

ret, thresh = cv2.threshold(enhanced,0,255,cv2.THRESH_OTSU)
cv2.imshow('Applying Otsu',thresh)
cv2.waitKey(0)    
colormask = np.zeros(img.shape, dtype=np.uint8)
colormask[thresh!=0] = np.array((0,0,255))
blended = cv2.addWeighted(img,0.5,colormask,0.5,0)
cv2.imshow('Blended', blended)
cv2.waitKey(0)  
ret, markers = cv2.connectedComponents(thresh)

#Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
#Get label of largest component by area
largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
#Get pixels which correspond to the brain
brain_mask = markers==largest_component

brain_out = img.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[brain_mask==False] = (0,0,0)
cv2.imshow('Connected Components',brain_out)
cv2.waitKey(0) 
vectorized = brain_out.reshape((-1,3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((brain_out.shape))
cv2.imshow('result', result_image)
cv2.waitKey(0)
