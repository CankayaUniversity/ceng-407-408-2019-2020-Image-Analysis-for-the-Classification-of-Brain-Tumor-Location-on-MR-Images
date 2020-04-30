from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QFileDialog, QLabel, QTextEdit
import sys
from PyQt5.QtGui import QPixmap, QFont
from scipy import ndimage
import cv2 
import numpy as np
import glob
import os

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Brain Tumor Detection App"
        self.top = 200
        self.left = 200
        self.width = 800
        self.height = 400
        self.imagePath = ""
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.InitWindow()
    
    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox("")
        layout = QGridLayout()
        layout.addWidget(self.originalImage,1,0)
        layout.addWidget(self.segmentedImage,1,1)
        layout.addWidget(self.contourImage,1,2)
        layout.addWidget(self.chooseFile,0,0)
        layout.addWidget(self.resultsButton,0,1)
        layout.addWidget(self.tumorInfo,2,0)
        
        self.horizontalGroupBox.setLayout(layout)

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.chooseFile = QPushButton("Open Image")
        self.chooseFile.clicked.connect(self.getImage)
        self.resultsButton = QPushButton("Results")
        self.resultsButton.clicked.connect(self.getResults)
        self.originalImage = QLabel()
        self.segmentedImage = QLabel()
        self.contourImage = QLabel()
        self.tumorInfo = QLabel()
        self.createGridLayout()
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)
        self.show()

    def getQImage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image
    
    def getImage(self):
        self.originalImage.clear()
        self.segmentedImage.clear()
        self.contourImage.clear()
        self.tumorInfo.clear()
        fileName = QFileDialog.getOpenFileName(self, "Open file", "Image files")
        self.imagePath = fileName[0]
        image = cv2.imread(self.imagePath) 
        resizedImage = cv2.resize(image, (400, 400))
        original = self.getQImage(resizedImage)
        self.originalImage.setPixmap(QPixmap(QPixmap.fromImage(original)))
        
    def getResults(self):
        if self.imagePath == "":
            return
        tumorStatus = ""
        tumorLocation = ""
        image = cv2.imread(self.imagePath) 
        resizedImage = cv2.resize(image, (400, 400))
        gray = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        ret, markers = cv2.connectedComponents(thresh)
        marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
        largest_component = np.argmax(marker_area)+1                    
        brain_mask = markers==largest_component
        brain_out = resizedImage.copy()
        brain_out[brain_mask==False] = (0,0,0)
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
        segmentedImage = self.getQImage(segmented)
        imageToPixmap = QPixmap(QPixmap.fromImage(segmentedImage))
        self.segmentedImage.setPixmap(QPixmap(imageToPixmap))
        
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
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        x, y, w, h = 0, 0, erode.shape[1]//2, erode.shape[0]
        left = erode[y:y+h, x:x+w]
        right = erode[y:y+h, x+w:x+w+w]
        left_pixels = cv2.countNonZero(left)
        right_pixels = cv2.countNonZero(right)
        ratio = -1       
        gray = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            area = cv2.contourArea(contour)
            cv2.drawContours(segmented, contours, -1, (0, 0, 255), 3)
            contourImage = self.getQImage(segmented)
            imageToPixmap = QPixmap(QPixmap.fromImage(contourImage))
            if area > 0:
                if left_pixels == 0 and right_pixels > 0:
                    tumorStatus = "Yes"
                    tumorLocation = "Right"
                elif right_pixels == 0 and left_pixels > 0:
                    tumorStatus = "Yes"
                    tumorLocation = "Left"
                elif right_pixels > left_pixels:
                    ratio = float(right_pixels/left_pixels)
                    if ratio >= 1.5:
                        tumorStatus = "Yes"
                        tumorLocation = "Right"
                    else:
                        tumorStatus = "No"
                else:
                    ratio = float(left_pixels/right_pixels)
                    if ratio >= 1.5:
                        tumorStatus = "Yes"
                        tumorLocation = "Left"
                    else:
                        tumorStatus = "No"
            else:
                tumorStatus = "No"
        else:
            tumorStatus = "No"
        
        fontStyle = QFont("Arial", 10, QFont.Bold)
        self.tumorInfo.setFont(fontStyle)
        if tumorStatus == "No":
            tumorInfoStr = "Tumor Status: " + tumorStatus 
            self.tumorInfo.setStyleSheet("QLabel { color : #39ff14; }");
            self.tumorInfo.setText(tumorInfoStr)
        else:
            self.contourImage.setPixmap(QPixmap(imageToPixmap))
            tumorInfoStr = "Tumor Status: " + tumorStatus + "\nTumor Location: " + tumorLocation
            self.tumorInfo.setStyleSheet("QLabel { color : #ff073a; }");
            self.tumorInfo.setText(tumorInfoStr)

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())

