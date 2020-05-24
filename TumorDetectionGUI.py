#!/usr/bin/env python
# coding: utf-8

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QFileDialog, QLabel, QTextEdit
import sys
from PyQt5.QtGui import QPixmap, QFont
from scipy import ndimage
import cv2 
import numpy as np
import glob
import os
import TumorDetection

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
        self.initWindow()
    
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

    def initWindow(self):
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
        image = TumorDetection.readImage(self.imagePath) 
        original = self.getQImage(image)
        self.originalImage.setPixmap(QPixmap(QPixmap.fromImage(original)))
        
    def getResults(self):
        if self.imagePath == "":
            return
        
        image = TumorDetection.readImage(self.imagePath) 
        preprocessed = TumorDetection.preprocessImage(image)
        segmented = TumorDetection.segmentImage(preprocessed)
        segmentedImage = self.getQImage(segmented)
        imageToPixmap = QPixmap(QPixmap.fromImage(segmentedImage))
        self.segmentedImage.setPixmap(QPixmap(imageToPixmap))
        postprocessed = TumorDetection.postprocessImage(segmented)
        contour = TumorDetection.drawContour(segmented, postprocessed)
        contourImage = self.getQImage(contour)
        imageToPixmap = QPixmap(QPixmap.fromImage(contourImage))
        self.contourImage.setPixmap(QPixmap(imageToPixmap))
        result = TumorDetection.findTumor(postprocessed)
        TumorDetection.createFeatureVector(image)

        fontStyle = QFont("Arial", 10, QFont.Bold)
        self.tumorInfo.setFont(fontStyle)
        if result[0] == "No":
            tumorInfoStr = "Tumor Status: " + result[0] 
            self.tumorInfo.setStyleSheet("QLabel { color : #39ff14; }");
            self.tumorInfo.setText(tumorInfoStr)
        else:
            self.contourImage.setPixmap(QPixmap(imageToPixmap))
            tumorInfoStr = "Tumor Status: " + result[0] + "\nTumor Location: " + result[1]
            self.tumorInfo.setStyleSheet("QLabel { color : #ff073a; }");
            self.tumorInfo.setText(tumorInfoStr)

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())

