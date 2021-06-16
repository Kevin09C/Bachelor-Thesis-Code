import pip
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2 as cv
import os
import os
import PIL



pil_image = PIL.Image.open("datasets/Cells/Q4/healthy/GelHA108.5µg.1_cnt1.tif").convert('RGB')
open_cv_image = np.array(pil_image)
# Convert RGB to BGR
open_cv_image = open_cv_image[:, :, ::-1].copy()

# image = cv.imread(open_cv_image)

image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)
image = imutils.resize(image, width=28)
#  image = imutils.resize(image, width=64)
image = img_to_array(image)

print(image)
