# USAGE
# python RunCustomLeNetModel.py --dataset datasets/Cells/Q
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from modelConstruction.nn.conv.lenet import LeNet
from modelConstruction.nn.conv.customLenet import LeNetCustom
from imutils import paths
import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
import argparse
import imutils
import cv2 as cv
import os
import PIL
from keras import backend as K 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of faces")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []
a = 0
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # Read in tiff
    pil_image = PIL.Image.open(imagePath).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=128)
    image = img_to_array(image)
    data.append(image)
    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2]
    # label = "smiling" if label == "positives" else "not_smiling"
    label = "healthy" if label == "healthy" else "unhealthy"
    labels.append(label)
    a += 1

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

trainX = data
trainY = labels
# Load trained CNN model
#json_file = open('output_to_json/modelQ4_128x128_customLenet.json', 'r')
#json_file = open('output_to_json/modelQ6_128x128_customLenet.json', 'r')
json_file = open('output_to_json/modelQ6_2.json', 'r') # June 26th 2020

loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#model.load_weights('output/lenetQ4_128x128_customLenet.hdf5')
#model.load_weights('output/lenetQ6_128x128_customLenet.hdf5')
model.load_weights('output/lenet6.hdf5')

trainLabels = list(le.inverse_transform(trainY.argmax(1)))
size = len(trainLabels)
predicted = 0
images = []
x = 0
for i in np.random.choice(np.arange(0, len(trainY)), size=(size,)):

    probs = model.predict(trainX[np.newaxis, i])
    # print(probs)
    prediction = probs.argmax(axis=1)
    label = le.inverse_transform(prediction)
    if label[0] == trainLabels[i]:
        predicted += 1

    # extract the image from the testData if using "channels_first"
    # ordering
    if K.image_data_format() == "channels_first":
        image = (trainX[i][0] * 255).astype("uint8")

    # otherwise we are using "channels_last" ordering
    else:
        image = (trainX[i] * 255).astype("uint8")

    # merge the channels into one image
    image = cv.merge([image] * 3)

    image = cv.resize(image, (128, 128), interpolation=cv.INTER_LINEAR)

    # show the image and prediction
    x += 1
    position = str(x)
    text = position + ' ' + label[0]
    cv.putText(image, str(text), (5, 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    print("[INFO]:{} Predicted: {}, Actual: {}".format(x, label[0],
                                                       trainLabels[i]))
    images.append(image)

print('Accuracy: ',
      predicted / size)

fig = plt.figure(figsize=(14, 14))
columns = 8
rows = 3
for i in range(0, columns * rows):
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(images[i])
plt.show()
