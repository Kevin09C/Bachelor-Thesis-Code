# USAGE
# python train_model.py --dataset datasets/Cells/Q --model output/lenet.hdf5 --model_json output_to_json/modelQ.json
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.nn.conv.customLenet import LeNetCustom
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2 as cv
import os
import PIL
import xlsxwriter


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-mj", "--model_json", required=True,
                help="path to output model to json")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

# loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # load the image, pre-process it, and store it in the data list

    # Read PNG
    # image = cv.imread(imagePath)
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Read in tiff
    pil_image = PIL.Image.open(imagePath).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)

    # image = imutils.resize(image, width=28)
    # image = imutils.resize(image, width=64)
    image = imutils.resize(image, width=128)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    # label = "smiling" if label == "positives" else "not_smiling"
    label = "healthy" if label == "healthy" else "unhealthy"
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.20, stratify=labels, random_state=42)

# initialize the model
print("[INFO] compiling model...")
# model = LeNet.build(width=28, height=28, depth=1, classes=2)
# model = LeNet.build(width=64, height=64, depth=1, classes=2)

model = LeNetCustom.build(width=128, height=128, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              class_weight=classWeight, batch_size=64, epochs=10, verbose=1)

# history = model.fit()

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

model_json = model.to_json()
with open(args["model_json"], 'w') as json_file:
    json_file.write(model_json)

#####

workbook = xlsxwriter.Workbook('Q6_v1.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "Accuracy")
worksheet.write(0, 1, "Val_accuracy")
worksheet.write(0, 2, "Loss")
worksheet.write(0, 3, "Val_loss")
row = 1
col = 0

for item in H.history['accuracy']:
    worksheet.write(row, col, item)
    row += 1
row = 1
col = 1
for item in H.history['val_accuracy']:
    worksheet.write(row, col, item)
    row += 1
row = 1
col = 2
for item in H.history['loss']:
    worksheet.write(row, col, item)
    row += 1
row = 1
col = 3
for item in H.history['val_loss']:
    worksheet.write(row, col, item)
    row += 1

workbook.close()

#######

# plot the training + testing loss and accuracy
# plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 10), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 10), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 10), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, 10), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

##########
# plt.plot(hist.history['accuracy'], label='train')
# plt.plot(hist.history['val_accuracy'], label='test')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()

# plt.plot(hist.history['loss'], label='train')
# plt.plot(hist.history['val_loss'], label='test')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.legend(['Train', 'Val'], loc='upper right')
# plt.show()
