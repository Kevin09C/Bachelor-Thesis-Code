# import the necessary packages
from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.nn.conv.customLenet import LeNetCustom
from keras.utils import plot_model
from keras.models import model_from_json
import cv2
import PIL
import numpy as np

# pil_image = PIL.Image.open("datasets/Cells/Q4/healthy/GelHA108.5µg.1_cnt6.tif").convert('RGB')
# open_cv_image = np.array(pil_image)
# open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
# image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

# image = cv2.imread("datasets/Cells/Q2/healthy/GelPluronic.5_107.crop.1.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# print(image.shape)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# initialize LeNet and then write the network architecture
# visualization graph to disk

#model = LeNet.build(28, 28, 1, 2)
# model = LeNet.build(64, 64, 1, 2)
# model = LeNet.build(128, 128, 1, 2)
json_file = open('output_to_json/modelQ6_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
print(model.summary())
plot_model(model, to_file="modelVisualization/lenet28x28.png", show_shapes=True,show_layer_names=True)
