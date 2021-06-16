# USAGE
# python unsharpmask.py --dataset datasets/Cells/Q

from skimage.filters import unsharp_mask
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import cv2 as cv
import PIL
import matplotlib.cm as mplcm
from PIL import Image as im

#create an empty file in datasets/cells/ and name it whatever you like (smthng like preprocessedQ suggested for cohesion)
#Create two subfiles in this dataset, one to hold healthy preprocessed images, the other to hold unhealthy preprocessed images.

copyPath="E:\\K Cuedari\\K Cuedari\\datasets\\Cells\\preprocessedQ6\\healthy"
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of faces")
args = vars(ap.parse_args())

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    copyString = imagePath[25:len(imagePath)] #healthy path
#   copyString = imagePath[27:len(imagePath)] #unhealthy path, comment line 23 and uncomment this line if you want to preprocess, then save, unhealthy images
    pil_image = PIL.Image.open(imagePath).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    ppimg = unsharp_mask(open_cv_image, radius=5, amount=2)
    imageToBeSaved = Image.fromarray((ppimg * 255).astype(np.uint8))
    imageToBeSaved.save(copyPath+copyString)
