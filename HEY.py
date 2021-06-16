from skimage import data
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import cv2 as cv
import PIL

imagePath="E:\\K Cuedari\\K Cuedari\\datasets\\Cells\\testingPurpose\\GelHA108.5µg.1_cnt1.tif"
pil_image = PIL.Image.open(imagePath).convert('RGB')
open_cv_image = np.array(pil_image)
open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
#image = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)
image = open_cv_image

result_1 = unsharp_mask(image, radius=1, amount=1)
result_2 = unsharp_mask(image, radius=5, amount=2)
result_3 = unsharp_mask(image, radius=20, amount=1)

fig, axes = plt.subplots(nrows=2, ncols=2,
                         sharex=True, sharey=True, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title('Original image')
ax[1].imshow(result_1 )
ax[1].set_title('Enhanced image, radius=1, amount=1.0')
ax[2].imshow(result_2 )
ax[2].set_title('Enhanced image, radius=5, amount=2.0')
ax[3].imshow(result_3, cmap=plt.cm.gray)
ax[3].set_title('Enhanced image, radius=20, amount=1.0')

for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show()
