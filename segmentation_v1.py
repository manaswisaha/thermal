import cv2
from matplotlib import pyplot as plt

filename = "images/IMG_1180.JPG"

img = cv2.imread(filename, 0)
# cv2.imshow('test image', img)

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
