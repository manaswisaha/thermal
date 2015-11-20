import sys
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

"""
OpenCV Samples e.g.: /usr/local/share/OpenCV/samples/python2/watershed.py
"""


def watershed(img, gray):
    """
    Using Watershed image segmentation algorithm
    """
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    return img, markers


def plot_results(in_img, out_img, filepath, algo_name):

    if algo_name == "canny":
        title = 'Edge Image'
    elif algo_name == "watershed":
        title = 'Segmented Image'
    plt.subplot(121), plt.imshow(in_img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(out_img, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.savefig(filepath)
    # plt.show()

# """
for filepath in glob.glob("images/*.JPG"):

    filename = filepath.split("/")[1]
    filename = filename.replace(".JPG", ".jpg")
    print filename

    img = cv2.imread(filepath)
    # gray = cv2.imread(filepath, 0)
    # Get a gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    threshold1 = 100
    threshold2 = 200
    edges = cv2.Canny(gray, threshold1, threshold2)
    plot_results(img, edges, "output/cannyedge/" + filename, "canny")

    # Watershed image segmentation
    s_img, markers = watershed(img, gray)
    plot_results(gray, s_img, "output/watershed/" + filename, "watershed")
    # cv2.imwrite("output/watershed/markers_" + filename, markers)


sys.exit(0)
# """
print cv2.__version__

# Input file
filepath = "images/IMG_2655.JPG"
filename = filepath.split("/")[1]
filename = filename.replace(".JPG", ".jpg")

# Get input image in gray scale
img = cv2.imread(filepath)
# gray = cv2.imread(filepath, 0)
# Get a gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# TODO: Do some image transformation to remove the edges to just focus on
# the temperature gradients


"""
Sift features
"""
sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp)

cv2.imwrite('sift_keypoints_2655_temp1.jpg', img)
cv2.imshow('test image', img)

# b, g, r = cv2.split(img)
# img2 = cv2.merge((r, g, b))
# plt.imshow(img2, interpolation='bicubic')
# plt.show()
