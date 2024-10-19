import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
import skimage
import cv2
from PIL import Image


im = cv2.imread('data/silty-thin-section.png', cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, (im.shape[1] * 2, im.shape[0] * 2))
_, thr = cv2.threshold(im, 190, 255, type = cv2.THRESH_BINARY)
plt.imshow(thr, cmap = 'Greys')
plt.show()

size_scale = 50**2 / (im.shape[0] * im.shape[1])

thresholds = np.linspace(0, 255, 100)
areas = []
angles = []

for t in thresholds:
    _, thr = cv2.threshold(im, t, 255, type = cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thr, 1, 2)

    if t > 180:
        cv2.drawContours(im, contours, -1, (0, 0, 0), 1)

    for con in contours:
        area = cv2.contourArea(con)
        center, size, angle = cv2.minAreaRect(con)
        pts = cv2.boxPoints((center, size, angle))

        if (t > 180) & (center[0] > 1500) & (center[0] < 2500) & (center[1] > 1500) & (center[1] < 2500) & (angle > 45) & (angle < 90) & (area > 1000):
            # plt.scatter(center[0], center[1], c = 'r')
            rect = patches.Rectangle((center[0] - size[0] / 2, center[1] - size[1] / 2), size[0], size[1], angle = angle, edgecolor = 'r', facecolor = 'none')
            plt.imshow(im, cmap = 'Greys')
            plt.gca().add_patch(rect)
            plt.show()
            quit()

        areas.append(area)

        if angle > 90:
            angle = 180 - angle
        elif angle < 0:
            angle = 180 + angle
        angles.append(angle)

areas = np.array(areas) * size_scale
angles = np.array(angles)

print(areas.mean(), areas.std())
plt.hist(areas, bins = 100)
plt.show()

print(angles.mean(), angles.std())
plt.hist(angles, bins = 50)
plt.show()




# thin = Image.open('data/silty-thin-section.png')
# thin = np.asarray(thin)
# thin = skimage.color.rgb2gray(thin[:, :, :3])

# seg = skimage.segmentation.chan_vese(thin)

# edges = skimage.feature.canny(thin, sigma = 0.5)
# fill_grains = scipy.ndimage.binary_fill_holes(edges)

# plt.imshow(seg, cmap = 'Greys')
# plt.show()