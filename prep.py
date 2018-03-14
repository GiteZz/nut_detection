import cv2
import numpy as np
from matplotlib import pyplot as plt
import bolt_const
import math
import logging
import time




img = cv2.imread('canny/1.png',0)
print(len(img))
img_height = len(img)
img_width = len(img[0])

x_max = [0]*img_height
x_min = [img_width]*img_height

y_max = [0]*img_width
y_min = [img_height]*img_width
print(y_max)
plot_elem = []

im = plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.ion()
stop_loop = False
def prep():
    for y in range(0, img_height):
        for x in range(0, img_width):
            if img[y][x]:
                if x_max[y] < x:
                    x_max[y] = x
                if x_min[y] > x:
                    x_min[y] = x

                if y_max[x] < y:
                    y_max[x] = y
                if y_min[x] > y:
                    y_min[x] = y

    for y in range(0, img_height):
        for x in range(0, img_width):
            if img[y][x]:
                if x_min[y] < x < x_max[y]:
                    img[y][x] = 0
                if y_min[x] < y < y_max[x]:
                    img[y][x] = 0
                im.set_data(img)



plt.pause(5000)

plt.show()