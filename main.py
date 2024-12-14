import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('airplaneU2.bmp')

def minimum_filter(image):
    kernel_size = 3
    return cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))

min_result = minimum_filter(image)


scale_factor = 0.5
original_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
min_resized = cv2.resize(min_result, None, fx=scale_factor, fy=scale_factor)


cv2.imshow('Original', original_resized)

cv2.imshow('Minimum Filter', min_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()