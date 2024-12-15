import cv2
import numpy as np

image = cv2.imread('img4.jpg')

def contra_harmonic_mean_filter(image, size, Q):
    image = image.astype(np.float32)
    kernel = np.ones((size, size), dtype=np.float32)
    numerator = cv2.filter2D(np.power(image, Q + 1), -1, kernel)
    denominator = cv2.filter2D(np.power(image, Q), -1, kernel)
    result = numerator / (denominator + 1e-8) 
    return np.clip(result, 0, 255).astype(np.uint8)


contra_harmonic_result = contra_harmonic_mean_filter(image, size=3, Q=1.5)


scale_factor = 2
original_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
chm_resized = cv2.resize(contra_harmonic_result, None, fx=scale_factor, fy=scale_factor)


cv2.imshow('Original', original_resized)

cv2.imshow('contra_harmonic_Filter', chm_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
