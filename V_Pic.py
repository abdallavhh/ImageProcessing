import cv2
import numpy as np

image = cv2.imread('img5.jpg')

def alpha_trimmed_mean_filter(image, kernel_size, alpha):
    pad = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)
    alpha_trim = alpha // 2

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract local window
            window = padded_image[i:i+kernel_size, j:j+kernel_size].flatten()
            # Sort and remove alpha/2 smallest and largest values
            trimmed_window = np.sort(window)[alpha_trim:-alpha_trim]
            # Calculate the mean of the trimmed window
            result[i, j] = np.mean(trimmed_window)
    return result.astype(np.uint8)


alpha_trimmed_result = alpha_trimmed_mean_filter(image, kernel_size=5, alpha=4)


scale_factor = .9
original_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
al_resized = cv2.resize(alpha_trimmed_result , None, fx=scale_factor, fy=scale_factor)


cv2.imshow('Original', original_resized)

cv2.imshow('alpha_trimmed_Filter', al_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
