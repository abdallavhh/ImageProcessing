import cv2
import matplotlib.pyplot as plt
import numpy as np

def average_filter(image):
    kernel_size = 3
    return cv2.blur(image, (kernel_size, kernel_size))

def maximum_filter(image):
    kernel_size = 3
    return cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))

def minimum_filter(image):
    kernel_size = 3
    return cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))

def median_filter(image):
    kernel_size = 3
    return cv2.medianBlur(image, kernel_size)

def contra_harmonic_mean_filter(image, size, Q):
    image = image.astype(np.float32)
    kernel = np.ones((size, size), dtype=np.float32)
    numerator = cv2.filter2D(np.power(image, Q + 1), -1, kernel)
    denominator = cv2.filter2D(np.power(image, Q), -1, kernel)
    result = numerator / (denominator + 1e-8)  # Avoid division by zero
    return np.clip(result, 0, 255).astype(np.uint8)

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

def adaptive_median_filter(image, max_kernel_size):
    image = image.astype(np.float32)
    padded_image = cv2.copyMakeBorder(image, max_kernel_size, max_kernel_size, max_kernel_size, max_kernel_size, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            k = 3
            while k <= max_kernel_size:
                # Extract local window
                window = padded_image[i:i+k, j:j+k]
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = padded_image[i + max_kernel_size, j + max_kernel_size]

                if z_med > z_min and z_med < z_max:
                    if z_xy > z_min and z_xy < z_max:
                        result[i, j] = z_xy
                    else:
                        result[i, j] = z_med
                    break
                k += 2

            if k > max_kernel_size:
                result[i, j] = z_med
    return result.astype(np.uint8)

# Load an image (ensure it's in grayscale)
image_path = 'airplane.bmp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply filters
avg_result = average_filter(image)
max_result = maximum_filter(image)
min_result = minimum_filter(image)
med_result = median_filter(image)
contra_harmonic_result = contra_harmonic_mean_filter(image, size=3, Q=1.5)
alpha_trimmed_result = alpha_trimmed_mean_filter(image, kernel_size=5, alpha=4)
adaptive_median_result = adaptive_median_filter(image, max_kernel_size=7)

# Save results
cv2.imwrite('average_filter_result.jpg', avg_result)
cv2.imwrite('maximum_filter_result.jpg', max_result)
cv2.imwrite('minimum_filter_result.jpg', min_result)
cv2.imwrite('median_filter_result.jpg', med_result)
cv2.imwrite('contra_harmonic_filter_result.jpg', contra_harmonic_result)
cv2.imwrite('alpha_trimmed_filter_result.jpg', alpha_trimmed_result)
cv2.imwrite('adaptive_median_filter_result.jpg', adaptive_median_result)

# Display results (optional)
cv2.imshow('Average Filter', avg_result)
cv2.imshow('Maximum Filter', max_result)
cv2.imshow('Minimum Filter', min_result)
cv2.imshow('Median Filter', med_result)
cv2.imshow('Contra Harmonic Mean Filter', contra_harmonic_result)
cv2.imshow('Alpha Trimmed Mean Filter', alpha_trimmed_result)
cv2.imshow('Adaptive Median Filter', adaptive_median_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Load the image
image = cv2.imread('noisy_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Save the result
cv2.imwrite('gaussian_blur_result.jpg', gaussian_blur)
