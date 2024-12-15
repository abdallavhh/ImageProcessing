import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Load the image
image_path = 'img6.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply adaptive median filter
max_kernel_size = 7
filtered_image = adaptive_median_filter(image, max_kernel_size)

# Display the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Adaptive Median Filtered Image')
plt.axis('off')

plt.show()
