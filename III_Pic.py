import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'img3.bmp'  
image = cv2.imread(image_path)


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply histogram equalization
equalized_img = cv2.equalizeHist(gray)

# Display the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.show()
