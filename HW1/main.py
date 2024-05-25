import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import imageio
from skimage import color, io
import cv2

def draw_histograms(image):
    """
    Draws the histogram of the L channel in two ways: with regular bins and with quantized bins.

    Parameters:
    image (numpy.ndarray): The input RGB image as a NumPy array.
    """
    # Convert the image to the Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    
    # Extract the L channel
    L_channel = lab_image[:, :, 0]
    
    # Plot histogram with regular bins
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(L_channel.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.75)
    plt.title('Histogram with Regular Bins')
    plt.xlabel('L channel intensity')
    plt.ylabel('Frequency')
    
    # Quantize the L channel
    L_flattened = L_channel.reshape(-1, 1)
    n_clusters = 10  # Number of bins for quantization
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(L_flattened)
    quantized_L = kmeans.labels_
    
    # Plot histogram with quantized bins
    plt.subplot(1, 2, 2)
    plt.hist(quantized_L, bins=n_clusters, color='gray', alpha=0.75)
    plt.title('Histogram with Quantized Bins')
    plt.xlabel('Quantized L channel intensity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

draw_histograms(io.imread('./HW1/images/Task2/colorful1.jpg'))