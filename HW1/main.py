import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import imageio
from skimage import color, io

def quantize_l_channel(image, k):
    """
    Converts an RGB image to LAB, quantizes the L channel, and converts it back to RGB.

    Parameters:
    image (numpy.ndarray): Input RGB image.
    k (int): Number of clusters for k-means quantization.

    Returns:
    numpy.ndarray: RGB image with quantized L channel.
    """
    # Convert RGB to LAB
    lab_image = color.rgb2lab(image)

    # Extract L, a, b channels
    L = lab_image[:, :, 0]
    a = lab_image[:, :, 1]
    b = lab_image[:, :, 2]

    # Reshape the L channel to a 2D array of pixels
    L_reshaped = L.reshape(-1, 1)

    # Apply k-means clustering to the L channel
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(L_reshaped)

    # Replace each pixel value with its corresponding cluster center
    L_quantized = kmeans.cluster_centers_[kmeans.labels_].reshape(L.shape)

    # Combine the quantized L channel with the original a and b channels
    quantized_lab_image = np.stack((L_quantized, a, b), axis=2)

    # Convert LAB back to RGB
    quantized_rgb_image = color.lab2rgb(quantized_lab_image)

    # Convert the image from float [0, 1] to uint8 [0, 255]
    quantized_rgb_image = (quantized_rgb_image * 255).astype(np.uint8)

    return quantized_rgb_image


# Create output directory if it does not exist
output_dir2 = 'Task2-solut'
os.makedirs(output_dir2, exist_ok=True)
imges = os.listdir('./Images/Task2')   

for img in imges:
    # Read the image
    image_path = path = './Images/Task2/' + img
    image = io.imread(image_path)
    # Number of clusters
    k = 8
    quantized_image = quantize_l_channel(image, k)
    # Save the quantized image to the output directory using imageio
    save_path = os.path.join(output_dir2, f'quantized_{img}')
    imageio.imwrite(save_path, quantized_image)
    
    # Plot the original and quantized images side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Plot quantized image
    ax[1].imshow(quantized_image)
    ax[1].set_title('Quantized L channel Image ')
    ax[1].axis('off')
    
    

    


    
