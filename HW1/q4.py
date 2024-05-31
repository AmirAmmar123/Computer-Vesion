import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma=1):
    """Generates a 2D Gaussian kernel."""
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(kernel_1d, kernel_1d)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def convolve(image, kernel):
    """Convolves an image with a given kernel."""
    k = kernel.shape[0] // 2
    padded_image = np.pad(image, k, mode='edge')
    convolved_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            convolved_image[i, j] = np.sum(region * kernel)
    return convolved_image


def downsample(image):
    """Downsamples the image by a factor of 2."""
    return image[::2, ::2]


def upsample(image):
    """Upsamples the image by a factor of 2."""
    upsampled = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=image.dtype)
    upsampled[::2, ::2] = image
    kernel = gaussian_kernel(5, sigma=1)
    return convolve(upsampled, kernel)


def gaussian_pyramid(image, levels):
    gp = [image]
    for i in range(1, levels):
        image = downsample(convolve(image, gaussian_kernel(5, sigma=1)))
        gp.append(image)
    return gp


def laplacian_pyramid(gp):
    lp = []
    for i in range(len(gp) - 1):
        GE = upsample(gp[i + 1])
        if GE.shape != gp[i].shape:
            GE = GE[:gp[i].shape[0], :gp[i].shape[1]]
        L = gp[i] - GE
        lp.append(L)
    lp.append(gp[-1])
    return lp


def reconstruct_image(lp):
    image = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        image = upsample(image)
        if image.shape != lp[i].shape:
            image = image[:lp[i].shape[0], :lp[i].shape[1]]
        image = image + lp[i]
    return image


# Helper function to display images in subplots
def display_images(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    for img, title, ax in zip(images, titles, axes):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Loading images
    mandril = plt.imread('Images/Task4/mandril.tif')
    toucan = plt.imread('Images/Task4/toucan.tif')

    # Convert images to grayscale
    if mandril.ndim == 3:
        mandril = np.mean(mandril, axis=2)
    if toucan.ndim == 3:
        toucan = np.mean(toucan, axis=2)

    # Generating Gaussian and Laplacian Pyramids
    levels = 5
    gp_mandril = gaussian_pyramid(mandril, levels)
    lp_mandril = laplacian_pyramid(gp_mandril)

    gp_toucan = gaussian_pyramid(toucan, levels)
    lp_toucan = laplacian_pyramid(gp_toucan)

    # Reconstructing the original images
    reconstructed_mandril = reconstruct_image(lp_mandril)
    reconstructed_toucan = reconstruct_image(lp_toucan)

    # Collecting images and titles for display
    images = [mandril, reconstructed_mandril] + gp_mandril + lp_mandril + [toucan, reconstructed_toucan] + gp_toucan + lp_toucan
    titles = ['Original Mandril', 'Reconstructed Mandril'] + [f'Mandril Gaussian Level {i}' for i in range(levels)] + \
             [f'Mandril Laplacian Level {i}' for i in range(levels)] + \
             ['Original Toucan', 'Reconstructed Toucan'] + [f'Toucan Gaussian Level {i}' for i in range(levels)] + \
             [f'Toucan Laplacian Level {i}' for i in range(levels)]

    # Displaying the images
    display_images(images, titles, 6, 5)

    # Checking if the original and reconstructed images are identical
    print('Mandril Image Reconstruction Difference:', np.sum(np.abs(mandril - reconstructed_mandril)))
    print('Toucan Image Reconstruction Difference:', np.sum(np.abs(toucan - reconstructed_toucan)))
