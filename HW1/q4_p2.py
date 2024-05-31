import numpy as np
import matplotlib.pyplot as plt
import q4 as a

# Load focus images
focus1 = plt.imread('Images/Task4/focus1.tif')
focus2 = plt.imread('Images/Task4/focus2.tif')

# Convert images to grayscale
if focus1.ndim == 3:
    focus1 = np.mean(focus1, axis=2)
if focus2.ndim == 3:
    focus2 = np.mean(focus2, axis=2)


def focus_blend(focus1, focus2, levels):
    # Create Gaussian pyramids
    gp_focus1 = a.gaussian_pyramid(focus1, levels)
    gp_focus2 = a.gaussian_pyramid(focus2, levels)

    # Create Laplacian pyramids
    lp_focus1 = a.laplacian_pyramid(gp_focus1)
    lp_focus2 = a.laplacian_pyramid(gp_focus2)

    # Blend Laplacian pyramids
    lp_blend = []
    for l1, l2 in zip(lp_focus1, lp_focus2):
        mask = np.abs(l1) > np.abs(l2)
        blend = l1 * mask + l2 * (1 - mask)
        lp_blend.append(blend)

    # Reconstruct the final blended image
    blended_image = a.reconstruct_image(lp_blend)
    return blended_image


# Blend the images using focus blending
blended_image = focus_blend(focus1, focus2, 4)

# Displaying the results
a.display_image('Focus Image 1', focus1)
a.display_image('Focus Image 2', focus2)
a.display_image('Blended Image', blended_image)
