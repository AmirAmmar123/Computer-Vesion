import numpy as np
from skimage.util import img_as_float
from skimage import io, color
import matplotlib.pyplot as plt
import copy

class SuperPixel:
    def __init__(self, l=0, a=0, b=0, h=0, w=0):
        self.update(l, a, b, h, w)
        self.pixels = []

    def update(self, l, a, b, h, w):
        self.l = l
        self.a = a
        self.b = b
        self.h = h
        self.w = w

def make_SuperPixel(h, w, img):
    return SuperPixel(img[h, w][0], img[h, w][1], img[h, w][2], h, w)

def show_image(img, title=''):
    plt.figure()
    plt.title(title)
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')
    plt.show()

def display_clusters(img, clusters, title):
    image = np.copy(img)
    for c in clusters:
        for p in c.pixels:
            image[p[0], p[1]][0] = c.l
            image[p[0], p[1]][1] = c.a
            image[p[0], p[1]][2] = c.b
        image[c.h, c.w][0] = 0
        image[c.h, c.w][1] = 0
        image[c.h, c.w][2] = 0
    rgb_arr = color.lab2rgb(image)
    show_image(rgb_arr, title)

def initialize_cluster_centers(S, image, img_h, img_w, clusters):
    for h in range(int(S/2), img_h, S):
        for w in range(int(S/2), img_w, S):
            clusters.append(make_SuperPixel(h, w, image))
    return clusters

def get_gradient(h, w, image):
    if w + 1 >= image.shape[1]:
        w = image.shape[1] - 2
    if h + 1 >= image.shape[0]:
        h = image.shape[0] - 2
    gradientx = np.sqrt((image[h][w+1][0] - image[h][w-1][0])**2 + (image[h][w+1][1] - image[h][w-1][1])**2 + (image[h][w+1][2] - image[h][w-1][2])**2)
    gradienty = np.sqrt((image[h+1][w][0] - image[h-1][w][0])**2 + (image[h+1][w][1] - image[h-1][w][1])**2 + (image[h+1][w][2] - image[h-1][w][2])**2)
    gradient = gradientx + gradienty
    return gradient

def relocate_cluster_center_at_lowgrad(clusters, image):
    for c in clusters:
        gradient = get_gradient(c.h, c.w, image)
        for h in range(-1, 2):
            for w in range(-1, 2):
                x = c.h + h
                y = c.w + w
                new_gradient = get_gradient(x, y, image)
                if new_gradient < gradient:
                    c.update(image[x][y][0], image[x][y][1], image[x][y][2], x, y)
                    gradient = new_gradient
    return None

def assign_cluster(clusters, S, image, img_h, img_w, cluster_tag, dis, M, return_distances=False):
    distances = np.zeros((img_h, img_w))
    for c in clusters:
        for h in range(c.h - 2 * S, c.h + 2 * S):
            if h < 0 or h >= img_h:
                continue
            for w in range(c.w - 2 * S, c.w + 2 * S):
                if w < 0 or w >= img_w:
                    continue
                l, a, b = image[h, w]
                Dc = np.sqrt((l - c.l)**2 + (a - c.a)**2 + (b - c.b)**2)
                Ds = np.sqrt((h - c.h)**2 + (w - c.w)**2)
                D = np.sqrt((Dc / M)**2 + (Ds / S)**2)
                if D < dis[h, w]:
                    if (h, w) not in cluster_tag:
                        cluster_tag[(h, w)] = c
                        c.pixels.append((h, w))
                    else:
                        cluster_tag[(h, w)].pixels.remove((h, w))
                        cluster_tag[(h, w)] = c
                        c.pixels.append((h, w))
                    dis[h, w] = D
                if return_distances:
                    distances[h, w] = D
    return distances if return_distances else None

def update_clusters(clusters, image):
    for c in clusters:
        sum_h, sum_w = 0, 0
        for pixel in c.pixels:
            sum_h += pixel[0]
            sum_w += pixel[1]
        mean_h = sum_h // len(c.pixels)
        mean_w = sum_w // len(c.pixels)
        c.update(image[mean_h, mean_w][0], image[mean_h, mean_w][1], image[mean_h, mean_w][2], mean_h, mean_w)
    return None

def compute_res_error(old_clusters, new_clusters):
    error = 0.0
    for new_c, old_c in zip(new_clusters, old_clusters):
        error_lab = np.abs(new_c.l - old_c.l) + np.abs(new_c.a - old_c.a) + np.abs(new_c.b - old_c.b)
        error_hw = np.abs(new_c.h - old_c.h) + np.abs(new_c.w - old_c.w)
        error += error_lab + error_hw
    return error

def display_heatmap(data, title=''):
    plt.figure()
    plt.title(title)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.show()

def superpixel_segmentation(image_path, S, M, weight_combinations, num_iterations=10, error_threshold=0.1):
    img = img_as_float(io.imread(image_path))
    img_lab = color.rgb2lab(img)
    img_h, img_w = img_lab.shape[:2]

    for i, (S_weight, M_weight) in enumerate(weight_combinations):
        clusters = []
        clusters = initialize_cluster_centers(S, img_lab, img_h, img_w, clusters)
        relocate_cluster_center_at_lowgrad(clusters, img_lab)

        dis = np.full((img_h, img_w), np.inf)
        cluster_tag = {}

        # Initial distances
        initial_distances = assign_cluster(clusters, S * S_weight, img_lab, img_h, img_w, cluster_tag, dis, M * M_weight, return_distances=True)
        display_heatmap(initial_distances, f'Initial Distances: S_weight={S_weight}, M_weight={M_weight}')

        for iteration in range(num_iterations):
            old_clusters = copy.deepcopy(clusters)
            assign_cluster(clusters, S * S_weight, img_lab, img_h, img_w, cluster_tag, dis, M * M_weight)
            update_clusters(clusters, img_lab)
            error = compute_res_error(old_clusters, clusters)
            if error < error_threshold:
                break

        display_clusters(img_lab, clusters, f'Weight Combination {i+1}: S_weight={S_weight}, M_weight={M_weight}')

        # Final distances
        final_distances = assign_cluster(clusters, S * S_weight, img_lab, img_h, img_w, cluster_tag, dis, M * M_weight, return_distances=True)
        display_heatmap(final_distances, f'Final Distances: S_weight={S_weight}, M_weight={M_weight}')

# Example usage
image_path = '/home/ameer/Computer-Vesion/HW2/Q1/castle.jpg'  # Replace with your image path
S = 10  # Example value for S
M = 10  # Example value for M
# Define different weights for the color values and the spatial coordinates
weight_combinations = [(1, 1), (1, 0.5), (0.5, 1)]

superpixel_segmentation(image_path, S, M, weight_combinations)
