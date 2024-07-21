import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor



# List of directories containing query images and template images
dirs = ['Roadsign','Starbucks' ,'Superman']  # Update this list with your actual directory names

DB = {} 
for dir in dirs:
    DB[dir] = {
                                  "images" :[f'Q4/{dir}/{img}'for img in  os.listdir(f'Q4/{dir}/') if not img.startswith('template')],
                                  "template":[f'Q4/{dir}/{img}' for img in  os.listdir(f'Q4/{dir}/') if img.startswith('template')] 
                                  }



# Function to resize the image to a specified width while maintaining aspect ratio
def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image
    h, w = image.shape[:2]
    if width is not None:
        ratio = width / float(w)
        dimension = (width, int(h * ratio))
    else:
        ratio = height / float(h)
        dimension = (int(w * ratio), height)
    return cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

# Function to adjust hue, saturation, and brightness
def adjust_hue_saturation_brightness(image, hue_shift=0, saturation_scale=1, brightness_scale=1):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h = cv2.add(h, hue_shift)
    s = cv2.multiply(s, saturation_scale)
    v = cv2.multiply(v, brightness_scale)
    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Function to flip the image horizontally or vertically
def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

# Function to apply Gaussian blur to the image
def blur_image(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function to apply perspective transformation
def perspective_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[0, 0], [w, 0], [int(0.33 * w), h], [int(0.66 * w), h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h))

# Function to perform template matching and return the best match
def match_templates(target_img, templates, scales, color_transformations, additional_transformations):
    best_match = None
    best_match_count = 0
    best_transform = None
    best_template = None
    best_match_distance = float('inf')

    sift = cv2.SIFT_create()
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    keypoints_target, descriptors_target = sift.detectAndCompute(target_img_gray, None)
    if descriptors_target is None:
        return target_img

    for template in templates:
        for hue_shift, saturation_scale, brightness_scale in color_transformations:
            transformed_template = adjust_hue_saturation_brightness(template, hue_shift, saturation_scale, brightness_scale)
            for scale in scales:
                scaled_template = resize_image(transformed_template, width=int(template.shape[1] * scale))
                for flip_code in additional_transformations['flips']:
                    flipped_template = flip_image(scaled_template, flip_code)
                    for kernel_size in additional_transformations['blurs']:
                        blurred_template = blur_image(flipped_template, kernel_size)
                        for apply_perspective in additional_transformations['perspectives']:
                            if apply_perspective:
                                final_template = perspective_transform(blurred_template)
                            else:
                                final_template = blurred_template

                            template_gray = cv2.cvtColor(final_template, cv2.COLOR_BGR2GRAY)
                            keypoints_template, descriptors_template = sift.detectAndCompute(template_gray, None)

                            if descriptors_template is None:
                                continue

                            # Initialize the FLANN based matcher
                            FLANN_INDEX_KDTREE = 1
                            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                            search_params = dict(checks=50)
                            flann = cv2.FlannBasedMatcher(index_params, search_params)

                            matches = flann.knnMatch(descriptors_template, descriptors_target, k=2)
                            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

                            if len(good_matches) >= 4:  # Ensure we have at least 4 good matches
                                src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                                if M is not None:
                                    num_inliers = np.sum(mask)  # Number of inliers
                                    avg_match_distance = np.mean([m.distance for m in good_matches])  # Average distance

                                    # Choose the best match based on the number of inliers and average match distance
                                    if (num_inliers > best_match_count) or (num_inliers == best_match_count and avg_match_distance < best_match_distance):
                                        best_match_count = num_inliers
                                        best_match_distance = avg_match_distance
                                        best_transform = M
                                        best_template = final_template

    if best_match_count > 0 and best_transform is not None:
        h, w = best_template.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, best_transform)
        target_img = cv2.polylines(target_img, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    return target_img

# Load templates from the directory
template_paths = DB['Superman']['template']  # List of paths to template images
templates = [cv2.imread(path) for path in template_paths if cv2.imread(path) is not None]

# Initialize scales, rotations, and color transformations
scales = [0.5, 0.2, 0.3, 0.4, 0.6, 1.0]  # Adjust scales as needed
color_transformations = [(0, 1.0, 1.0), (10, 0.9, 1.1), (-10, 1.1, 0.9), (20, 0.8, 1.2)]  # Adjust color transformations as needed
additional_transformations = {
    'flips': [0, 1, -1],  # 0: horizontal flip, 1: vertical flip, -1: both horizontal and vertical flip
    'blurs': [(5, 5), (7, 7)],  # Kernel sizes for Gaussian blur
    'perspectives': [True, False]  # Whether to apply perspective transformation
}

# Loop over the images in the dataset
for image_path in DB['Superman']['images']:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found at {image_path}")
        continue

    result_img = match_templates(img, templates, scales, color_transformations, additional_transformations)

    # Display the results
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Matches in {image_path}")
    plt.axis('off')
    plt.show()
