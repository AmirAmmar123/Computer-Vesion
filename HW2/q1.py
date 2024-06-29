import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(images, titles=None):
    for i, img in enumerate(images):
        plt.figure(figsize=(10, 10))
        if len(img.shape) == 2:  # Grayscale image
            plt.imshow(img, cmap='gray')
        else:  # Color image
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.show()

# Load the images
image1 = cv2.imread('Q2/Working set/I1.jpg')
image2 = cv2.imread('Q2/Working set/I2.jpg')
image3 = cv2.imread('Q2/Working set/I3.jpg')

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

display_images([gray1, gray2, gray3], ['Image 1', 'Image 2', 'Image 3'])

# Step 2: Detection
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
keypoints3, descriptors3 = sift.detectAndCompute(gray3, None)

# Draw keypoints
img1_with_kp = cv2.drawKeypoints(image1, keypoints1, None)
img2_with_kp = cv2.drawKeypoints(image2, keypoints2, None)

display_images([img1_with_kp, img2_with_kp], ['Image 1 Keypoints', 'Image 2 Keypoints'])

# Step 3: Matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
plt.title('Matches')
plt.show()

# Step 4: Homography
if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    h, w = gray1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    img2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Homography')
    plt.show()
else:
    print("Not enough matches found")

# Step 5: Mapping
result = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))
result[0:image2.shape[0], 0:image2.shape[1]] = image2

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Warped Image')
plt.show()

# Step 6: Merging I
def create_panorama(img1, img2, H):
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    return result

panorama = create_panorama(image1, image2, H)
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title('Panorama')
plt.show()

# Step 8: Generalizing I (Add the third image)
keypoints1_3, descriptors1_3 = sift.detectAndCompute(result, None)
matches3 = bf.knnMatch(descriptors1_3, descriptors3, k=2)

good_matches3 = []
for m, n in matches3:
    if m.distance < 0.75 * n.distance:
        good_matches3.append(m)

if len(good_matches3) > 10:
    src_pts3 = np.float32([keypoints1_3[m.queryIdx].pt for m in good_matches3]).reshape(-1, 1, 2)
    dst_pts3 = np.float32([keypoints3[m.trainIdx].pt for m in good_matches3]).reshape(-1, 1, 2)

    H3, mask3 = cv2.findHomography(src_pts3, dst_pts3, cv2.RANSAC, 5.0)
    matches_mask3 = mask3.ravel().tolist()

    panorama3 = create_panorama(result, image3, H3)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(panorama3, cv2.COLOR_BGR2RGB))
    plt.title('Panorama with Three Images')
    plt.show()
else:
    print("Not enough matches found for the third image")
