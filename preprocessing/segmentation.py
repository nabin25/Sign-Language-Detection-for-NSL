import cv2
import numpy as np

# Function to segment hand from background using color-based segmentation
def segment_hand_grabcut(hand_image):
    mask = np.zeros(hand_image.shape[:2], np.uint8)
    rect = (50, 50, hand_image.shape[1]-50, hand_image.shape[0]-50)  # Adjust the rectangle as needed

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(hand_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to create a binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Bitwise AND to get the segmented hand
    segmented_hand = cv2.bitwise_and(hand_image, hand_image, mask=mask2)

    return segmented_hand

# Path to the input image
input_image_path = "E:/major-project/new/0/hand_0.png"

# Read the input image
original_image = cv2.imread(input_image_path)

# Display the original image
cv2.imshow("Original Image", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Segment the hand from the background using color-based segmentation
segmented_hand = segment_hand_grabcut(original_image)

# Display the segmented hand image
cv2.imshow("Segmented Hand", segmented_hand)
cv2.waitKey(0)
cv2.destroyAllWindows()
