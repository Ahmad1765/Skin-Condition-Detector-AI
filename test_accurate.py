import cv2
import numpy as np
import skin_score
import urllib.request
import os

img_path = r'd:\Skin Condition\1758197001910.jpg'
if not os.path.exists(img_path):
    print(f"Image not found at {img_path}")
    exit(1)

bgr = cv2.imread(img_path)
regs = skin_score.detect_face_and_eyes(bgr)

viz_masks = {
    "face_skin": skin_score.build_clean_skin_mask(bgr, regs)
}

lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
L = lab[:, :, 0]

# Real Dark Spots detection: Blackhat
# Blackhat extracts dark regions smaller than the kernel.
# Kernel size should be larger than a pore, but smaller than a shadow. 
kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, kernel_bh)
# Threshold the blackhat
dark_spots_mask = (blackhat > 12).astype(np.uint8) * 255

# Filter out tiny noise from blackhat (pores)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dark_spots_mask = cv2.morphologyEx(dark_spots_mask, cv2.MORPH_OPEN, kernel_open)

# Real Wrinkles detection:
# Wrinkles are also dark lines. We can use a different blackhat, or bilateral + canny.
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Bilateral filter preserves edges (wrinkles) but removes texture
smooth_gray = cv2.bilateralFilter(gray, 9, 50, 50)
edges = cv2.Canny(smooth_gray, 40, 90)

print("Dark spots active pixels:", (dark_spots_mask > 0).sum())
print("Wrinkles active pixels:", (edges > 0).sum())

cv2.imwrite('test_dark_spots_mask.png', dark_spots_mask)
cv2.imwrite('test_wrinkles_mask.png', edges)
