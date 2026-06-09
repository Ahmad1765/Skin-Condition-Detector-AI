import cv2
import numpy as np
import skin_score

img = cv2.imread(r'd:\Skin Condition\1758197001910.jpg')
regs = skin_score.detect_face_and_eyes(img)
skin = skin_score.skin_color_mask(img)
mask = skin_score.union(
    skin_score.rect_to_mask(regs["left_cheek"], img.shape),
    skin_score.rect_to_mask(regs["right_cheek"], img.shape)
) & skin

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L = lab[:, :, 0]

# Original method
L_smooth = cv2.medianBlur(L, 5)
blurred = cv2.GaussianBlur(L_smooth, (25, 25), 0)
diff = blurred.astype(np.int16) - L_smooth.astype(np.int16)

# Blackhat method
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
blackhat = cv2.morphologyEx(L_smooth, cv2.MORPH_BLACKHAT, kernel)

print("Diff > 15:", float((diff[mask>0] > 15).mean()))
print("Blackhat > 15:", float((blackhat[mask>0] > 15).mean()))
