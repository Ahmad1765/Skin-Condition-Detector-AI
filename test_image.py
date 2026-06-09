import cv2
import sys
import skin_score

img_path = r'd:\Skin Condition\1758197001910.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"Failed to load image: {img_path}")
    sys.exit(1)

scores, regs, overlays = skin_score.analyze(img)

print("--- SCORES ---")
for k, v in scores.items():
    print(f"{k}: {v}")

print("\n--- DEBUG STATS ---")
for k, v in skin_score.DEBUG.items():
    print(f"{k}: {v}")

card_grid = skin_score.render_card_grid(img, regs, scores, overlays)
if card_grid is not None:
    cv2.imwrite(r'd:\Skin Condition\test_output_1758197001910.jpg', card_grid)
    print("\nSaved test_output_1758197001910.jpg")
else:
    print("\nFailed to generate card grid.")
