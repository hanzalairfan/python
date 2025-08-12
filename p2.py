# # n = [1,2,3,4,5,6,5]
# # m = (len(n))
# # unique = []
# # print(n)
# # print(max(n))
# # print(min(n))

# # total = sum(n)
# # print(total)
# # sum = total / m
# # print(sum)

# # for i in n[:]:
# #     if (i % 2 == 0):
# #         n.remove(i)
# # print(n)
        
# # for i in n:
# #     if n.count(i) > 1 and i not in unique:
# #         unique.append(i)
# # print(unique)

# # o = [2,4,5,7]
# # s= n + o
# # u = []
# # for i in s:
# #     if i not in u:
# #         u.append(i)
# #     print(u)
    
# # for i in range(len(s)):
# #     for j in range(0,len(s) - i - 1):
# #         if s[j] > s[j + 1]:
# #             temp = s[j]
# #             s[j] = s[j + 1]
# #             s[j + 1] = temp
            
# # print(s)

# # no = [1,2,3,4,5,6,5]
# # for i in no:
# #     sq = i * i
# #     print(sq)

# # n = int(input("enter a number: "))
# # i = 0
# # while(i < 11):
# #     print(n ,"x", i, "=" ,(n * i))
# #     i += 1
# num = int(input("enter a number: "))
# sum = 0
# i = 1
# while(i<=num):
#     sum += i
#     i += 1

# print("the sum is:", sum)
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
import os

# Input file
img_path = "images.jpg"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# Convert to grayscale for contour detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours for all holograms
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Output folder for enhanced holograms
enhanced_dir = "hologram_icons_glow"
os.makedirs(enhanced_dir, exist_ok=True)

for i, cnt in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[0])):
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:  # ignore small noise
        # Crop
        crop = img[y:y+h, x:x+w]
        crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        
        # Make transparent background
        mask = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        crop_rgba[:, :, 3] = alpha

        # Convert to PIL for enhancement
        pil_img = Image.fromarray(crop_rgba)
        
        # Slight sharpen and contrast boost
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)

        # Create green glow
        glow = pil_img.copy().convert("RGBA")
        glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
        r, g, b, a = glow.split()
        glow = Image.merge("RGBA", (Image.new("L", r.size, 0), g, Image.new("L", b.size, 0), a))

        # Composite glow under original
        final_img = Image.alpha_composite(glow, pil_img)

        # Save
        icon_path = os.path.join(enhanced_dir, f"hologram_{i+1}.png")
        final_img.save(icon_path)

print("✅ All holograms extracted with glow! Check 'hologram_icons_glow/' folder.")
