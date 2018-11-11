from PIL import Image
from PIL import ImageDraw
import numpy as np

# im = Image.open(r"E:\save_path\landmarks\48\images\000007.jpg")
# imDraw = ImageDraw.Draw(im)
# # imDraw.rectangle((59.60,56.00,78.29,74.94))
# imDraw.point((12.94,14.02),fill="red")
# imDraw.point((33.44,13.48),fill="red")
# imDraw.point((21.03,26.43),fill="red")
# imDraw.point((14.02,35.60),fill="red")
# imDraw.point((31.28,35.60),fill="red")
# im.show(im)

a = np.zeros((100),dtype=np.int32)
a[99] = 1

print(a)