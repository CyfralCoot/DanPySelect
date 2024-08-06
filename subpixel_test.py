import cv2
import matplotlib.pyplot as plt
import numpy as np

from subpixel_edges import subpixel_edges
from PIL import Image, ImageTk

# (optional) 
help(subpixel_edges)

pil_image = Image.open("test_img.bmp")
cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
edges = subpixel_edges(cv_image, 25, 0, 2)

plt.imshow(cv_image)
plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40)
plt.show()