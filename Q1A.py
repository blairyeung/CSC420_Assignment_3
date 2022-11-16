import matplotlib.pyplot as plt
import scipy
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg


size = 1000
sigma = 100
center_coord = (int(size/2), int(size/2))

thickness = -1

radius = 100

Img = 255 * np.ones(shape=(size, size,), dtype=np.uint8)

image = cv2.circle(Img, center_coord, radius, color=0, thickness=-1)

unclipped = np.array(Img, dtype=float)

rslt = scipy.ndimage.gaussian_laplace(unclipped, sigma=sigma)

array = np.array(list(rslt))

# print(array)

mid = int(size/2)

print(sigma)
# print(np.max(array), np.min(array))
plt.plot(list(range(size)), array[mid])
plt.title("Center-slice of the image with sigma " + str(sigma)+ ' and radius ' + str(radius))
plt.show()
"""

plt.imshow(rslt, cmap='Blues', interpolation='none')
print(np.min(array), np.max(array))
plt.show()
"""

plt.imshow(rslt)
plt.title("Image with sigma " + str(sigma) + ' and radius ' + str(radius))
plt.show()