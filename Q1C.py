import matplotlib.pyplot as plt
import scipy
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg


size = 1000
sigma = 34
sqsize = 100
start_coord = (int(size/2 - sqsize / 2), int(size/2 - sqsize/2))
end_coord = (int(size/2 + sqsize / 2), int(size/2 + sqsize/2))

thickness = -1


Img = 255 * np.ones(shape=(size, size,), dtype=np.uint8)



image = cv2.rectangle(Img, start_coord, end_coord, color=0, thickness=-1)

unclipped = np.array(Img, dtype=float)

rslt = scipy.ndimage.gaussian_laplace(unclipped, sigma=sigma)

array = np.array(list(rslt))

# print(array)

mid = int(size/2)

print(sigma)
# print(np.max(array), np.min(array))
plt.plot(list(range(size)), array[mid])
plt.title("Center-slice of the image with sigma " + str(sigma)+ ' and size ' + str(sqsize))
plt.show()
"""

plt.imshow(rslt, cmap='Blues', interpolation='none')
print(np.min(array), np.max(array))
plt.show()
"""

plt.imshow(rslt)
plt.show()

plt.imshow(image)
plt.show()