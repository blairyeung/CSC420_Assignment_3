import matplotlib.pyplot as plt
import scipy
import cv2
import numpy as np
import math


def getVerticalGradient(image):
    new_image = np.zeros(shape=image.shape, dtype=int)
    kernal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for m in range(3):
                for n in range(3):
                    shifted = (i + m - 1, j + n - 1)
                    if min(shifted) < 0 or shifted[0] == shape[0] or shifted[1] == shape[1]:
                        continue
                    new_image[i][j] += kernal[m][n] * image[shifted[0]][shifted[1]]
    return new_image


def getHorizontalGradient(image):
    new_image = np.zeros(shape=image.shape, dtype=int)
    kernal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for m in range(3):
                for n in range(3):
                    shifted = (i + m - 1, j + n - 1)
                    if min(shifted) < 0 or shifted[0] == shape[0] or shifted[1] == shape[1]:
                        continue
                    new_image[i][j] += kernal[m][n] * image[shifted[0]][shifted[1]]
    return new_image


def getGradient(image):
    gradient = np.zeros(shape=(image.shape[0], image.shape[1], 2), dtype=int)
    vertical_slice = getVerticalGradient(image)
    horizontal_slice = getHorizontalGradient(image)
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            gradient[i][j] = np.array([horizontal_slice[i][j], vertical_slice[i][j]])
    return gradient


if __name__ == '__main__':
    Image = cv2.imread('Image.jpg', cv2.IMREAD_GRAYSCALE)
    shape = np.array(Image).shape
    print(np.max(Image))
    sobelx = np.array(cv2.Sobel(Image, cv2.CV_64F, 1, 0, ksize=3))
    sobely = np.array(cv2.Sobel(Image, cv2.CV_64F, 0, 1, ksize=3))
    gradient = c = np.divide(sobelx, sobely, out=np.zeros_like(sobelx), where=sobely!=0)
    # gradient = gradient.transpose(1, 2, 0)
    directions = np.arctan(gradient) * 360 / math.pi

    plt.imshow((directions))
    plt.show()

    print(np.max(directions))
    print(np.min(directions))

    pass

    """
    tau = 10

    print(np.max(np.array(list(sobelx))))
    plt.imshow(np.clip(sobely, a_min=0, a_max=255))
    plt.show()
    """