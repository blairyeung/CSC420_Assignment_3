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

def group_by_box(directions, tau=48):
    grouped = np.zeros(shape=(tau, tau, 6), dtype=int)
    shape = directions.shape
    dx = int(shape[0] / tau)
    dy = int(shape[1] / tau)
    print(dx, dy)
    for i in range(tau):
        for j in range(tau):
            start_x = i * dx
            end_x = start_x + dx
            start_y = j * dy
            end_y = start_y + dy
            for k in range(6):
                slice = directions[start_x:end_x]
                block = slice.T[start_y:end_y]
                grouped[i][j][k] = np.count_nonzero(block == (k+1))
    return grouped

def gradient_clipping(gradient, threshold):
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            magnitude = gradient[i][j][0] * gradient[i][j][0] + \
            gradient[i][j][1] * gradient[i][j][1]
            if gradient <= threshold:
                gradient[i][j][0] = 0
                gradient[i][j][1] = 0

def graph_by_box(image, grouped, tau=48):
    # grouped = np.zeros(shape=(tau, tau, 6), dtype=int)
    shape = image.shape
    dx = int(shape[0] / tau)
    dy = int(shape[1] / tau)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 20, forward=True)
    print(dx, dy)
    ax.imshow(image)
    for i in range(tau):
        for j in range(tau):
            x_pos = i * dx + int(dx/2)
            y_pos = j * dy + int(dy/2)
            for k in range(6):
                val = grouped[i][j][k]
                print(val)
                x_direct = np.sin(k*30)
                y_direct = np.cos(k*30)
                ax.quiver(y_pos, x_pos, x_direct*val/10, y_direct*val/10, width=0.002)
                print(i,j,k)
    plt.show()
    return None


if __name__ == '__main__':
    """
        Appraoch 2
    """
    raw_image = cv2.imread('Image.jpg', cv2.IMREAD_GRAYSCALE)
    shape = np.array(image).shape
    print(np.max(image))
    sobelx = np.array(cv2.Sobel(Image, cv2.CV_64F, 1, 0, ksize=3))
    sobely = np.array(cv2.Sobel(Image, cv2.CV_64F, 0, 1, ksize=3))
    gradient = c = np.divide(sobelx, sobely, out=np.zeros_like(sobelx), where=sobely!=0)

    threshlod = 0
    gradient[gradient < threshlod] = 360
    # gradient = gradient.transpose(1, 2, 0)
    directions = np.arctan(gradient) * 360 / math.pi

    directions = np.abs(directions)

    """
        Remember to do clipping! The clipped values should be labeled as -1 or 0, and not counted
    """

    directions = directions + 15 * np.ones(shape=directions.shape, dtype=int)

    directions = (directions / 30 + np.ones(shape=directions.shape, dtype=int)).astype(int)

    directions[directions > 6] = 1

    grouped = group_by_box(directions)

    graph_by_box(Image, grouped)

    print(directions)

    # directions = group_by_box(directions)

    # plt.imshow((directions))
    # plt.show()

    # print(np.max(directions))
    # print(np.min(directions))

    pass

    """
    tau = 10

    print(np.max(np.array(list(sobelx))))
    plt.imshow(np.clip(sobely, a_min=0, a_max=255))
    plt.show()
    """