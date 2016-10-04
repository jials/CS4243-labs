import cv2
import numpy as np

files = ['test1', 'test2', 'test3']

sobel_x = np.array([
    [-1 ,0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewitt_y = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])


def convolve(img, kernel):
    # http://stackoverflow.com/questions/31615554/python-laplace-filter-returns-wrong-values
    result = np.zeros_like(img)
    r, c = img.shape
    for row in range(1, r - 1):
        for col in range(1, c - 1):
            val = kernel * img[(row - 1):(row + 2), (col - 1):(col + 2)]
            result[row, col] = np.sum(val)

    return result


def bgr_to_hsv(bgr):
    # Code from Week 2's Lab
    bgr = bgr.astype('float')
    hsv = np.zeros_like(bgr)
    # opencv represents it as B G R
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    b, g, r = b / 255, g / 255, r / 255
    c_max = np.maximum(b, g)
    c_max = np.maximum(c_max, r)
    c_min = np.minimum(b, g)
    c_min = np.minimum(c_min, r)

    # assigning brightness
    hsv[..., 2] = c_max

    # assigning saturation
    delta = c_max - c_min
    mask = c_max != c_min
    hsv[mask, 1] = delta[mask] / c_max[mask]  # by default they are zeroes, only assign when applicable

    # assigning hue
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = r[mask] / delta[mask]
    gc[mask] = g[mask] / delta[mask]
    bc[mask] = b[mask] / delta[mask]
    hsv[..., 0] = np.select(
        [r == c_max, g == c_max], [(gc - bc) % 6.0, 2.0 + bc - rc], default=4.0 + rc - gc)
    hsv[..., 0] = (hsv[..., 0] * 60.0 / 360.0)   # normalised
    return hsv * 255


def edge_strength(img_x, img_y):
    result = np.sqrt(img_x * img_x + img_y * img_y)
    # normalise
    result *= 255.0 / np.max(result)
    return result


def thinning(img):
    result = np.zeros_like(img)
    r, c = img.shape
    for row in range(1, r - 1):
        for col in range(1, c - 1):
            # check if its edge strength is the maximum either along the horizontal direction or the vertical direction
            if (img[row, col] > img[row + 1, col] and img[row, col] > img[row - 1, col]) \
                    or (img[row, col] > img[row, col - 1] and img[row, col] > img[row, col + 1]):
                result[row, col] = img[row, col]
                continue
            # non-maximal suppression
            result[row, col] = 0

    return result


for file in files:
    image = cv2.imread(file + '.jpg')
    # take the value only (grey scale)
    image = bgr_to_hsv(image)[..., 2]

    image_x_sobel = convolve(image, sobel_x)
    image_y_sobel = convolve(image, sobel_y)
    result = edge_strength(image_x_sobel, image_y_sobel)
    cv2.imwrite(file + '_sobel.jpg', result)
    result = thinning(result)
    cv2.imwrite(file + '_sobel_thinned.jpg', result)

    image_x_prewitt = convolve(image, prewitt_x)
    image_y_prewitt = convolve(image, prewitt_y)
    result = edge_strength(image_x_prewitt, image_y_prewitt)
    cv2.imwrite(file + '_prewitt.jpg', result)
    result = thinning(result)
    cv2.imwrite(file + '_prewitt_thinned.jpg', result)
