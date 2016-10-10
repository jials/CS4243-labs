import cv2
import numpy as np

files = ['checker', 'flower', 'test1', 'test2', 'test3']
steps = 10

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])


def flip_kernel(kernel):
    kernel = np.fliplr(kernel)
    kernel = np.flipud(kernel)
    return kernel


def gauss_kernels(size, sigma=1.0):
    # returns a 2d gaussian kernel
    if size < 3:
        size = 3
    m = size/2
    x, y = np.mgrid[-m:m+1, -m:m+1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel_sum = kernel.sum()
    if not sum == 0:
        kernel = kernel / kernel_sum

    return kernel


def convolve(img, kernel):
    # flip the kernel (part of the convolution process)
    kernel = flip_kernel(kernel)
    result = np.zeros_like(img)
    r, c = img.shape
    for row in range(1, r - 1):
        for col in range(1, c - 1):
            val = kernel * img[(row - 1):(row + 2), (col - 1):(col + 2)]
            result[row, col] = np.sum(val)

    return result


def bgr_to_hsv(bgr):
    # from lab 2
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


for file in files:
    orig_image = cv2.imread(file + '.jpg')
    # take the value only (grey scale)
    image = bgr_to_hsv(orig_image)[..., 2]

    gx = convolve(image, sobel_x)
    gy = convolve(image, sobel_y)
    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy

    W_xx = convolve(I_xx, gauss_kernels(size=3))
    W_xy = convolve(I_xy, gauss_kernels(size=3))
    W_yy = convolve(I_yy, gauss_kernels(size=3))

    max_x, max_y = image.shape
    res = np.empty_like(image)
    for x in range(steps, max_x, steps):
        for y in range(steps, max_y, steps):
            W = np.array([
                [W_xx[x][y], W_xy[x][y]],
                [W_xy[x][y], W_yy[x][y]]
            ])
            detW = np.linalg.det(W)
            traceW = np.trace(W)
            res[x][y] = detW - 0.06 * traceW * traceW

    maxRes = np.max(res)
    corners = []
    threshold = 0.1 * maxRes

    for rows in range(max_x):
        for cols in range(max_y):
            if res[rows][cols] > threshold:
                corners.append((rows, cols))

    for corner in corners:
        r, c = corner
        for x, y in [(x, y) for x in [-3, -2, -1, 0, 1, 2, 3] for y in [-3, -2, -1, 0, 1, 2, 3]]:
            if r + x < max_x and c + y < max_y:
                # only draw the border
                if r + x == r - 3 or r + x == r + 3 or c + y == c - 3 or c + y == c + 3:
                    orig_image[r+x][c+y] = [255, 0, 0]

    cv2.imwrite(file + '_corners_with_10_steps.jpg', orig_image)
