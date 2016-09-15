import cv2
import numpy as np

name_of_files = ['concert', 'sea1', 'sea2']


def bgr_to_hsv(bgr):
    # http://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
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


def hsv_to_bgr(hsv):
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    bgr = np.zeros_like(hsv)
    h, s, v = hsv[..., 0] / 255.0 * 360.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    c = s * v
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c

    q, p = m.shape
    for i in range(0, q):
        for j in range(0, p):
            if 0 <= h[i, j] < 60:
                bgr[i, j, 0] = 0
                bgr[i, j, 1] = x[i, j]
                bgr[i, j, 2] = c[i, j]
            elif 60 <= h[i, j] < 120:
                bgr[i, j, 0] = 0
                bgr[i, j, 1] = c[i, j]
                bgr[i, j, 2] = x[i, j]
            elif 120 <= h[i, j] < 180:
                bgr[i, j, 0] = x[i, j]
                bgr[i, j, 1] = c[i, j]
                bgr[i, j, 2] = 0
            elif 180 <= h[i, j] < 240:
                bgr[i, j, 0] = c[i, j]
                bgr[i, j, 1] = x[i, j]
                bgr[i, j, 2] = 0
            elif 240 <= h[i, j] < 300:
                bgr[i, j, 0] = c[i, j]
                bgr[i, j, 1] = 0
                bgr[i, j, 2] = x[i, j]
            elif 300 <= h[i, j] < 360:
                bgr[i, j, 0] = x[i, j]
                bgr[i, j, 1] = 0
                bgr[i, j, 2] = c[i, j]

            bgr[i, j, 0] = (bgr[i, j, 0] + m[i, j]) * 255
            bgr[i, j, 1] = (bgr[i, j, 1] + m[i, j]) * 255
            bgr[i, j, 2] = (bgr[i, j, 2] + m[i, j]) * 255

    return bgr.astype('uint8')


def histogram_equalization(value):
    hist, bins = np.histogram(value.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    value = value.astype('uint8')   # expects integer not float
    value_normalised = cdf[value]
    return value_normalised

for name_of_file in name_of_files:
    image = cv2.imread(name_of_file + '.jpg')
    hsv = bgr_to_hsv(image)
    cv2.imwrite(name_of_file + '_hue.jpg', hsv[..., 0])
    cv2.imwrite(name_of_file + '_saturation.jpg', hsv[..., 1])
    cv2.imwrite(name_of_file + '_brightness.jpg', hsv[..., 2])

    value_normalised = histogram_equalization(hsv[..., 2])
    hsv_normalised = np.zeros_like(hsv)
    hsv_normalised[..., 0] = hsv[..., 0]
    hsv_normalised[..., 1] = hsv[..., 1]
    hsv_normalised[..., 2] = value_normalised
    bgr_normalised = hsv_to_bgr(hsv_normalised)
    cv2.imwrite(name_of_file + '_histeq.jpg', bgr_normalised)

    bgr = hsv_to_bgr(hsv)
    cv2.imwrite(name_of_file + '_hsv2rgb.jpg', bgr)