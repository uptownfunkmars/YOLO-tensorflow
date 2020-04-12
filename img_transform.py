import numpy as np
import cv2 as cv

def imgCV2_recolor(img, a=0.1):
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2.0 - 1.0

    img = img * (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform() * 2 - 1
    img = cv.pow(img/mx, 1.0 + up * 0.5)

    return np.array(img * 255.0, np.uint8)

def imgCV2_affine_trans(img):
    h, w, c = img.shape
    scale = np.random.uniform() / 10.0 + 1.0
    max_offx = (scale - 1.0) * w
    max_offy = (scale - 1.0) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    img = cv.resize(img, (0, 0), fx = scale, fy=scale)
    img = img[offy: (offy + h), offx : (offx + w)]

    flip = np.random.binomial(1, 0.5)
    if flip: img = cv.flip(img, 1)
    return img, [w, h, c], [scale, [offx, offy], flip]
