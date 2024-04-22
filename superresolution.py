import numpy as np
import cv2

def superres(imgs):
    if len(imgs) == 0:
        return
    if len(imgs) == 2:
        A, B = imgs
        C = np.zeros_like(B)
    else:
        A, B, C = imgs
    a = 5/9 * A + 4/9 * B
    b = 2/9 * A + 5/9 * B + 2/9 * C
    c = 4/9 * B + 5/9 * C
    return [a, b, c]
