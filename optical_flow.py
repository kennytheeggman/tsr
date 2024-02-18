import cv2
import numpy as np
from copy import deepcopy as dc


def opt_flow(max_accel, prev_img, next_img, prev_segments, centers):
    # max_accel: maximum allowed estimated pixel based motion acceleration
    # prev_img: last frame
    # next_img: current frame
    # prev_segments: list of list of 4-tuples, where 4-tuples are (x, y, dx, dy), grouped into kmeans groups
    # centers: kmeans centers

    displacements = []
    statuses = []

    for corners, center in zip(prev_segments, centers):
        pyrlk_config, transform = _calc_params(center, max_accel, prev_img, next_img, corners)
        _apply_rotation_matrix(prev_img)
        _apply_rotation_matrix(next_img)

        pts, status, err = cv2.calcOpticalFlowPyrLK(**pyrlk_config)
        pts = _apply_inverse_rotation_matrix(pts)
        displacements += pts
        statuses += status

    return displacements, statuses

def _calc_params(center, max_acceli, prev_img, next_img, corners):
    ux, uy, udx, udy = center
    displacement = (udx**2 + udy**2)**0.5
    pyrlk_config = {
        "winSize": (2*max_accel, 2*(displacement + max_accel)),
        "maxLevel": 2,
        "nextPts": None,
        "prevImg": prev_img,
        "nextImg": next_img,
        "prevPts": corners.astype(np.float32)
    }
    udx, udy = udx / displacement, udy / displacement
    cv_affine_matrix = np.array([
        [udy, udx, dx],
        [-udx, udy, dy],
    ])

    return pyrlk_config, cv_affine_matrix
    
def _apply_rotation_matrix(frame, transform):
    affine_config = {
        "src": frame,
        "M": transform
        # TODO: add dst size argument to preserve data loss
    }
    frame = cv2.warpAffine(**affine_config)
    return frame
