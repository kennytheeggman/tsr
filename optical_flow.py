import cv2
import numpy as np
from copy import deepcopy as dc


def of(last_frame, next_frame, point):
    return


def optical_flow_1(last_frame, next_frame, grouped_points, scale, max_accel=20, centers=None):

    last_frame = cv2.resize(last_frame, None, fx=scale, fy=scale)
    next_frame = cv2.resize(next_frame, None, fx=scale, fy=scale)

    pyrlk_config = {
        "winSize": (int(300*scale)+1, int(600*scale)+1),
        "maxLevel": 2,
        "nextPts": None
    }

    if centers:
        ux, uy, udx, udy = centers
        transform = np.array([
            []
        ])

    displacements = None
    statuses = None
    for idx, group in enumerate(grouped_points):
        pyrlk_config |= {
            "prevImg": last_frame,
            "nextImg": next_frame,
            "prevPts": group * scale
        }
        
        if centers:
            ux, uy, udx, udy = centers
            pyrlk_config |= {
                "winSize": (max_accel * scale * 2, (norm(np.array([udx, udy])) + max_accel) * 2)
            }

        pts, status, err = cv2.calcOpticalFlowPyrLK(**pyrlk_config)
        if displacements != None and statuses != None:
            displacements = np.concatenate((displacements, (pts/scale - group).astype(int)))
            statuses = statuses + status
        else:
            displacements = (pts/scale - group).astype(int)
            statuses = status

    return displacements, statuses

def norm(vec):
    return (vec[0]**2 + vec[1]**2)**0.5

def opt_flow(prev_frame, next_frame, max_accel, prev_segments, centers):
    # max_accel: maximum allowed estimated pixel based motion acceleration
    # prev_img: last frame
    # next_img: current frame
    # prev_segments: list of list of 4-tuples, where 4-tuples are (x, y, dx, dy), grouped into kmeans groups
    # centers: kmeans centers

    displacements = []
    statuses = []

    prev_img = dc(prev_frame)
    next_img = dc(next_frame)

    for corners, center in zip(prev_segments, centers):
        pyrlk_config, transform = _calc_params(center, max_accel, prev_img, next_img, corners)
        inv_transform = _calc_params_inverse(center)
        prev_img = _apply_rotation_matrix(prev_img, transform)
        next_img = _apply_rotation_matrix(next_img, transform)

        pts, status, err = cv2.calcOpticalFlowPyrLK(**pyrlk_config)
        pts = _apply_inverse_rotation_matrix(pts, transform)
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

def _calc_params_inverse(center):
    ux, uy, udx, udy = center
    displacement = (udx**2 + udy**2)**0.5
    udx, udy = udx / displacement, udy / displacement
    inv_affine_matrix = np.array([
        [udy, -udx, dx],
        [udx, udy, dy],
        [0, 0, 1]
    ])

    return inv_affine_matrix
    
def _apply_rotation_matrix(frame, transform):
    affine_config = {
        "src": frame,
        "M": transform
        # TODO: add dst size argument to preserve data loss
    }
    frame = cv2.warpAffine(**affine_config)
    return frame

def _transform_point(point, transform):
    vector_b = np.array([point[0], point[1], 1])
    vector_a = transform
    ret_vector = np.dot(vector_a, vector_b)
    print(ret_vector[:-1])

if __name__ == "__main__":
    _transform_point((1, 1), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
