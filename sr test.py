import cv2
import numpy as np
from numpy.linalg import inv


# dot product of pixels of a frame and a matrix
def frame_dot(frame, matrix):
    # error assertion
    if not (frame.shape[2] == matrix.shape[0] == matrix.shape[1]):
        raise ValueError("depth of frame did not match matrix size, or matrix was not square")
    # deep copy frame and cast to float
    dc_frame = cv2.resize(frame, (frame.shape[1], frame.shape[0])).astype(np.float64)
    shape = matrix.shape[0]

    # sum-product
    for i in range(shape):
        for j in range(shape):
            dc_frame[:, :, i] += dc_frame[:, :, i] * matrix[i, j]

    # recast frame
    dc_frame = cv2.max(dc_frame, 0).astype(np.uint8)
    return dc_frame


# contribution function of a timing point between 0 and 1
def c_func(t):
    return t


if __name__ == "__main__":
    # init video feeds
    vid1 = cv2.VideoCapture("seq 20 vid 1.mp4")
    vid2 = cv2.VideoCapture("seq 20 vid 2.mp4")

    # qol parameters
    initial_frame_offset = 60
    # feed matching parameters
    resized = (400, 800)
    frame_diff = 12  # 8; 12
    mim_crop = 5, 35  # 8, 37; 5, 35
    size = (resized[0] - mim_crop[1], resized[1] - mim_crop[0])
    # superresolution parameters
    q = 0.8
    a = np.array([[1, 1-c_func(1-q/2)], [c_func(q/2), 1]])
    a_inv = inv(a)
    print(a_inv)

    # video writer
    writer = cv2.VideoWriter("test1.mp4", -1, 60, size, True)

    # set initial frames
    vid1.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_offset + frame_diff)
    vid2.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_offset)

    # while videos are open
    while vid1.isOpened() and vid2.isOpened():
        ret1, frame1 = vid1.read()
        ret2, frame2 = vid2.read()

        # end of video
        if not ret1 or not ret2:
            break

        # resize with matching params
        frame1 = cv2.resize(frame1, resized)[mim_crop[0]:resized[1], mim_crop[1]:resized[0]]
        frame2 = cv2.resize(frame2, resized)[0:size[1], 0:size[0]]

        # linalg superresolution
        calc_r = frame_dot(np.dstack((frame1[:, :, 0], frame2[:, :, 0])), a_inv)
        calc_g = frame_dot(np.dstack((frame1[:, :, 1], frame2[:, :, 1])), a_inv)
        calc_b = frame_dot(np.dstack((frame1[:, :, 2], frame2[:, :, 2])), a_inv)

        calc_1 = np.dstack((calc_r[:, :, 0], calc_g[:, :, 0], calc_b[:, :, 0]))
        calc_2 = np.dstack((calc_r[:, :, 1], calc_g[:, :, 1], calc_b[:, :, 1]))

        # show frames
        cv2.imshow("blend", frame1)
        cv2.imshow("calc", calc_2)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
        cv2.imshow("calc", calc_1)

        # blending for calibration
        # blend = np.array(dimensions + (3,), np.uint8)
        # blend = (frame1 * 0.5).astype(np.uint8) + (frame2 * 0.5).astype(np.uint8)

        # write to video
        writer.write(calc_2)
        writer.write(calc_1)

        # keyboard exit condition
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    # release handles
    vid1.release()
    vid2.release()
    writer.release()

    cv2.destroyAllWindows()
