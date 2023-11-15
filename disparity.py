import cv2
import numpy as np


if __name__ == "__main__":
    # init video feeds
    vid1 = cv2.VideoCapture("videos/seq 20 vid 1.mp4")
    vid2 = cv2.VideoCapture("videos/seq 20 vid 2.mp4")

    # qol parameters
    initial_frame_offset = 60
    # feed matching parameters
    resized = (400, 800)
    frame_diff = 12  # 8; 12
    mim_crop = 9, 0  # 8, 37; 5, 35
    size = (resized[0] - mim_crop[1], resized[1] - mim_crop[0])
    
    vid1.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_offset + frame_diff)
    vid2.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_offset)

    while vid1.isOpened() and vid2.isOpened():
    
        ret1, frame1 = vid1.read()
        ret2, frame2 = vid2.read()
        if not (ret1 and ret2):
            print("frames not available")
            raise RuntimeError

        frame1 = cv2.resize(frame1, resized)[mim_crop[0]:resized[1], mim_crop[1]:resized[0]]
        frame2 = cv2.resize(frame2, resized)[0:size[1], 0:size[0]]

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        stereo = cv2.StereoBM_create(numDisparities=48, blockSize=5)
        stereo.setMinDisparity(4)
        stereo.setSpeckleRange(16)
        stereo.setSpeckleWindowSize(45)
        stereo.setTextureThreshold(100)
        disparity1 = stereo.compute(frame1, frame2)

        cv2.imshow("frame1", frame1)
        cv2.imshow("frame2", frame2)
        cv2.imshow("disparity1", disparity1*256)

        if cv2.waitKey(250) & 0xFF == ord('q'):
            break

    vid1.release()
    vid2.release()
    
    cv2.destroyAllWindows()

