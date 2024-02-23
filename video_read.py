import numpy as np
import cv2

def every_frame(videos, funcs):
    vids = []
    tsmeta = []
    for video in videos:
        vids.append(cv2.VideoCapture(video))
        tsmeta.append({})
    while all_opened(vids):
        frames = []
        for vid in vids:
            r, f = vid.read()
            if not r:
                return
            frames.append(f)
        for func in funcs:
            frames, tsmeta = func(frames, tsmeta)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

def all_opened(vids):
    for vid in vids:
        if not vid.isOpened():
            return False
    return True
