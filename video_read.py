import numpy as np
import cv2

def every_frame(videos, funcs):
    vids = []
    for video in videos:
        vids.append(cv2.VideoCapture(video))
    while all_opened(vids):
        frames = []
        tsmeta = []
        for vid in vids:
            r, f = vid.read()
            if not r:
                return
            frames.append(f)
            tsmeta.append({})
        for func in funcs:
            frames, tsmeta = func(frames, tsmeta)
        cv2.waitKey(1)

def all_opened(vids):
    for vid in vids:
        if not vid.isOpened():
            return False
    return True
