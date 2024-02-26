import numpy as np
import cv2

def every_frame(videos, funcs):
    vids = []
    tsmeta = []
    for video in videos:
        cap = cv2.VideoCapture(video)
        vids.append(cap)
        tsmeta.append({"capture": cap})
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
        if key == ord(" "):
            key = cv2.waitKey(-1)
            if key == ord("q"):
                break
            if key == ord(" "):
                continue
            else:
                pass

def all_opened(vids):
    for vid in vids:
        if not vid.isOpened():
            return False
    return True
