import numpy as np
import cv2

def every_frame(videos, offsets, funcs, duration=-1, debug=False, start=0):
    vids = []
    tsmeta = []
    for video, offset in zip(videos, offsets):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset + start)
        vids.append(cap)
        tsmeta.append({"capture": cap})
    while all_opened(vids):
        frames = []
        if debug:
            print(str(vids[0].get(cv2.CAP_PROP_POS_FRAMES)) + "/" + str(vids[0].get(cv2.CAP_PROP_FRAME_COUNT)))
        for idx, vid in enumerate(vids):
            current_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame - offsets[idx] == duration:
                break
            r, f = vid.read()
            if not r:
                return
            frames.append(f)
            # if debug:
                # print(current_frame, end=", ")
        #if debug:
            #print()
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

def show_sequential(frames, names):
    name1, name2 = names
    for frame1, frame2 in zip(*frames):
        cv2.imshow(name1, frame1)
        cv2.imshow(name2, frame2)

        cv2.waitKey(100)
