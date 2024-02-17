from video_read import every_frame
import cv2
import numpy as np
import copy

def gray(frames, meta):
    new_frames = []
    for frame in frames:
        new_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return new_frames, meta

def sharpen(frames, meta):
    new_frames = []
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    for frame in frames:
        new_frames.append(cv2.filter2D(frame, -1, kernel))
    return new_frames, meta

def features(frames, meta):
    track_config = {"maxCorners": 200, "qualityLevel": 0.01, "minDistance": 10}
    meta_copy = copy.deepcopy(meta)
    for idx, frame in enumerate(frames):
        frame_config = copy.deepcopy(track_config)
        frame_config["image"] = frame
        corners = cv2.goodFeaturesToTrack(**frame_config)
        meta_copy[idx]["corners"] = corners[:,0,:]
    return frames, meta_copy

def draw_features(frames, meta):
    new_frame = []
    for idx, frame in enumerate(frames):
        cv2.circle(frame, (int(meta[idx]["corners"][idx,0]), int(meta[idx]["corners"][idx,1])), 5, (0, 0, 255), -1)
        new_frame.append(frame)
    return new_frame, meta

def print_meta(frames, meta):
    print(meta)
    return frames, meta

def show(frames, meta):
    for idx, frame in enumerate(frames):
        cv2.imshow(str(idx), frame)
    return frames, meta

processing_order = [gray, sharpen, features, draw_features, print_meta, show]
every_frame(["videos/seq 34 vid 2.mp4", "videos/seq 34 vid 1.mp4"], processing_order)
