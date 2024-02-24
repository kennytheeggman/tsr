from video_read import every_frame
import cv2
import numpy as np
from copy import deepcopy as dc
from optical_flow import optical_flow_1


#################################### Image Processing Pipeline ####################################  


# convert to grayscale
def gray(frames, meta):
    for idx, frame in enumerate(frames):
        # convert color
        frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frames, meta

# sharpen image with 3x3 kernel
def sharpen(frames, meta):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    for idx, frame in enumerate(frames):
        # apply kernel
        frames[idx] = cv2.filter2D(frame, -1, kernel)
    
    return frames, meta

# identify features to track
def features(frames, meta):
    feature_config = {
        "maxCorners": 30, 
        "qualityLevel": 0.01, 
        "minDistance": 15
    }
    for idx, frame in enumerate(frames):
        current_frame = meta[idx]["capture"].get(cv2.CAP_PROP_POS_FRAMES)
        # find corners
        feature_config |= { "image": frame }
        corners = cv2.goodFeaturesToTrack(**feature_config) 
        if "corners" in meta[idx].keys() and current_frame%8!=0 and len(meta[idx]["corners"])>len(corners):
            return frames, meta
        # add corner info to metadata
        meta[idx]["corners"] = corners[:,0,:].astype(int)
    
    return frames, meta

# subtract consecutive frames
def subtract(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "old frame" not in frame_meta.keys():
            break
        frame_copy = frame_meta["copy"][0].astype(int) 
        old_frame = frame_meta["old frame"][0].astype(int)
        frames[idx] = frames[idx].astype(int)
        frames[idx] = (frame_copy - old_frame) // 2 + 128
        frames[idx] = frames[idx].astype(np.uint8)

        meta[idx]["copy"].append(frames[idx])

    return frames, meta

# optical flow calculation
def optical_flow(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "old frame" not in frame_meta.keys() or len(frame_meta["old frame"])<2:
            break
        last_frame = frame_meta["old frame"][1]
        next_frame = frame_meta["copy"][1]
        meta[idx]["displacements"], meta[idx]["status"] = optical_flow_1(last_frame, next_frame, [frame_meta["corners"].astype(np.float32)], 0.25)
    
    return frames, meta

# draw points on features
def draw_features(frames, meta):
    circle_config = {
        "radius": 5,
        "color": (0, 0, 255),
        "thickness": -1
    }
    for idx, frame in enumerate(frames):
        # draw circles on features
        for corner in meta[idx]["corners"]:
            circle_config |= { "img": frame, "center": corner }
            cv2.circle(**circle_config)
        # overwrite frame
        frames[idx] = frame
    
    return frames, meta

# draw optical flow arrows
def draw_of(frames, meta):
    arrowed_line_config = {
        "color": (0, 0, 255),
        "thickness": 2
    }
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        arrowed_line_config |= { "img": frame }
        if "displacements" not in frame_meta.keys():
            break
        for c, d, status in zip(frame_meta["corners"], frame_meta["displacements"], frame_meta["status"]):
            if status[0] == 0:
                break
            arrowed_line_config |= {
                "pt1": c,
                "pt2": c + d
            }
            cv2.arrowedLine(**arrowed_line_config)

    return frames, meta

# store frames 
def store_copy(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        meta[idx]["old frame"] = frame_meta["copy"]
        if "displacements" in meta[idx].keys():
            meta[idx]["corners"] += meta[idx]["displacements"]

    return frames, meta

# store a copy of the original frame
def copy_frame(frames, meta):
    for idx, frame in enumerate(frames):
        if "frame copy queue" not in meta[idx].keys():
            meta[idx]["copy"] = []
        meta[idx]["copy"].append(dc(frame))

    return frames, meta

# show frames
def show(frames, meta):
    for idx, frame in enumerate(frames):
        # show frame
        cv2.imshow(str(idx), frame)
    
    return frames, meta


#################################### Metadata Processing Pipeline ####################################  


# print metadata
def print_meta(frames, meta):
    print(meta)
    return frames, meta

# print frames
def print_frames(frames, meta):
    print(frames)
    return frames, meta


#################################### Main Execution Loop ####################################  


if __name__ == "__main__":
    processing_order = [
            gray, 
            copy_frame, 
            sharpen, 
            subtract, 
            features, 
            optical_flow, 
            draw_features, 
            draw_of, 
            show, 
            store_copy
    ]
    # processing_order.append(print_meta)
    # processing_order.append(print_frames)
    every_frame(["videos/seq 34 vid 2.mp4", "videos/seq 34 vid 1.mp4"], processing_order)
