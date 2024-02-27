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
        meta[idx]["original"] = frame.copy()
        frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if "frame number" in meta[idx].keys():
            meta[idx]["frame number"] += 1
        else:
            meta[idx]["frame number"] = 0
    
    return frames, meta

# convert to color
def color(frames ,meta):
    for idx, frame in enumerate(frames):
        frames[idx] = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
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
        "qualityLevel": 0.1, 
        "minDistance": 15
    }
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        frame[frame < 20] = 0
        current_frame = meta[idx]["capture"].get(cv2.CAP_PROP_POS_FRAMES)
        # find corners
        feature_config |= { "image": frame }
        corners = cv2.goodFeaturesToTrack(**feature_config)
        dont_check = lambda: len(frame_meta["corners"]) > len(corners) and frame_meta["frame number"]%10!=0
        if "corners" in frame_meta.keys() and dont_check():
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

# k means clustering
def kmeans(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "corners" not in frame_meta.keys():
            break
        compact, labels, centers = cv2.kmeans(frame_meta["corners"].astype(np.float32), 10, frame_meta["corners"].copy(), (cv2.TERM_CRITERIA_EPS, 100, 100), 100, cv2.KMEANS_PP_CENTERS)
        meta[idx]["centers"] = centers.astype(int)
    return frames, meta

# contouring
def contour(frames, meta):
    np_abs = np.vectorize(abs)
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        frame = frame.astype(int)
        frame = abs(frame - 127)
        frame = frame.astype(np.uint8)
        ret, thresh = cv2.threshold(frame, 20, 255, 0)
        contours, hier, = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        new_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                new_contours.append(contour)

        meta[idx]["contours"] = new_contours
        frames[idx] = frame
    return frames, meta

# optical flow calculation
def optical_flow(frames, meta):
    of_config = {
            "scale": 0.05
    }
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "old frame" not in frame_meta.keys() or len(frame_meta["old frame"])<2:
            break
        last_frame = frame_meta["old frame"][1]
        next_frame = frame_meta["copy"][1]
        of_config |= {
            "last_frame": last_frame,
            "next_frame": next_frame,
            "grouped_points": [frame_meta["corners"].astype(np.float32)]
        }
        meta[idx]["displacements"], meta[idx]["status"] = optical_flow_1(**of_config)
    
    return frames, meta


def velocity(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "corners" not in frame_meta.keys():
            break
        if "displacements" not in frame_meta.keys():
            break
        contour_disps = []
        contour_idcs = []
        for contour_idx, contour in enumerate(frame_meta["contours"]):
            contour = contour.astype(np.float32)
            x, y = contour.mean(axis=0)[0]
            contour -= np.array([x, y]).astype(np.float32)
            contour *= 1.1
            disp_sum = np.array([0, 0])
            len_disp = 0
            this_contour_idcs = []
            for corner_idx, (disp, corner) in enumerate(zip(frame_meta["displacements"], frame_meta["corners"])):
                corner = corner.astype(np.float32)
                corner -= np.array([x, y]).astype(np.float32)
                if cv2.pointPolygonTest(contour, corner, False) >= 0:
                    disp_sum[0] += disp[0]
                    disp_sum[1] += disp[1]
                    len_disp += 1
                    this_contour_idcs.append(corner_idx)
            contour_idcs.append(this_contour_idcs)

            contour_disps.append(disp_sum / len_disp if len_disp != 0 else np.array([0, 0]))
        meta[idx]["contour velocities"] = contour_disps
        meta[idx]["contour corner indices"] = contour_idcs
    return frames, meta


def pair_contours(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "corners" not in frame_meta.keys():
            break
        if "displacements" not in frame_meta.keys():
            break
        if "contours" not in frame_meta.keys():
            break

        grouped_contours = []
        for idx1, contour_1 in enumerate(frame_meta["contours"]):
            min_diff = 1e8
            min_idx = 0
            for idx2, contour_2 in enumerate(frame_meta["contours"]):
                if idx1 == idx2:
                    continue
                moments_1 = cv2.moments(contour_1)
                moments_2 = cv2.moments(contour_2)
                
                dmu20 = abs(np.log2(abs(moments_1["mu20"]))-np.log2(abs(moments_2["mu20"])))
                dmu11 = abs(np.log2(abs(moments_1["mu11"]))-np.log2(abs(moments_2["mu11"])))
                dmu02 = abs(np.log2(abs(moments_1["mu02"]))-np.log2(abs(moments_2["mu02"])))
                dmu30 = abs(np.log2(abs(moments_1["mu30"]))-np.log2(abs(moments_2["mu30"])))
                dmu21 = abs(np.log2(abs(moments_1["mu21"]))-np.log2(abs(moments_2["mu21"])))
                dmu12 = abs(np.log2(abs(moments_1["mu12"]))-np.log2(abs(moments_2["mu12"])))
                dmu03 = abs(np.log2(abs(moments_1["mu03"]))-np.log2(abs(moments_2["mu03"])))

                vx1 = frame_meta["contour velocities"][idx1][0]
                vx2 = frame_meta["contour velocities"][idx2][0]
             
                vy1 = frame_meta["contour velocities"][idx1][1]
                vy2 = frame_meta["contour velocities"][idx2][1]

                dvx = (vx1-vx2)**2 if vx1 != 0 and vx2 != 0 else 10
                dvy = (vy1-vy2)**2 if vy1 != 0 and vy2 != 0 else 10

                # print(moments_2)
                
                diff = dmu20 + dmu11 + dmu02 + dmu30 + dmu21 + dmu12 + dmu03 + dvx + dvy
                if diff < min_diff:
                    min_diff = diff
                    min_idx = idx2
            pair = [idx1, min_idx, min_diff]
            grouped_contours.append(pair)
            # when 2 values are the same, they are a pair

        meta[idx]["paired"] = grouped_contours
        print(str(idx) + ": " + str(frame_meta["paired"]))
    return frames, meta


# draw contours
def draw_contours(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        # frame = frame_meta["original"]
        if "contour velocities" not in frame_meta.keys():
            continue
        shape = frame.shape
        disps = frame_meta["contour velocities"]
        for cidx, (contour, disp) in enumerate(zip(frame_meta["contours"], frame_meta["contour velocities"])):
            if disp[0] == 0 and disp[1] == 0:
                cv2.drawContours(frame, frame_meta["contours"], cidx, (255, 255, 0), 2)
                continue
            cv2.drawContours(frame, frame_meta["contours"], cidx, (disp[0]/100*127 + 127, disp[1]/100*127 + 127, 0), -1)
        frames[idx] = frame
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

# draw centers for k means
def draw_kmeans(frames, meta):
    circle_config = {
        "radius": 15,
        "color": (0, 255, 0),
        "thickness": -1
    }
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "centers" not in frame_meta.keys():
            break
        for center in frame_meta["centers"]:
            circle_config |= {"img": frame, "center": center}
            cv2.circle(**circle_config)
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
        if "contours" in frame_meta.keys():
            meta[idx]["old contours"] = frame_meta["contours"]
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
        resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow(str(idx), resized)
    
    return frames, meta


#################################### Metadata Processing Pipeline ####################################  


# print metadata
def print_meta(frames, meta):
    if meta[0]["frame number"] == 0:
        return frames, meta
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
            # kmeans,
            optical_flow,
            contour,
            velocity,
            pair_contours,
            color, 
            draw_contours,
            draw_features,
            # draw_kmeans,
            draw_of, 
            show, 
            store_copy
    ]
    # processing_order.append(print_meta)
    # processing_order.append(print_frames)
    every_frame(["videos/seq 34 vid 2.mp4", "videos/seq 34 vid 1.mp4"], processing_order)
