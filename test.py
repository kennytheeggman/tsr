from video_read import every_frame, show_sequential
import cv2
import numpy as np
from copy import deepcopy as dc
from optical_flow import optical_flow_1
from superresolution import superres
import time


#################################### Image Processing Pipeline ####################################  

def crop(frames, meta):
    meta[0]["start time"] = time.time()
    height, width, depth = frames[0].shape
    h_offsets = [40, 75]#[0, 23]
    v_offsets = [30, 10]
    hoff = max(h_offsets)
    voff = max(v_offsets)
    cropped_width = width - hoff
    cropped_height = height - voff
    frames[0] = frames[0][voff-v_offsets[0]:voff-v_offsets[0]+cropped_height, hoff:width, :]
    frames[1] = frames[1][voff-v_offsets[1]:voff-v_offsets[1]+cropped_height, hoff-h_offsets[0]:hoff-h_offsets[0]+cropped_width, :]
    frames[2] = frames[2][voff:height, hoff-h_offsets[1]:hoff-h_offsets[1]+cropped_width, :]
    return frames, meta

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
        cont = np.vstack([n for n in meta[idx]["contours"]]) if len(meta[idx]["contours"]) > 0 else np.array([])
        corners = np.array([cont[i][0] for i in range(0, len(cont), len(cont)//80+1)])
        if len(corners) == 0:
            corners = np.array([[0, 0]])
        #corners = cv2.goodFeaturesToTrack(**feature_config)
        dont_check = lambda: len(frame_meta["corners"]) > len(corners) and frame_meta["frame number"]%10!=0
        if "corners" in frame_meta.keys() and dont_check():
            return frames, meta
        # add corner info to metadata
        meta[idx]["corners"] = corners# .astype(int) #
        # meta[idx]["corners"] = corners[:,0,:].astype(int)
    
    return frames, meta

# subtract consecutive frames
def subtract(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "old frame" not in frame_meta.keys():
            break
        frame_copy = frame_meta["copy"][0].astype(np.int8) 
        old_frame = frame_meta["old frame"][0].astype(np.int8)
        #frames[idx] = frames[idx].astype(int)
        frames[idx] = (frame_copy - old_frame) // 2
        # frames[idx] = frames[idx].astype(np.uint8)


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
        # frame = frame.astype(np.int16)
        frame = abs(frame)
        frame = frame.astype(np.uint8)
        ret, thresh = cv2.threshold(frame, 4, 255, 0)
        contours, hier, = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        new_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 400:
                new_contours.append(contour)

        meta[idx]["contours"] = new_contours
        frames[idx] = frame
        meta[idx]["copy"].append(frame)
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
                
                d = lambda key: abs(np.log2(abs(moments_1[key]))-np.log2(abs(moments_2[key])))

                vx1 = frame_meta["contour velocities"][idx1][0]
                vx2 = frame_meta["contour velocities"][idx2][0]
             
                vy1 = frame_meta["contour velocities"][idx1][1]
                vy2 = frame_meta["contour velocities"][idx2][1]

                dvx = (vx1-vx2)**2 if vx1 != 0 and vx2 != 0 else 10
                dvy = (vy1-vy2)**2 if vy1 != 0 and vy2 != 0 else 10

                # print(moments_2)
                
                diff = d("mu20") + d("mu11") + d("mu02") + d("mu30") + d("mu21") + d("mu12") + d("mu03") + dvx * 10 + dvy * 10
                if diff < min_diff:
                    min_diff = diff
                    min_idx = idx2
            pair = [idx1, min_idx, min_diff]
            grouped_contours.append(pair)
            # when 2 values are the same, they are a pair

        meta[idx]["paired"] = grouped_contours
        reciprocated_pairs = []
        for pair in frame_meta["paired"]:
            reciprocal = [pair[1], pair[0], pair[2]]
            if reciprocal in meta[idx]["paired"] and reciprocal not in reciprocated_pairs:
                reciprocated_pairs.append(reciprocal)
        meta[idx]["reciprocated pairs"] = reciprocated_pairs
    return frames, meta

# combine pairs
def combine_pairs(frames, meta):
    rotate_dir = lambda p, v: [p[0]*v[1]-p[1]*v[0], p[0]*v[0]+p[1]*v[1]]
    reverse_dir = lambda p, v: [p[0]*v[1]+p[1]*v[0], p[1]*v[1]-p[0]*v[0]]
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        meta[idx]["repaired contours"] = []
        meta[idx]["repaired contour velocities"] = []
        if "reciprocated pairs" not in frame_meta.keys():
            return frames, meta
        already_processed = []
        for pair in frame_meta["reciprocated pairs"]:
            if pair[0] in already_processed or pair[1] in already_processed:
                continue
            if len(already_processed) > 6:
                break
            else:
                already_processed.append(pair[0])
                already_processed.append(pair[1])
            c1 = frame_meta["contours"][pair[0]]
            c2 = frame_meta["contours"][pair[1]]
            # create find the minimum distance between contours
            overlap = False
            condition = len(c1) // 100 + 1
            top1 = min(c1, key=lambda x: x[0][1])[0][1]
            bottom1 = max(c1, key=lambda x: x[0][1])[0][1]
            left1 = min(c1, key=lambda x: x[0][0])[0][0]
            right1 = max(c1, key=lambda x: x[0][0])[0][0]
            top2 = min(c2, key=lambda x: x[0][1])[0][1]
            bottom2 = max(c2, key=lambda x: x[0][1])[0][1]
            left2 = min(c2, key=lambda x: x[0][0])[0][0]
            right2 = max(c2, key=lambda x: x[0][0])[0][0]

            overlap = left1 <= left2 <= right1 and top1 <= top2 <= bottom1
            overlap = overlap or (left1 <= right2 <= right1 and top1 <= bottom2 <= bottom1)
            new_contour = np.array([[[0, 0]]])
            v1 = frame_meta["contour velocities"][pair[0]]
            v2 = frame_meta["contour velocities"][pair[1]]
            v = (v1 + v2) / 2
            mag_v = (v[0] ** 2 + v[1] ** 2) ** 0.5
            v /= mag_v if mag_v != 0 else 1
            if mag_v == 0:
                continue
            if overlap:
                com1 = sum(c1) / len(c1)
                com2 = sum(c2) / len(c2)
               
                c = []

                cont = np.vstack((c1, c2))
                
                new_contour = cv2.convexHull(cont)

            else:
                com1 = sum(c1) / len(c1)
                com2 = sum(c2) / len(c2)

                d = com2 - com1
                if np.dot(d, v) > 0:
                    new_contour = cv2.convexHull(c2)
                else:
                    new_contour = cv2.convexHull(c1)
            
            new_contour = new_contour.astype(np.int32)
            if cv2.contourArea(new_contour) < 5000:
                continue

            if "repaired contours" in meta[idx].keys():
                meta[idx]["repaired contours"].append(new_contour)
                meta[idx]["repaired contour velocities"].append(v)
            else:
                meta[idx]["repaired contours"] = [new_contour]
                meta[idx]["repaired contour velocities"] = [v]

    return frames, meta

def remove_entry(matches, e):
    for idx, d in reversed(list(enumerate(matches))):
        for f in e:
            for g in d[0]:
                if len(matches) > 0 and f["contour"].shape == g["contour"].shape and np.array_equal(f["contour"], g["contour"]):
                    try:
                        matches.pop(idx)
                    except:
                        #print(matches)
                        return
# match contours
def match_contours(frames, meta):
    all_repaired_contours = []
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        if "repaired contours" not in frame_meta.keys():
            return frames, meta
        context_repaired_contours = []
        for contour, velocity in zip(frame_meta["repaired contours"], frame_meta["repaired contour velocities"]):
            entry = {
                "contour": contour,
                "velocity": velocity,
                "area": cv2.contourArea(contour),
                "com": sum(contour) / len(contour),
                "origin": idx
            }
            context_repaired_contours.append(entry)
        context_repaired_contours.sort(key=lambda e: e["area"])
        all_repaired_contours.append(context_repaired_contours)
    
    matched_dists = []
    for idx1, entry1 in enumerate(all_repaired_contours[0]):
        for idx2, entry2 in enumerate(all_repaired_contours[1]):
            for idx3, entry3 in enumerate(all_repaired_contours[2]):
                context_match = [entry1, entry2, entry3]
                com1biased = entry1["com"] + 0.2*entry1["velocity"]
                com2biased = entry2["com"] + 0.2*entry2["velocity"]
                com3biased = entry3["com"] + 0.2*entry3["velocity"]

                total_dist = sum(sum((com1biased-com2biased)**2)) + \
                                sum(sum((com2biased-com3biased)**2)) + \
                                sum(sum((com3biased-com1biased)**2))
                matched_dists.append(tuple([list(context_match), float(total_dist / 9)]))
    for mask in ((0, 1), (1, 2), (0, 2)):
        for idx1, entry1 in enumerate(all_repaired_contours[mask[0]]):
            for idx2, entry2 in enumerate(all_repaired_contours[mask[1]]):
                context_match = [entry1, entry2]
                com1biased = entry1["com"] + 0.2*entry1["velocity"]
                com2biased = entry2["com"] + 0.2*entry2["velocity"]

                total_dist = sum(sum((com1biased-com2biased)**2))
                matched_dists.append(tuple([list(context_match), float(total_dist)]))


    matched_dists.sort(key=lambda e: e[1])
    # print([e[1] for e in matched_dists])
    already_processed = []
    new_matched = []


    for matches in matched_dists:
        skip = False
        for m in matches[0]:
            for p in already_processed:
                if p.shape == m["contour"].shape or np.array_equal(m["contour"], p):
                    skip = True
        if skip or matches[1] > 10000:
            continue
        
        context_matched = []
        for m in matches[0]:
            already_processed.append(m["contour"])
            context_matched.append(m)
        if skip:
            continue
        new_matched.append(context_matched)

    meta[0]["matches"] = new_matched
    # print([[i["com"] for i in j] for j in new_matched])
        
    return frames, meta

def superresolution(frames, meta):
    if "matches" not in meta[0].keys():
        return frames, meta 

    # len(meta[0]["matches"]) == 0

    if "super frames" in meta[0].keys():
        meta[0]["super frames"] = []
    
    matches = meta[0]["matches"]
    supered_matches = []
    super_backgrounds = superres([cv2.cvtColor(meta[i]["original"], cv2.COLOR_RGB2RGBA) for i in range(3)])
    # super_backgrounds = [np.zeros(meta[0]["original"].shape[:2] + (4,)) for i in range(3)]
    super_backgrounds.reverse()
    # print(super_backgrounds)
    for mat in matches:
        master = sorted(mat, key=lambda e: e["area"])
        shape = master[-1]["contour"]
        translate1 = master[-1]["com"]-master[-2]["com"]

        contours = []
        imageidcs = []
        if len(master) == 3:
            translate2 = master[-1]["com"]-master[0]["com"]
            contours = [shape, shape+translate1, shape+translate2]
            imageidcs = [master[i]["origin"] for i in (2, 1, 0)]
        elif len(master) == 2:
            contours = [shape, shape+translate1]
            imageidcs = [master[1]["origin"], master[0]["origin"]]
        #print(contours)
        extract_translated = []
        for idx, m in enumerate(imageidcs):
            #mask = np.zeros_like(meta[m]["original"])
            mask = np.zeros(meta[m]["original"].shape[:2])
            contours[idx] = np.array(contours[idx]).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(mask, contours, idx, 255, -1)
            out = np.zeros(meta[m]["original"].shape[:2] + (4,))
            out[mask==255] = cv2.cvtColor(meta[m]["original"], cv2.COLOR_RGB2RGBA)[mask==255]
            t = [0, 0] if idx == 0 else translate1[0] if idx == 1 else translate2[0]
            translation = np.float32([
                [1, 0, t[0]],
                [0, 1, t[1]]
            ])
            out = cv2.warpAffine(out, translation, (out.shape[1], out.shape[0]))
            extract_translated.append(out)
        result = superres(extract_translated)
        if result != None:
            result.reverse()
            supered_matches.append(tuple([result, translation, master[-1]["velocity"]]))
        #  extracted_matches.append(extract_translated)
    def helper(ndarr):
        return ndarr[:, :, -1]
    for supered in supered_matches:
        sup, trans, vel = supered
        for idx, s in enumerate(sup):
            new_trans = np.float32([
                [1, 0, vel[0] * idx * 5],
                [0, 1, vel[1] * idx * 5]
            ])
            new_sup = cv2.warpAffine(s, new_trans, (s.shape[1], s.shape[0]))
            alphas = dc(new_sup)[:, :, 3]
            new_sup[alphas!=0][3] = 255
            # print(alphas)
            super_backgrounds[idx][alphas!=0] = new_sup[alphas!=0]
            # print(new_sup)
    # print(len(super_backgrounds))
    meta[0]["super frames"] = [cv2.cvtColor(super_back.astype(np.uint8), cv2.COLOR_RGBA2RGB) for super_back in super_backgrounds]

    return frames, meta

# draw contours
def draw_contours(frames, meta):
    for idx, (frame, frame_meta) in enumerate(zip(frames, meta)):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (127, 127, 127), -1)
        if "contour velocities" not in frame_meta.keys() or "repaired contours"not in frame_meta.keys(): 
            continue
        shape = frame.shape
        disps = frame_meta["contour velocities"]
        for cidx, (contour, disp) in enumerate(zip(frame_meta["repaired contours"], frame_meta["repaired contour velocities"])):
            if disp[0] == 0 and disp[1] == 0:
                cv2.drawContours(frame, frame_meta["repaired contours"], cidx, (127, 127, 127), 2)
                continue
            cv2.drawContours(frame, frame_meta["repaired contours"], cidx, (disp[0]/5*127 + 127, disp[1]/5*127 + 127, 127), -1)
            cv2.drawContours(frame, frame_meta["repaired contours"], cidx, (0, 0, 0), 3)
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

def get_data(frames, meta):

    #if meta[0]["frame number"] == 0:
    #    path = "videos/phone3/2.mp4"
    #    offset = 250
    #    meta[0]["ground truth"] = cv2.VideoCapture(path)

    res, fr1 = meta[0]["ground truth"].read()
    res, fr2 = meta[0]["ground truth"].read()

    fr1 = cv2.resize(fr1, None, fx=.5, fy=.5)[:945, :502]
    fr2 = cv2.resize(fr2, None, fx=.5, fy=.5)[:945, :502]

    if "super frames" not in meta[0].keys() or "original" not in meta[0].keys():
        return frames, meta

    starttime = time.time()
    shechtman = superres([meta[i]["original"] for i in range(3)])
    shechtman.reverse()
    shechtman = [cv2.resize(sh.astype(np.uint8), None, fx=.5, fy=.5) for sh in shechtman]
    meta[0]["shechtman"] = shechtman
    endtime = time.time()
    meta[0]["super time"] = endtime-starttime

    shechtman_frame1 = shechtman[0] + shechtman[1] / 2
    shechtman_frame2 = shechtman[1] / 2 + shechtman[2]

    supers = [cv2.resize(sf, None, fx=.5, fy=.5) for sf in meta[0]["super frames"]]
    meta[0]["supers"] = supers

    supers_frame1 = supers[0] + supers[1] / 2
    supers_frame2 = supers[1] / 2 + supers[2]

    shechtman_diff1 = sum(sum(sum((shechtman_frame1 - fr1)**2)))
    shechtman_diff2 = sum(sum(sum((shechtman_frame2 - fr2)**2)))

    supers_diff1 = sum(sum(sum((supers_frame1 - fr1)**2)))
    supers_diff2 = sum(sum(sum((supers_frame2 - fr2)**2)))

    
    meta[0]["shechtman error"] = [shechtman_diff1, shechtman_diff2]
    meta[0]["supers error"] = [supers_diff1, supers_diff2]

    return frames, meta

# show frames
def show(frames, meta):
    for idx, frame in enumerate(frames):
        # show frame
        resized = cv2.resize(frame, None, fx=0.25, fy=0.25)
        cv2.imshow(str(idx), resized)
        resized2 = cv2.resize(meta[idx]["copy"][0], None, fx=0.25, fy=0.25)
        cv2.imshow(str(idx) + "org", resized2)

    if "super frames" not in meta[0].keys():
        return frames, meta
    if "original" not in meta[0].keys():
        return frames, meta

    show_sequential([meta[0]["shechtman"], meta[0]["supers"]], ["shechtman", "supers"])

    # overlay = (frames[0].astype(int) + frames[1].astype(int) + frames[2].astype(int)) // 3
    # overlay = (meta[0]["copy"][0].astype(int) + meta[1]["copy"][0].astype(int) + meta[2]["copy"][0].astype(int)) // 3
    # overlay = overlay.astype(np.uint8)
    # overlay = cv2.resize(overlay, None, fx=0.5, fy=0.5)
    # cv2.imshow("overlay", overlay)
    
    return frames, meta

def write(frames, meta):
    if "shechtman error" not in meta[0].keys():
        return frames, meta
    with open("shechtmans.csv", "a") as f:
        f.write(str(meta[0]["shechtman error"][0]) + "," + str(meta[0]["shechtman error"][1]) + ",")
    f.close()
    with open("shechtmant.csv", "a") as f:
        f.write(str(meta[0]["super time"]) + ",")
    f.close()
    with open("supers.csv", "a") as f:
        f.write(str(meta[0]["supers error"][0]) + "," + str(meta[0]["supers error"][1]) + ",")
    f.close()
    with open("supert.csv", "a") as f:
        f.write(str(time.time()-meta[0]["start time"]) + ",")
    f.close()
    with open("results.csv", "a") as f:
        f.write(str(meta[0]["shechtman error"][0]-meta[0]["supers error"][0]) + "," + str(meta[0]["shechtman error"][1]-meta[0]["supers error"][1]) + ",")
    f.close()
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


offsets = [
        [187, 278, 396],
        [189, 282, 327],
        [260, 353, 381],
        [142, 218, 236],
        [102, 184, 203],
        [142, 229, 273],
        [125, 201, 226],
        [81, 148, 172],
        [57, 114, 128],
        [142, 82, 146],
        [121, 185, 205],
        [135, 192, 227],
        [56, 113, 156],
        [31, 91, 121],
        [173, 230, 258],
        [114, 191, 214],
        [158, 215, 247],
        [42, 97, 118],
        [109, 196, 224],
        [34, 90, 104],
        [40, 110, 134],
        [292, 357, 378],
]
current_vid = 0
def state(frames, meta):
    if meta[0]["frame number"] == 0:
        off = (offsets[current_vid][0] + offsets[current_vid][1]) // 2
        file = f"videos/phone3/{current_vid+1}.mp4"
        meta[0]["ground truth"] = cv2.VideoCapture(file)
        meta[0]["ground truth"].set(cv2.CAP_PROP_POS_FRAMES, off)
    return frames, meta

if __name__ == "__main__":
    processing_order = [
            crop,
            gray, 
            copy_frame, 
            sharpen, 
            subtract,
            contour,
            features,
            # kmeans,
            optical_flow,
            velocity,
            pair_contours,
            combine_pairs,
            match_contours,
            superresolution,
            color, 
            draw_contours,
            draw_features,
            # draw_kmeans,
            draw_of,
            state,
            get_data,
            # show, 
            write,
            store_copy
    ]
    # processing_order.append(print_meta)
    # processing_order.append(print_frames)
    for idx in range(22):
        args = lambda i: [[f"videos/phone{str(j)}/{str(i+1)}.mp4" for j in (1,2,4)], offsets[i]]
        current_vid = idx
        start_time = time.time()
        every_frame(*args(idx), processing_order, debug=True)
        with open("results.csv", "a") as f:
            f.write("\n")
        f.close()
        with open("supers.csv", "a") as f:
            f.write("\n")
        f.close()
        with open("shechtmans.csv", "a") as f:
            f.write("\n")
        f.close()
        with open("supert.csv", "a") as f:
            f.write("\n")
        f.close()
        with open("shechtmant.csv", "a") as f:
            f.write("\n")
        f.close()
        end_time = time.time()
        total_time = end_time-start_time

