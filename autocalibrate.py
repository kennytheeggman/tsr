import cv2
import numpy as np

def set_frame(caps, nums):
    for cap, num in zip(caps, nums):
        cap.set(cv2.CAP_PROP_POS_FRAMES, num)

def pair_error(frame1, frame2):
    area = frame1.shape[0] * frame1.shape[1]
    diff = (frame1 - frame2)**2
    return sum(sum(diff)) / area

def total_error(frames):
    return pair_error(frames[0], frames[1]) + pair_error(frames[0], frames[2])

def crop(frames, offsets, voffsets=[0,0]):
    height, width = frames[0].shape
    off = max(offsets)
    voff = max(voffsets)
    cropped_width = width-off
    cropped_height = height-voff
    new_f1 = frames[0][voff-voffsets[0]:voff-voffsets[0]+cropped_height, off:width]
    new_f2 = frames[1][voff-voffsets[1]:voff-voffsets[1]+cropped_height, off-offsets[0]:off-offsets[0]+cropped_width]
    new_f3 = frames[2][voff:height, off-offsets[1]:off-offsets[1]+cropped_width]
    return [new_f1, new_f2, new_f3]

def offset_error(frames, offsets, voffsets=[0, 0]):
    cropped = crop(frames, offsets, voffsets)
    return total_error(cropped)

def show_offset(frames, offsets, voffsets=[0, 0]):
    nf = crop(frames, offsets, voffsets)
    final_frame = (nf[0].astype(int) + nf[1].astype(int) + nf[2].astype(int)) // 3
    final_frame = final_frame.astype(np.uint8)
    cv2.imshow("1", cv2.resize(final_frame, None, fx=0.5, fy=0.5))

if __name__ == "__main__":
    caps = [cv2.VideoCapture(f"videos/phone{i}/12.mp4") for i in (1, 2, 4)]
    nums = [56, 113, 156]
    set_frame(caps, nums)
    frames = [caps[i].read()[1] for i in range(3)]
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]

    min_running_err = 1e12
    min_running_offset = [0, 0]
    voffset = [0, 20]
    width = frames[0].shape[1]
    """
    for i in range(30):
        for j in range(30):
            offset = [i, j]
            err = offset_error(frames, offset)
            if err < min_running_err:
                min_running_err = err
                min_running_offset = offset
                print(min_running_offset)
    # """
    print(offset_error(frames, [22, 26]))
   
    curr_h_offset = [0, 0]
    curr_v_offset = [0, 0]
    while True:
        cmd = input()
        if cmd == "q":
            break
        if cmd == "r":
            print(curr_h_offset, curr_v_offset)
            continue
        show_offset(frames, curr_h_offset, curr_v_offset)
        cv2.waitKey(1)
        if cmd == "c":
            continue
        value = int(cmd[2:])
        if cmd.startswith("x1"):
            curr_h_offset[0] = value
        if cmd.startswith("x2"):
            curr_h_offset[1] = value
        if cmd.startswith("y1"):
            curr_v_offset[0] = value
        if cmd.startswith("y2"):
            curr_v_offset[1] = value

    print(min_running_offset)
    t_err = total_error(frames)
    print(t_err)
