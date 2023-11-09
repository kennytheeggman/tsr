import cv2

if __name__ == "__main__":
    slow = cv2.VideoCapture("slow seq.mp4")
    fast = cv2.VideoCapture("fast seq.mp4")

    initial_frame = 19
    multiplier = 10
    frame_blur_num = 8

    slow.set(cv2.CAP_PROP_POS_FRAMES, initial_frame * 10+6)
    fast.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

    while slow.isOpened() and fast.isOpened():
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        slow_frame = None
        for i in range(multiplier):
            slow_ret, slow_sub_frame = slow.read()
            if slow_ret:
                slow_sub_frame = cv2.resize(slow_sub_frame, (400, 800))
                if i == 0:
                    slow_frame = cv2.multiply(slow_sub_frame, (1/frame_blur_num, 1/frame_blur_num, 1/frame_blur_num, 1/frame_blur_num))
                elif i in tuple([8, 9]):
                    continue
                else:
                    slow_frame = cv2.add(slow_frame, cv2.multiply(slow_sub_frame, (0.1, 0.1, 0.1, 0.1)))
            else:
                break

        cv2.imshow("Slow", slow_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fast_ret, fast_frame = fast.read()
        fast_frame = cv2.resize(fast_frame, (400, 800))
        if fast_ret:
            cv2.imshow("Fast", fast_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    slow.release()
    fast.release()

    cv2.destroyAllWindows()


