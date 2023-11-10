import cv2

cap1 = cv2.VideoCapture("BasketBall_Input_14.avi")
cap2 = cv2.VideoCapture("BasketBall_Input_30.avi")
cap3 = cv2.VideoCapture("BasketBall_Input_5.avi")
# Check if camera opened successfully

counter = 0
frame_num = 0

# Read until video is completed
while cap1.isOpened():
    counter+=1
    counter%=3
    # Capture frame-by-frame
    feed = cap1 if counter == 1 else cap2 if counter == 2 else cap3
    if counter == 0:
        frame_num += 25
    ret, frame = feed.read()
    feed.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    if not ret:
        break
    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap1.release()
cap2.release()
cap3.release()

# Closes all the frames
cv2.destroyAllWindows()