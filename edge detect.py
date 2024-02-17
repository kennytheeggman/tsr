import numpy as np
import cv2

def is_valid_center(c, frame):
    return 0 < c[1] < frame.shape[0] and 0 < c[3] < frame.shape[1]

if __name__ == "__main__":
    vid = cv2.VideoCapture("videos/seq 34 vid 2.mp4")
    old_frame = cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2GRAY)
    # old_frame = cv2.Laplacian(old_frame, -1)*10

    #of = cv2.cuda_NvidiaOpticalFlow_1_0.create(old_frame.shape[1], old_frame.shape[0], 5, False, False, False, 0)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        npabs = np.vectorize(abs)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)
        corners = cv2.goodFeaturesToTrack(frame, 200, 0.01, 10)
       
        image = frame.copy()

        # frame = npabs(frame-old_frame)
        # image = frame.copy()
        # frame = cv2.Laplacian(frame, -1) * 10
        
        # flow = cv2.calcOpticalFlowFarneback(old_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # f = flow[:, :, 1].astype(np.uint8)
        # cv2.imshow("test2", f)

        segments = cv2.Canny(frame, 50, 180).astype(np.int32) * -1
        segments[segments==-255] = -1
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, image, corners, None, winSize=(15, 15), maxLevel=2)
        # ret, frame = cv2.threshold(frame, 5, 10, cv2.THRESH_BINARY)
        #dist = cv2.distanceTransform(frame, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        #dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX) 
        #dist_output = cv2.cvtColor(dist_output, cv2.COLOR_GRAY2BGR)
        
        old_frame = image.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
 
        displacements = p1[:, 0, :] - corners[:, 0, :]
        coordinates = corners[:, 0, :]

        stacked = np.stack((displacements, coordinates), 2)
        #for c in corners:
        #    cv2.circle(frame, (int(c[0, 0]), int(c[0, 1])), 5, (0, 0, 255), -1)
        compact, labels, centers = cv2.kmeans(stacked, 10, stacked.copy(), (cv2.TERM_CRITERIA_EPS, 1000, 1), 100, cv2.KMEANS_PP_CENTERS)
        
        #segments = np.zeros(old_frame.shape, dtype=np.int32)

        for i, p in enumerate(p1):
            if st[i] == 1:
                cv2.circle(frame, (int(corners[i][0, 0]), int(corners[i][0, 1])), 5, (0, 0, 255), -1)
                segments[int(corners[i][0, 1]), int(corners[i][0, 0])] = -1
                cv2.line(frame, (int(corners[i][0, 0]), int(corners[i][0, 1])), (int(p[0, 0]), int(p[0, 1])), (0, 255, 0), 2)
                cv2.circle(frame, (int(p[0, 0]), int(p[0, 1])), 5, (0, 255, 0), -1)
            if i < len(centers) and is_valid_center(centers[i], frame):
                cv2.circle(frame, (int(centers[i,1]), int(centers[i,3])), 5, (255, 0, 0), -1)
                cv2.circle(segments, (int(centers[i,1]), int(centers[i,3])), 5, i, -1)
                #segments[int(centers[i,1]), int(centers[i,3])] = i

        lmax = max(labels)
        # print(len(p1))
        # print(len(labels))
        for i in range(lmax[0]):
            displacement = [centers[i,0], centers[i,2]]
            if displacement[0]**2 + displacement[1]**2 > 36:
                to_hull_points = corners[labels==i]
                # print(to_hull_points)
                hull_points = cv2.convexHull(to_hull_points, None, True)
                hull_points = hull_points
                com = [centers[i,1], centers[i,3]]
                scaled = []
                for point in hull_points:
                    norm = point - com
                    norm = norm * 1.5
                    norm = com + norm
                    norm = norm.astype(np.int32)
                    scaled.append(norm)
                if len(hull_points) > 0:
                    cv2.drawContours(frame, [np.array(scaled)], 0, (255, 255, 0), -1)
                cv2.putText(frame, str(i), (int(com[0]), int(com[1])), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2)
        
        f2 = segments.copy()
        f2[segments==-1] = 100
        f2[segments>0] = 200
        markers = cv2.watershed(frame, segments)
        frame[markers==-1] = [0, 255, 255]
        f2[markers==-1] = 255
        cv2.imshow("test1", f2.astype(np.uint8))
       #print(centers)
        #for i, c in enumerate(centers):
        #    cv2.circle(frame, (int(c[2]), int(c[3])), 5, (255, 0, 0), -1)
        #    print(labels)

        cv2.imshow("test", frame)
        cv2.waitKey(1)

