import cv2
import numpy as np

def detect_and_compute(image, sift):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(des1, des2, matcher):
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  
            good_matches.append(m)
    return good_matches

def stabilize_video(input_video_path, output_video_path):
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video")
        return

    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detect_and_compute(first_frame_gray, sift)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp2, des2 = detect_and_compute(frame_gray, sift)

        good_matches = match_keypoints(des1, des2, matcher)

        points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

      
        if len(good_matches) >= 3:  
            matrix, inliers = cv2.estimateAffine2D(points2, points1)
            stabilized_frame = cv2.warpAffine(frame, matrix, (frame_width, frame_height))

            out.write(stabilized_frame)
        else:
            print("Not enough matches to estimate affine transform")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_video_path = 'apple.mp4'
output_video_path = 'apple_stabilized.avi'

stabilize_video(input_video_path, output_video_path)
