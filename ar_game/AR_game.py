import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import time
import random

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(0)

last_src_points = None

prev_gray = None

last_finger_pos = None
frames = 0
detected_finger = None

targets = []
score = 0
last_spawn = 0
game_initialized = False

def init_game(w, h):
    global targets, score, last_spawn

    targets = []
    score = 0
    last_spawn = time.time()


def update_game(finger_pos, w, h, frame):
    global targets, score, last_spawn

    now = time.time()

    #new targets
    if(now - last_spawn > 0.8):
        targets.append({
            "x": random.randint(50, w - 50),
            "y": 0,
            "r": random.randint(15, 30),
            "speed": random.uniform(2, 5)
        })
        last_spawn = now

    new_targets = []

    for t in targets:
        t["y"] += t["speed"]

        #draw
        cv2.circle(frame, (int(t["x"]), int(t["y"])), t["r"], (0, 0, 255), -1)

        hit = False

        #collision
        if(finger_pos is not None):
            fx, fy = finger_pos
            dist = np.sqrt((fx - t["x"])**2 + (fy - t["y"])**2)

            if(dist < t["r"] + 15):
                score += 1
                hit = True
            
        if(t["y"] < h + 50 and not hit):
            new_targets.append(t)

    targets[:] = new_targets

    #score
    cv2.putText(frame, f"Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


while True:

    ret, frame = cap.read()

    if not ret:
        continue

    #reset marker positions
    marker_positions = []

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    # Check if marker is detected
    if ids is not None:

        for i in range(len(ids)):
            c = corners[i][0]

            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))

            marker_positions.append((cx, cy))

        #arrange markers
        if(len(marker_positions) >= 4):

            pts = np.array(marker_positions, dtype=np.float32)

            pts = pts[np.argsort(pts[:, 1])]
            
            top = pts[:2]
            bottom = pts[2:]

            top = top[np.argsort(top[:, 0])]
            bottom = bottom[np.argsort(bottom[:, 0])]

            top_left = top[0]
            top_right = top[1]
            bottom_left = bottom[0]
            bottom_right = bottom[1]

            src_points = np.array([
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ], dtype=np.float32)

            if(last_src_points is None):
                last_src_points = src_points.copy()
            else:
                alpha = 0.8
                last_src_points = (alpha * last_src_points + (1 - alpha) * src_points)

    #warp image
    if last_src_points is not None:
                
        h, w = frame.shape[:2]
        
        #initialize game
        if not game_initialized: 
            init_game(w, h)
            game_initialized = True

        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(last_src_points.astype(np.float32), dst_points)
        warped = cv2.warpPerspective(frame, matrix, (w, h))

        #mirror
        warped = cv2.flip(warped, 1)

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        #track finger
        if(prev_gray is None):
            prev_gray = gray
            finger_pos = None

        else:
            diff = cv2.absdiff(prev_gray, gray)

            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)

            thresh = cv2.erode(thresh, kernel, iterations=1)
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_finger = None

            if(contours):
                largest = max(contours, key=cv2.contourArea)

                if(cv2.contourArea(largest) > 800):
                    hull = cv2.convexHull(largest)
                    fingertip = tuple(hull[hull[:, :, 1].argmin()][0])

                    detected_finger = fingertip


            if detected_finger is not None:
                last_finger_pos = detected_finger
                frames = 0
            else:
                frames += 1


            if(frames < 10):
                finger_pos = last_finger_pos
            else:
                finger_pos = None

            prev_gray = gray


        update_game(finger_pos, w, h, warped)

        cv2.imshow("AR Game", warped)

    else:
        cv2.imshow("AR Game", frame)


    # Wait for a key press and check if it's the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()