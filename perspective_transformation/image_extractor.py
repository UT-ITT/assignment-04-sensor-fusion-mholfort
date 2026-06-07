import cv2
import numpy as np
import sys

if (len(sys.argv) != 5):
    print("Enter correct command line parameters: ")
    print("python image_extractor.py input.png output.png width height")
    print("")
    exit()

image_name = sys.argv[1]
output_path = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])

#load image
image = cv2.imread(image_name)

if image is None:
    print("Image not found")
    exit()

points = []

def mouse_callback(event, x, y, flags, param):
    global points, point_selection

    if (event == cv2.EVENT_LBUTTONDOWN and len(points) < 4):
        points.append((x, y))


while True:

    copy_image = image.copy()

    for point in points:
        cv2.circle(copy_image, point, 5, (16, 16, 188), -1)

    cv2.imshow("Image Extractor", copy_image)
    key = cv2.waitKey(1)

    if(key == 27):
        if(len(points) == 0):
            break
        elif(len(points) == 4):
            cv2.destroyWindow("Result")
            points.clear()
        else:
            points.clear()
    

    cv2.setMouseCallback("Image Extractor", mouse_callback)

    if(len(points) == 4):

        dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        src_points = np.array(points, dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        warped = cv2.warpPerspective(image, matrix, (width, height))
        cv2.imshow("Result", warped)

        if(key == ord("s")):
            print("Saved Image")
            cv2.imwrite(output_path, warped)

        
cv2.destroyAllWindows()