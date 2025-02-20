import cv2
import numpy as np
import urllib.request
import os

# this is a traditional appraoch with haardcascades
haar_eye = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/haarcascade_eye.xml"
haar_face = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/haarcascade_frontalface_default.xml"

if not os.path.exists("haarcascade_eye.xml"):
    urllib.request.urlretrieve(haar_eye, "haarcascade_eye.xml")
if not os.path.exists("haarcascade_face.xml"):
    urllib.request.urlretrieve(haar_face, "haarcascade_face.xml")


face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')



cap = cv2.VideoCapture(0)



def process_face(img, face):
    x, y, w, h = face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes_detected = eye_cascade.detectMultiScale(roi_gray)
    # we might also detect nosdrils
    # we get the pair with the two biggest areas
    compute_area= lambda x: x[2] * x[3]
    eyes_detected = sorted(eyes_detected, key=compute_area, reverse=True)[:2]
    if len(eyes_detected) < 2:
        return
    def process_eye(eye):
        ex, ey, ew, eh = eye
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        origin_points= np.float32([[ex, ey], [ex+ew, ey], [ex, ey+eh], [ex+ew, ey+eh]])
        # target poitns and we use a perspective transformation
        # we need to get the eye in a square
        target_points = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])
        # apply the transformation
        M = cv2.getPerspectiveTransform(origin_points, target_points)
        dst = cv2.warpPerspective(roi_color, M, (100, 100))
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        # if the white distrubution is higher on the left the person is looking to the right
        # if the white distrubution is higher on the right the person is looking to the left
        distribution = dst[:, :, 2]
        distribution = cv2.resize(distribution, (100, 100))
        cv2.line(distribution, (50, 0), (50, 100), (0, 0, 255), 2)
        left_sum = np.sum(distribution[:, :50])
        right_sum = np.sum(distribution[:, 50:])
        print(left_sum, right_sum)
        if left_sum > right_sum:
            distribution[:, :50] = 255
        else:
            distribution[:, 50:] = 255

        cv2.imshow('distribution', distribution)

        cv2.waitKey(1)

        return dst
    get_origin = lambda x: x[0] + x[1]
    left_eye, right_eye = sorted(eyes_detected, key=get_origin)
    left_eye = process_eye(left_eye)
    right_eye = process_eye(right_eye)



    # a classical problem now, we might get the eye and an open mouth so our bounding boxes
    # so the verticals of each bounding box should not fall in the same range
    # verticals will be box_0_left, box_0_right, box_1_left, box_1_right TODO

    cv2.imshow('img', img)
    cv2.waitKey(1)



while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5) # https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    if len(faces_detected) == 0:
        continue # no faces detected
    biggest_face = float('-inf')
    biggest_face_index = 0
    for i, face in enumerate(faces_detected):
        x, y, w, h = face
        area = w * h
        if area > biggest_face:
            biggest_face = area
            biggest_face_index = i
    face = faces_detected[biggest_face_index]
    process_face(img, face)
