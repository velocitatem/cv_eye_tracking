"""
This is a program that uses a webcam to get frames of a calssroomk, it detects faces. At each frame f we extract all the faces and for each face we detect the eyes.
We aim to detect the direction of the eyes. The goal is to track participants in a classroom and detect if they are paying attention to the teacher or not.
"""


import cv2
import numpy as np
import urllib.request
import os
import mediapipe as mp

class Classroom:
    def __init__(self):
        self.haar_eye_url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/haarcascade_eye.xml"
        self.haar_face_url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/haarcascade_frontalface_default.xml"

        # Download cascades if not already present
        if not os.path.exists("haarcascade_eye.xml"):
            urllib.request.urlretrieve(self.haar_eye_url, "haarcascade_eye.xml")
        if not os.path.exists("haarcascade_face.xml"):
            urllib.request.urlretrieve(self.haar_face_url, "haarcascade_face.xml")

        # Load cascades
        self.face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.cap = cv2.VideoCapture(0)

        # Set up MediaPipe Face Mesh with iris refinement enabled.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def analyze_attention(self, face_img):
        """
        Analyze the cropped face image using MediaPipe Face Mesh to determine if
        the participant is paying attention. This is done by computing the relative
        position of the iris centers in the eye bounding boxes.

        Returns:
            True if both eyes appear to be looking forward, False otherwise.
        """
        # Convert face image from BGR to RGB as required by MediaPipe.
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(face_rgb)
        if not results.multi_face_landmarks:
            return False

        # Use the first detected face.
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = face_img.shape

        def to_pixel(landmark):
            return np.array([landmark.x * w, landmark.y * h])

        # Define approximate eye regions using landmark indices.
        # Right eye (appears on left side): landmarks 33 and 133.
        right_eye_indices = [33, 133]
        # Left eye (appears on right side): landmarks 362 and 263.
        left_eye_indices = [362, 263]

        right_eye_pts = [to_pixel(landmarks[i]) for i in right_eye_indices]
        left_eye_pts = [to_pixel(landmarks[i]) for i in left_eye_indices]

        # Determine bounding boxes for each eye.
        right_box_left = min(pt[0] for pt in right_eye_pts)
        right_box_right = max(pt[0] for pt in right_eye_pts)
        left_box_left = min(pt[0] for pt in left_eye_pts)
        left_box_right = max(pt[0] for pt in left_eye_pts)

        right_box_width = right_box_right - right_box_left
        left_box_width = left_box_right - left_box_left

        # Check if iris landmarks are available (expected total landmarks >= 478 with iris refinement).
        if len(landmarks) < 478:
            return False

        # Compute iris centers using refined iris landmarks.
        # Right eye iris: landmarks 468 to 472.
        right_iris_pts = [to_pixel(landmarks[i]) for i in range(468, 473)]
        # Left eye iris: landmarks 473 to 477.
        left_iris_pts = [to_pixel(landmarks[i]) for i in range(473, 478)]

        right_iris_center = np.mean(right_iris_pts, axis=0)
        left_iris_center = np.mean(left_iris_pts, axis=0)

        # Calculate the relative horizontal positions within the bounding boxes.
        right_relative = (right_iris_center[0] - right_box_left) / (right_box_width + 1e-6)
        left_relative = (left_iris_center[0] - left_box_left) / (left_box_width + 1e-6)

        # If iris centers are near the middle (threshold between 0.35 and 0.65), mark as paying attention.
        if 0.35 < right_relative < 0.65 and 0.35 < left_relative < 0.65:
            return True
        else:
            return False

    def run(self):
        while True:
            ret, img = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for face in faces:
                x, y, w, h = face
                face_img = img[y:y+h, x:x+w]
                attention = self.analyze_attention(face_img)
                label = "Attention" if attention else "Not Attention"
                color = (0, 255, 0) if attention else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("Classroom", img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    classroom = Classroom()
    classroom.run()
