"""
This is a program that uses a webcam to get frames of a calssroomk, it detects faces. At each frame f we extract all the faces and for each face we detect the eyes.
We aim to detect the direction of the eyes. The goal is to track participants in a classroom and detect if they are paying attention to the teacher or not.
"""


import cv2
import numpy as np
import urllib.request
import os
import mediapipe as mp
class CV:
    def __init__(self, faces_method="haar", eyes_method="haar"):
        """
        Initialize the OpenCV face and eye detectors.
        Parameters:
            faces_method: The method to use for face detection. Options are "haar" and "mediapipe".
            eyes_method: The method to use for eye detection. Options are "haar" and "mediapipe".
        """
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
        if 'mediapipe' in [faces_method, eyes_method]:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        self.eyes_method = eyes_method
        self.faces_method = faces_method
        self.cache = []


    def save_observation(self, face_image, attention_level):
        """
        Save an observation to disk.
        Parameters:
            face_image: The face image to save.
            attention_level: The attention level of the participant.
        """
        self.cache.append((face_image, attention_level))
        import pickle
        if len(self.cache) > 10:
            print("Saving observations to disk...")
            last = self.load_observations() or []
            print(f"Loaded {len(last)} observations.")
            complete = last + self.cache
            print(f"Total observations: {len(last)}")
            with open("observations.pkl", "wb") as f:
                pickle.dump(complete, f)
            self.cache = []

    def load_observations(self):
        """
        Load observations from disk.
        """
        import pickle
        if os.path.exists("observations.pkl"):
            with open("observations.pkl", "rb") as f:
                cache = pickle.load(f)
        else:
            cache = []
        return cache




    def find_faces(self, frame):
        faces = []
        if self.faces_method == "mediapipe":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            if not results.multi_face_landmarks:
                return faces
            for face_landmarks in results.multi_face_landmarks:
                x = min(landmark.x for landmark in face_landmarks.landmark)
                y = min(landmark.y for landmark in face_landmarks.landmark)
                w = max(landmark.x for landmark in face_landmarks.landmark) - x
                h = max(landmark.y for landmark in face_landmarks.landmark) - y
                faces.append((int(x * frame.shape[1]), int(y * frame.shape[0]), int(w * frame.shape[1]), int(h * frame.shape[0])))
            return faces
        if self.faces_method == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces


    def find_eyes(self, frame, face):
        """
        Detects eyes within a face region using either Haar cascade or MediaPipe.

        Parameters:
            frame: The full frame from the camera
            face: Tuple (x, y, w, h) representing the face region

        Returns:
            List of eyes as (x, y, w, h) tuples
        """
        x, y, w, h = face
        face_roi = frame[y:y+h, x:x+w]

        if self.eyes_method == "haar":
            # Convert the face region to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)

            # Filter and sort the eyes by area (to get the two largest)
            if len(eyes) > 0:
                # Sort by area in descending order
                eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

                # Adjust coordinates to be relative to the full frame
                for i in range(len(eyes)):
                    ex, ey, ew, eh = eyes[i]
                    eyes[i] = (x + ex, y + ey, ew, eh)

                # Sort left to right if we have two eyes
                if len(eyes) == 2:
                    eyes = sorted(eyes, key=lambda e: e[0])

                return eyes
            return []

        elif self.eyes_method == "mediapipe":
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            eyes = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # https://gist.github.com/Asadullah-Dal17/fd71c31bac74ee84e6a31af50fa62961
                    left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    left_iris_indices = [474, 475, 476, 477]

                    right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_iris_indices = [469, 470, 471, 472] # this comes from
                    # the follwoing:

                    # Extract left eye coordinates
                    left_eye_points = [(int(face_landmarks.landmark[idx].x * frame.shape[1]),
                                       int(face_landmarks.landmark[idx].y * frame.shape[0]))
                                       for idx in left_eye_indices]

                    # Extract right eye coordinates
                    right_eye_points = [(int(face_landmarks.landmark[idx].x * frame.shape[1]),
                                        int(face_landmarks.landmark[idx].y * frame.shape[0]))
                                        for idx in right_eye_indices]

                    # maybe we can optimize by using matrices??
                    left_x = min(pt[0] for pt in left_eye_points)
                    left_y = min(pt[1] for pt in left_eye_points)
                    left_w = max(pt[0] for pt in left_eye_points) - left_x
                    left_h = max(pt[1] for pt in left_eye_points) - left_y
                    right_x = min(pt[0] for pt in right_eye_points)
                    right_y = min(pt[1] for pt in right_eye_points)
                    right_w = max(pt[0] for pt in right_eye_points) - right_x
                    right_h = max(pt[1] for pt in right_eye_points) - right_y
                    padding = 5
                    left_x = max(0, left_x - padding)
                    left_y = max(0, left_y - padding)
                    left_w += padding * 2
                    left_h += padding * 2
                    right_x = max(0, right_x - padding)
                    right_y = max(0, right_y - padding)
                    right_w += padding * 2
                    right_h += padding * 2

                    # Add the eyes to the result
                    eyes.append((left_x, left_y, left_w, left_h))
                    eyes.append((right_x, right_y, right_w, right_h))

                    # Store iris landmarks for attention measurement
                    self.left_iris = [(int(face_landmarks.landmark[idx].x * frame.shape[1]),
                                      int(face_landmarks.landmark[idx].y * frame.shape[0]))
                                      for idx in left_iris_indices]

                    self.right_iris = [(int(face_landmarks.landmark[idx].x * frame.shape[1]),
                                       int(face_landmarks.landmark[idx].y * frame.shape[0]))
                                       for idx in right_iris_indices]

            return eyes

        return []


    def measure_attention(self, frame, face, eyes):
        """
        This returns a measure of attention based on the position of the eyes and face.
        If they are looking at the camera, the measure is high.
        If they are looking away, the measure is low.

        Parameters:
            frame: The full frame from the camera
            face: Tuple (x, y, w, h) representing the face region
            eyes: List of eye regions as (x, y, w, h) tuples

        Returns:
            Float in range [0, 1] representing attention level
        """
        if not eyes or len(eyes) < 2:
            return 0.0

        if self.eyes_method == "haar":
            fx, fy, fw, fh = face
            face_roi = frame[fy:fy+fh, fx:fx+fw]
            attention_scores = []

            for eye in eyes:
                ex, ey, ew, eh = eye
                # Adjust coordinates to be relative to the face ROI
                rel_ex, rel_ey = ex - fx, ey - fy

                # Extract the eye region
                eye_roi = face_roi[rel_ey:rel_ey+eh, rel_ex:rel_ex+ew]

                if eye_roi.size == 0:
                    continue

                # Convert to grayscale and threshold to find pupil
                eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)

                # Divide the eye horizontally and compare the white pixel distribution
                left_half = thresh[:, :ew//2]
                right_half = thresh[:, ew//2:]

                left_sum = np.sum(left_half)
                right_sum = np.sum(right_half)
                total_sum = left_sum + right_sum

                if total_sum == 0:
                    attention_scores.append(0.5)  # Neutral if no data
                else:
                    # Calculate balance ratio
                    ratio = min(left_sum, right_sum) / max(left_sum, right_sum) if max(left_sum, right_sum) > 0 else 0

                    # If evenly distributed (ratio close to 1), likely looking at camera
                    attention_scores.append(ratio)

            if not attention_scores:
                return 0.0

            return sum(attention_scores) / len(attention_scores)

        elif self.eyes_method == "mediapipe":
            if not hasattr(self, 'left_iris') or not hasattr(self, 'right_iris'):
                return 0.0

            # Calculate the center of each iris
            if self.left_iris and self.right_iris:
                left_center_x = sum(p[0] for p in self.left_iris) / len(self.left_iris)
                left_center_y = sum(p[1] for p in self.left_iris) / len(self.left_iris)

                right_center_x = sum(p[0] for p in self.right_iris) / len(self.right_iris)
                right_center_y = sum(p[1] for p in self.right_iris) / len(self.right_iris)

                # Get eye bounding boxes
                left_eye = eyes[0]
                right_eye = eyes[1]

                # Calculate the relative position of iris in each eye (0-1 range)
                left_eye_rel_x = (left_center_x - left_eye[0]) / left_eye[2]
                right_eye_rel_x = (right_center_x - right_eye[0]) / right_eye[2]

                # Calculate vertical position
                left_eye_rel_y = (left_center_y - left_eye[1]) / left_eye[3]
                right_eye_rel_y = (right_center_y - right_eye[1]) / right_eye[3]

                # For horizontal attention: when looking straight ahead, iris should be centered (0.5)
                # Deviation from center indicates looking away
                h_attention = 1.0 - min(
                    abs(left_eye_rel_x - 0.5) + abs(right_eye_rel_x - 0.5),
                    1.0
                )

                # For vertical attention: iris shouldn't be too high or too low
                v_attention = 1.0 - min(
                    abs(left_eye_rel_y - 0.5) + abs(right_eye_rel_y - 0.5),
                    1.0
                )

                # Combine horizontal and vertical attention
                return (h_attention * 0.7 + v_attention * 0.3)

        return 0.5  # Default neutral value


    def step(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        faces = self.find_faces(frame)
        for face in faces:
            x, y, w, h = face
            face_img = frame[y:y+h, x:x+w]
            eyes = self.find_eyes(frame, face) or []
            for eye in eyes:
                x, y, w, h = eye
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            attention = self.measure_attention(frame, face, eyes)
            # make rgb image
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            self.save_observation(face_img, attention)
            cv2.putText(frame, f"Attention: {attention:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Classroom", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            self.cap.release()
            cv2.destroyAllWindows()
            return



if __name__ == "__main__":
    eyes = "mediapipe"
    faces = "mediapipe"
    setup = CV(faces, eyes)
    while True:
        setup.step()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    setup.cap.release()
