"""
This is a program that uses a webcam to get frames of a classroom, it detects faces. At each frame f we extract all the faces and for each face we detect the eyes.
We aim to detect the direction of the eyes. The goal is to track participants in a classroom and detect if they are paying attention to the teacher or not.
"""


import cv2
import numpy as np
import urllib.request
import os
import mediapipe as mp
import time
from deepface import DeepFace
from retinaface import RetinaFace

def deepface_find_faces(frame):
    """
    Takes a frame and returns the faces detected in the frame (x, y, w, h) tuples.

    Args:
        frame: A numpy array representing an image (BGR format from OpenCV)

    Returns:
        A list of tuples, where each tuple contains (x, y, w, h) coordinates of a detected face
    """
    from deepface import DeepFace
    # save fram


class CV:
    def __init__(self, faces_method="retina", eyes_method="hough", video_path=None, gui=False):
        """
        Initialize the face and eye detectors.
        Parameters:
            faces_method: The method to use for face detection. Options are:
            [
            'opencv',
            'ssd',
            'dlib',
            'mtcnn',
            'fastmtcnn',
            'retinaface',
            'mediapipe',
            'yolov8',
            'yolov11s',
            'yolov11n',
            'yolov11m',
            'yunet',
            'centerface',
            ]
            eyes_method: The method to use for eye detection. Options are "haar", "mediapipe", "hough".
        """
        self.haar_eye_url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/haarcascade_eye.xml"
        self.haar_face_url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/haarcascade_frontalface_default.xml"
        # Download cascades if not already present
        if not os.path.exists("haarcascade_eye.xml"):
            urllib.request.urlretrieve(self.haar_eye_url, "haarcascade_eye.xml")
        if not os.path.exists("haarcascade_face.xml"):
            urllib.request.urlretrieve(self.haar_face_url, "haarcascade_face.xml")
        self.gui = gui

        # Load cascades with improved parameters
        self.face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        # Initialize video capture
        if video_path:
            format = video_path.split('.')[-1]
            format = cv2.CAP_FFMPEG if format in ['mp4', 'mov'] else cv2.CAP_GSTREAMER if format in ['avi', 'mkv'] else cv2.CAP_ANY if format in ['webm'] else None
            print(format)
            self.cap = cv2.VideoCapture(video_path, format)
        else:
            self.cap = cv2.VideoCapture(0)

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
                print(f"Saving {len(complete)} observations.")
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
        """
        Detects faces in the frame using the selected method.
        Implements a tracking mechanism to maintain face detection between frames.

        Parameters:
            frame: The current frame from the video stream

        Returns:
            List of faces as (x, y, w, h) tuples
        """

        cv2.imwrite("frame.jpg", frame)

        # Get the detector - default is opencv
        detector_backend = self.faces_method

        try:
            face_objs = DeepFace.extract_faces(
                img_path="frame.jpg",
                detector_backend=detector_backend,
                align=True,
            )
            face_objs = [face['facial_area'] for face in face_objs]
            face_objs = [(face['x'], face['y'], face['w'], face['h']) for face in face_objs]
            print(face_objs)
            return face_objs

        except Exception as e:
            print(f"Error in face detection: {e}")
            return []



    def find_eyes(self, frame, face):
        """
        Detects eyes within a face region using state-of-the-art methods.
        Parameters:
            frame: The full frame from the camera
            face: Tuple (x, y, w, h) representing the face region
        Returns:
            List of eyes as (x, y, w, h) tuples
        """
        import cv2
        import numpy as np
        import mediapipe as mp
        import dlib

        x, y, w, h = face
        # Make sure we don't exceed frame boundaries
        y = max(0, y)
        x = max(0, x)
        h = min(h, frame.shape[0] - y)
        w = min(w, frame.shape[1] - x)
        if h <= 0 or w <= 0:
            return []
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return []

        # Method 1: MediaPipe Face Mesh for precise facial landmark detection
        try:
            mp_face_mesh = mp.solutions.face_mesh

            # Convert the RGB image to BGR (MediaPipe uses RGB)
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Initialize Face Mesh with appropriate parameters
            with mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

                # Process the face ROI
                results = face_mesh.process(rgb_face)

                if results.multi_face_landmarks:
                    # Get the first face's landmarks
                    face_landmarks = results.multi_face_landmarks[0]

                    # MediaPipe eye landmarks
                    # Left eye landmarks: 362, 385, 387, 263, 373, 380
                    # Right eye landmarks: 33, 160, 158, 133, 153, 144

                    # Convert normalized coordinates to pixel coordinates
                    h_roi, w_roi, _ = face_roi.shape

                    # Get left eye landmarks
                    left_eye_landmarks = [
                        (int(face_landmarks.landmark[362].x * w_roi), int(face_landmarks.landmark[362].y * h_roi)),
                        (int(face_landmarks.landmark[385].x * w_roi), int(face_landmarks.landmark[385].y * h_roi)),
                        (int(face_landmarks.landmark[387].x * w_roi), int(face_landmarks.landmark[387].y * h_roi)),
                        (int(face_landmarks.landmark[263].x * w_roi), int(face_landmarks.landmark[263].y * h_roi)),
                        (int(face_landmarks.landmark[373].x * w_roi), int(face_landmarks.landmark[373].y * h_roi)),
                        (int(face_landmarks.landmark[380].x * w_roi), int(face_landmarks.landmark[380].y * h_roi))
                    ]

                    # Get right eye landmarks
                    right_eye_landmarks = [
                        (int(face_landmarks.landmark[33].x * w_roi), int(face_landmarks.landmark[33].y * h_roi)),
                        (int(face_landmarks.landmark[160].x * w_roi), int(face_landmarks.landmark[160].y * h_roi)),
                        (int(face_landmarks.landmark[158].x * w_roi), int(face_landmarks.landmark[158].y * h_roi)),
                        (int(face_landmarks.landmark[133].x * w_roi), int(face_landmarks.landmark[133].y * h_roi)),
                        (int(face_landmarks.landmark[153].x * w_roi), int(face_landmarks.landmark[153].y * h_roi)),
                        (int(face_landmarks.landmark[144].x * w_roi), int(face_landmarks.landmark[144].y * h_roi))
                    ]

                    # Calculate the bounding boxes for the eyes
                    l_min_x = min([p[0] for p in left_eye_landmarks])
                    l_max_x = max([p[0] for p in left_eye_landmarks])
                    l_min_y = min([p[1] for p in left_eye_landmarks])
                    l_max_y = max([p[1] for p in left_eye_landmarks])

                    r_min_x = min([p[0] for p in right_eye_landmarks])
                    r_max_x = max([p[0] for p in right_eye_landmarks])
                    r_min_y = min([p[1] for p in right_eye_landmarks])
                    r_max_y = max([p[1] for p in right_eye_landmarks])

                    # Add padding to the eye bounding boxes for better visibility
                    padding = int(min(w, h) * 0.05)  # 5% padding

                    # Left eye bounding box with padding
                    left_eye_x = max(0, l_min_x - padding)
                    left_eye_y = max(0, l_min_y - padding)
                    left_eye_w = min(w_roi - left_eye_x, (l_max_x - l_min_x) + 2 * padding)
                    left_eye_h = min(h_roi - left_eye_y, (l_max_y - l_min_y) + 2 * padding)

                    # Right eye bounding box with padding
                    right_eye_x = max(0, r_min_x - padding)
                    right_eye_y = max(0, r_min_y - padding)
                    right_eye_w = min(w_roi - right_eye_x, (r_max_x - r_min_x) + 2 * padding)
                    right_eye_h = min(h_roi - right_eye_y, (r_max_y - r_min_y) + 2 * padding)

                    # Convert coordinates to be relative to the original frame
                    left_eye = (x + left_eye_x, y + left_eye_y, left_eye_w, left_eye_h)
                    right_eye = (x + right_eye_x, y + right_eye_y, right_eye_w, right_eye_h)

                    return [left_eye, right_eye]

        except Exception as e:
            print(f"MediaPipe face mesh error: {str(e)}")

        # Method 2: dlib's facial landmark detector as a backup
        try:
            # Initialize dlib's face detector and facial landmark predictor
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this model file is available

            # Convert face ROI to grayscale for dlib
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Detect faces in grayscale image
            dlib_faces = detector(gray_face)

            if dlib_faces:
                # Get facial landmarks
                landmarks = predictor(gray_face, dlib_faces[0])

                # Extract eye landmarks (36-41 for left eye, 42-47 for right eye in the 68-point model)
                left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                # Calculate eye bounding boxes
                l_min_x = min([p[0] for p in left_eye_points])
                l_max_x = max([p[0] for p in left_eye_points])
                l_min_y = min([p[1] for p in left_eye_points])
                l_max_y = max([p[1] for p in left_eye_points])

                r_min_x = min([p[0] for p in right_eye_points])
                r_max_x = max([p[0] for p in right_eye_points])
                r_min_y = min([p[1] for p in right_eye_points])
                r_max_y = max([p[1] for p in right_eye_points])

                # Add padding
                padding = int(min(w, h) * 0.05)

                # Create bounding boxes
                left_eye_x = max(0, l_min_x - padding)
                left_eye_y = max(0, l_min_y - padding)
                left_eye_w = min(w_roi - left_eye_x, (l_max_x - l_min_x) + 2 * padding)
                left_eye_h = min(h_roi - left_eye_y, (l_max_y - l_min_y) + 2 * padding)

                right_eye_x = max(0, r_min_x - padding)
                right_eye_y = max(0, r_min_y - padding)
                right_eye_w = min(w_roi - right_eye_x, (r_max_x - r_min_x) + 2 * padding)
                right_eye_h = min(h_roi - right_eye_y, (r_max_y - r_min_y) + 2 * padding)

                # Convert coordinates to be relative to the original frame
                left_eye = (x + left_eye_x, y + left_eye_y, left_eye_w, left_eye_h)
                right_eye = (x + right_eye_x, y + right_eye_y, right_eye_w, right_eye_h)

                return [left_eye, right_eye]

        except Exception as e:
            print(f"dlib landmark detection error: {str(e)}")

        # Method 3: Enhanced Haar Cascade with additional preprocessing
        try:
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_face = clahe.apply(gray_face)

            # Apply bilateral filtering to reduce noise while preserving edges
            filtered_face = cv2.bilateralFilter(enhanced_face, 9, 75, 75)

            # Load the eye cascade classifier
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            # Detect eyes - tweak parameters for better accuracy
            eyes = eye_cascade.detectMultiScale(
                filtered_face,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(int(w*0.1), int(h*0.05)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # If eyes detected, adjust coordinates relative to original frame
            if len(eyes) > 0:
                # Filter out false positives by checking if they're in the upper half of the face
                valid_eyes = []
                for (ex, ey, ew, eh) in eyes:
                    if ey < h * 0.6:  # Eye should be in upper 60% of face
                        valid_eyes.append((x + ex, y + ey, ew, eh))

                # If we have too many eyes, keep only the most likely pair
                if len(valid_eyes) > 2:
                    # Sort by y-coordinate to find eyes that are at similar height
                    valid_eyes.sort(key=lambda e: e[1])

                    # Find the best pair with similar y-coordinates
                    best_pair = None
                    min_y_diff = float('inf')

                    for i in range(len(valid_eyes)-1):
                        y_diff = abs(valid_eyes[i][1] - valid_eyes[i+1][1])
                        if y_diff < min_y_diff:
                            min_y_diff = y_diff
                            best_pair = [valid_eyes[i], valid_eyes[i+1]]

                    if best_pair and min_y_diff < h * 0.1:  # Eyes should be at similar heights
                        # Sort by x-coordinate to identify left and right eyes
                        best_pair.sort(key=lambda e: e[0])
                        return best_pair

                # If we have exactly 2 eyes, check if they make sense as a pair
                elif len(valid_eyes) == 2:
                    valid_eyes.sort(key=lambda e: e[0])  # Sort by x-coordinate
                    left_eye, right_eye = valid_eyes

                    # Check if they're at similar heights
                    if abs(left_eye[1] - right_eye[1]) < h * 0.1:
                        # Check if their horizontal distance makes sense
                        if 0.2 * w < (right_eye[0] - left_eye[0]) < 0.8 * w:
                            return valid_eyes

                # If we have at least one valid eye, return it
                elif valid_eyes:
                    return valid_eyes

        except Exception as e:
            print(f"Enhanced Haar cascade error: {str(e)}")

        # Fallback: Geometric estimation based on facial proportions
        left_eye_x = x + int(w * 0.3)
        right_eye_x = x + int(w * 0.7)
        eyes_y = y + int(h * 0.3)
        eye_width = int(w * 0.15)
        eye_height = int(h * 0.1)

        left_eye = (left_eye_x, eyes_y, eye_width, eye_height)
        right_eye = (right_eye_x, eyes_y, eye_width, eye_height)

        return [left_eye, right_eye]


    def measure_attention(self, frame, face, eyes):
        """
        Calculates an attention score based on eye position and gaze direction.
        If they are looking at the camera, the measure is high.
        If they are looking away, the measure is low.
        Parameters:
            frame: The full frame from the camera
            face: Tuple (x, y, w, h) representing the face region
            eyes: List of eye regions as (x, y, w, h) tuples
        Returns:
            Float in range [0, 1] representing attention level
        """
        import cv2
        import numpy as np

        # If no eyes detected, return minimum attention
        if not eyes or len(eyes) < 2:
            return 0.0

        # Extract face region
        face_x, face_y, face_w, face_h = face
        face_center_x = face_x + face_w // 2
        face_center_y = face_y + face_h // 2

        eyes = sorted(eyes, key=lambda eye: eye[0])
        left_eye, right_eye = eyes[:2]  # Take the first two in case more than 2 were detected

        # Process each eye to detect pupil/iris position
        def process_eye(eye_rect):
            x, y, w, h = eye_rect
            eye_roi = frame[y:y+h, x:x+w].copy()

            # Convert to grayscale if not already
            if len(eye_roi.shape) > 2:
                eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            else:
                eye_gray = eye_roi

            # Apply histogram equalization to enhance contrast
            eye_gray = cv2.equalizeHist(eye_gray)

            # Apply Gaussian blur to reduce noise
            eye_gray = cv2.GaussianBlur(eye_gray, (7, 7), 0)

            # Use adaptive thresholding to isolate the darker regions (pupils)
            _, thresh = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours of potential pupil regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If no contours found, return center of eye as fallback
            if not contours:
                return w // 2, h // 2, False

            # Find the largest contour which is likely to be the pupil
            largest_contour = max(contours, key=cv2.contourArea)

            # Check if the contour is reasonably sized to be a pupil
            contour_area = cv2.contourArea(largest_contour)
            eye_area = w * h
            if contour_area < 0.01 * eye_area or contour_area > 0.5 * eye_area:
                return w // 2, h // 2, False

            # Get the center of the pupil
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return w // 2, h // 2, False

            pupil_x = int(M["m10"] / M["m00"])
            pupil_y = int(M["m01"] / M["m00"])

            # Calculate relative position of pupil in eye
            rel_x = pupil_x / w  # 0 = left edge, 0.5 = center, 1 = right edge
            rel_y = pupil_y / h  # 0 = top edge, 0.5 = center, 1 = bottom edge

            # Eye is looking straight if pupil is near center
            is_centered = (0.35 < rel_x < 0.65) and (0.35 < rel_y < 0.65)

            return pupil_x, pupil_y, is_centered

        # Process both eyes
        left_pupil_x, left_pupil_y, left_centered = process_eye(left_eye)
        right_pupil_x, right_pupil_y, right_centered = process_eye(right_eye)

        # Calculate attention score based on multiple factors

        # 1. Are both pupils detected and centered?
        pupil_centered_score = 0.5 if (left_centered and right_centered) else 0.0

        # 2. Eye symmetry - are both eyes looking in the same direction?
        left_rel_x = left_pupil_x / left_eye[2]
        right_rel_x = right_pupil_x / right_eye[2]
        symmetry_score = 0.3 * (1.0 - min(abs(left_rel_x - right_rel_x) * 3, 1.0))

        # 3. Head orientation based on eye positions
        eye_line_x1 = left_eye[0] + left_pupil_x
        eye_line_y1 = left_eye[1] + left_pupil_y
        eye_line_x2 = right_eye[0] + right_pupil_x
        eye_line_y2 = right_eye[1] + right_pupil_y

        # Calculate angle of eye line
        eye_angle = np.degrees(np.arctan2(eye_line_y2 - eye_line_y1, eye_line_x2 - eye_line_x1))
        head_angle_score = 0.2 * (1.0 - min(abs(eye_angle) / 20.0, 1.0))

        # Combine scores, ensuring output is in [0, 1] range
        attention_score = min(pupil_centered_score + symmetry_score + head_angle_score, 1.0)

        return attention_score


    def step(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        faces = self.find_faces(frame)
        print(f"Found {len(faces)} faces.")
        show_frame = np.copy(frame)
        for face in faces:
            x, y, w, h = face
            face_img = frame[y:y+h, x:x+w]
            cv2.imshow('face', face_img)
            eyes = self.find_eyes(frame, face) or []
            print(f"eyes {eyes}")
            for eye in eyes:
                xe, ye, we, he = eye
                if self.gui:
                    cv2.rectangle(show_frame, (xe, ye), (xe+we, ye+he), (255, 0, 0), 2)
            attention = self.measure_attention(frame, face, eyes)
            # make rgb image
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            self.save_observation(face_img, attention)
            if self.gui:
                cv2.putText(show_frame, f"Attention: {attention:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.rectangle(show_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if self.gui:
            cv2.imshow('frame', show_frame)



if __name__ == "__main__":
    # Use improved methods as default

    faces = "opencv"  # Use RetinaFace for more robust face detection in classroom
    eyes = "hough"    # Use Hough transform for better eye detection

    # For testing, you can uncomment and use the alternative options:
    # faces = "haar"   # Improved Haar cascade
    # eyes = "haar"    # Improved Haar cascade for eyes

    # Initialize with video path if available, or use webcam
    video_path = None
    video_path = "/home/velocitatem/Downloads/IMG_9561.mp4"  # Uncomment to use video file

    # Initialize detection system with GUI for visualization
    setup = CV(faces, eyes, video_path=video_path, gui=True)

    # Process first frame
    setup.step()

    # Main processing loop
    try:
        frame_count = 0
        while True:
            # Process current frame
            setup.step()
            frame_count += 1

            # Exit on 'q' key press
            if setup.gui:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"Error in processing: {e}")
    finally:
        # Clean up resources
        cv2.destroyAllWindows()
        setup.cap.release()
        print(f"Processed {frame_count} frames")
