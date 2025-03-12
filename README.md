# Classroom Attention Tracker

A computer vision system for tracking and analyzing student attention levels in classroom environments using face and eye detection.

## Project Overview

This project uses advanced computer vision techniques to analyze attention levels of individuals based on facial and eye detection. The system processes video input (either from a webcam or a video file), detects faces, tracks eye movements, and calculates an attention score based on gaze direction. The results are processed to identify unique individuals and analyze their attention patterns over time.

## Components

The project consists of three main components:

### 1. Classroom.py - Data Collection

`classroom.py` handles video capture, face detection, eye tracking, and attention scoring.

**Key Features:**
- Multiple face detection methods (OpenCV, RetinaFace, MediaPipe, etc.)
- Advanced eye detection with multiple fallback mechanisms
- Real-time attention scoring based on eye position and gaze direction
- Observation storage for later analysis

**Usage:**
```bash
python classroom.py [OPTIONS]
```

**Options:**
- `--faces`: Face detection method (default: 'opencv')
  - Choices: `opencv`, `ssd`, `dlib`, `mtcnn`, `fastmtcnn`, `retinaface`, `mediapipe`, `yolov8`, `yolov11s`, `yolov11n`, `yolov11m`, `yunet`, `centerface`
- `--eyes`: Eye detection method (default: 'hough')
  - Choices: `haar`, `mediapipe`, `hough`
- `--video`: Path to video file (default: None, uses webcam)
- `--gui`: Enable GUI visualization (default: False)
- `--output`: File to save observations (default: "observations.pkl")

### 2. Analysis.py - Data Processing

`analysis.py` processes the saved observations to identify unique individuals and analyze attention patterns.

**Key Features:**
- Face embedding extraction using DeepFace
- Identity clustering using DBSCAN
- Calculation of per-person attention metrics
- Time-series analysis of attention patterns
- Visualization generation

**Usage:**
```bash
python analysis.py
```

### 3. Dashboard.py - Visualization Interface

`dashboard.py` provides an interactive web dashboard to explore the analysis results.

**Key Features:**
- Interactive Streamlit-based UI
- Gallery view of detected individuals
- Attention time-series visualization
- Statistical analysis of attention patterns
- Individual person detailed analysis

**Usage:**
```bash
streamlit run dashboard.py
```

## How It Works

1. **Data Collection**: The system captures video frames and processes them to detect faces and eyes. For each detected face, it calculates an attention score based on eye position and gaze direction.

2. **Identity Recognition**: Using facial embeddings, the system identifies unique individuals across multiple frames, allowing it to track attention patterns for each person over time.

3. **Attention Scoring**: The attention score (0-1) is calculated based on:
   - Pupil/iris position within the eye
   - Eye symmetry (whether both eyes are looking in the same direction)
   - Head orientation

4. **Analysis**: The system clusters observations by identity, calculates average attention scores and other metrics, and generates visualizations showing attention patterns over time.

5. **Visualization**: The dashboard provides an intuitive interface to explore the results, view attention trends, and analyze individual attention patterns.

## Technical Details

- **Face Detection**: Multiple methods supported including OpenCV Haar Cascades, RetinaFace, MediaPipe, and others.
- **Eye Detection**: Uses a combination of MediaPipe Face Mesh, dlib facial landmarks, and enhanced Haar cascades.
- **Identity Recognition**: Uses DeepFace for face embeddings and DBSCAN for clustering similar faces.
- **Attention Calculation**: Based on pupil position relative to eye center, eye symmetry, and head angle.
- **Visualization**: Matplotlib for static visualizations and Streamlit for interactive dashboard.

## Dependencies

The project requires several Python libraries including:
- OpenCV
- MediaPipe
- DeepFace
- RetinaFace
- Dlib
- NumPy
- Matplotlib
- Streamlit
- Scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Applications

This tool can be used for:
- Educational research on student engagement
- Classroom monitoring and feedback for teachers
- Identifying disengaged students who may need additional support
- Analyzing the effectiveness of teaching methods based on student attention