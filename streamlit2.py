import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to compute angle
def compute_angle(start, middle, end):
    try:
        vector1 = middle - start
        vector2 = end - middle
        dot_product = np.dot(vector1, vector2)
        magnitude_v1 = np.linalg.norm(vector1)
        magnitude_v2 = np.linalg.norm(vector2)
        
        if magnitude_v1 * magnitude_v2 == 0:
            return None
        
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    except:
        return None

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame with MediaPipe Pose
        results = self.pose.process(img)
        
        if results.pose_landmarks:
            # Draw pose landmarks and connections
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Calculate knee angle
            left_knee = results.pose_landmarks.landmark[23]
            left_hip = results.pose_landmarks.landmark[23 - 6]  # Adjusted index for hip
            left_ankle = results.pose_landmarks.landmark[27]
            knee_angle = compute_angle(np.array([left_hip.x, left_hip.y]), np.array([left_knee.x, left_knee.y]), np.array([left_ankle.x, left_ankle.y]))
            
            if knee_angle is not None:
                cv2.putText(img, f"Knee Angle: {round(knee_angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit App
st.title("Real-Time Pose Estimation with MediaPipe")

webrtc_ctx = webrtc_streamer(
    key="pose-estimation",
    video_processor_factory=VideoProcessor,
    async_processing=True
)
