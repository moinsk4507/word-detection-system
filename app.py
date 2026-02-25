"""
Streamlit Web Application for Sign Language Detection
Provides a user-friendly interface for real-time sign language recognition
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque
import time
from PIL import Image

# Page config
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #4CAF50;
        color: white;
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
    }
    .word-box {
        background-color: #f0f0f0;
        font-size: 2rem;
        padding: 1rem;
        border-radius: 5px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
        min-height: 60px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)


class SignLanguageDetectorApp:
    def __init__(self):
        """Initialize the detector"""
        self.model_path = 'models/sign_language_model.keras'
        self.params_path = 'models/preprocessing_params.json'
        
        # Initialize session state
        if 'detector_initialized' not in st.session_state:
            st.session_state.detector_initialized = False
        if 'current_word' not in st.session_state:
            st.session_state.current_word = ""
        if 'word_mode' not in st.session_state:
            st.session_state.word_mode = False
        if 'detection_active' not in st.session_state:
            st.session_state.detection_active = False
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = deque(maxlen=5)
        if 'last_letter' not in st.session_state:
            st.session_state.last_letter = None
        if 'last_letter_time' not in st.session_state:
            st.session_state.last_letter_time = time.time()
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.confidence_threshold = 0.6
        self.letter_hold_time = 1.5
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            return False
        
        try:
            st.session_state.model = tf.keras.models.load_model(self.model_path)
            
            with open(self.params_path, 'r') as f:
                params = json.load(f)
                st.session_state.mean = np.array(params['mean'])
                st.session_state.std = np.array(params['std'])
                st.session_state.labels = params['labels']
            
            st.session_state.detector_initialized = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def preprocess_landmarks(self, landmarks):
        """Normalize landmarks"""
        normalized = (landmarks - st.session_state.mean) / st.session_state.std
        return normalized.reshape(1, -1)
    
    def predict(self, landmarks):
        """Predict alphabet"""
        X = self.preprocess_landmarks(landmarks)
        predictions = st.session_state.model.predict(X, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_label = st.session_state.labels[predicted_idx]
        
        return predicted_label, confidence, predictions
    
    def smooth_prediction(self, predicted_label, confidence):
        """Smooth predictions"""
        if confidence < self.confidence_threshold:
            return None, 0.0
        
        st.session_state.prediction_history.append(predicted_label)
        
        if len(st.session_state.prediction_history) >= 3:
            from collections import Counter
            most_common = Counter(st.session_state.prediction_history).most_common(1)[0]
            if most_common[1] >= 3:
                return most_common[0], confidence
        
        return predicted_label, confidence
    
    def update_word(self, letter):
        """Update word in spelling mode"""
        current_time = time.time()
        
        if letter == st.session_state.last_letter:
            if current_time - st.session_state.last_letter_time >= self.letter_hold_time:
                if len(st.session_state.current_word) == 0 or st.session_state.current_word[-1] != letter:
                    st.session_state.current_word += letter
                    st.session_state.last_letter_time = current_time
        else:
            st.session_state.last_letter = letter
            st.session_state.last_letter_time = current_time
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hands:
            results = hands.process(rgb_frame)
            
            predicted_label = None
            confidence = 0.0
            top_predictions = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Draw bounding box
                    h, w, c = frame.shape
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords) * w) - 20, int(max(x_coords) * w) + 20
                    y_min, y_max = int(min(y_coords) * h) - 20, int(max(y_coords) * h) + 20
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Predict
                    landmarks = self.extract_landmarks(hand_landmarks)
                    pred_label, conf, top_preds = self.predict(landmarks)
                    predicted_label, confidence = self.smooth_prediction(pred_label, conf)
                    top_predictions = top_preds
                    
                    # Update word mode
                    if st.session_state.word_mode and predicted_label:
                        self.update_word(predicted_label)
            
            return frame, predicted_label, confidence, top_predictions
    
    def render_ui(self):
        """Render the Streamlit UI"""
        # Header
        st.markdown('<h1 class="main-header">🤟 Sign Language Alphabet Detector</h1>', 
                   unsafe_allow_html=True)
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            st.error("⚠️ Model not found!")
            st.info("""
                Please follow these steps to set up the detector:
                1. Run `python collect_data.py` to collect training data
                2. Run `python train_model.py` to train the model
                3. Refresh this page
            """)
            return
        
        # Load model
        if not st.session_state.detector_initialized:
            with st.spinner("Loading model..."):
                if self.load_model():
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("Failed to load model")
                    return
        
        # Sidebar controls
        with st.sidebar:
            st.header("⚙️ Controls")
            
            # Detection toggle
            if st.button("🎥 Start/Stop Detection"):
                st.session_state.detection_active = not st.session_state.detection_active
            
            st.write(f"**Status:** {'🟢 Active' if st.session_state.detection_active else '🔴 Inactive'}")
            
            st.divider()
            
            # Word mode
            st.header("📝 Word Spelling Mode")
            st.session_state.word_mode = st.checkbox("Enable Word Mode", 
                                                      value=st.session_state.word_mode)
            
            if st.session_state.word_mode:
                st.info(f"Hold a sign for {self.letter_hold_time}s to add it to the word")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➕ Add Space"):
                        st.session_state.current_word += " "
                with col2:
                    if st.button("⌫ Backspace"):
                        st.session_state.current_word = st.session_state.current_word[:-1]
                
                if st.button("🗑️ Clear Word"):
                    st.session_state.current_word = ""
                
                if st.button("💾 Save Word"):
                    if st.session_state.current_word:
                        with open('detected_words.txt', 'a') as f:
                            f.write(f"{st.session_state.current_word}\n")
                        st.success(f"Saved: {st.session_state.current_word}")
                        st.session_state.current_word = ""
            
            st.divider()
            
            # Settings
            st.header("🔧 Settings")
            self.confidence_threshold = st.slider(
                "Confidence Threshold", 
                0.0, 1.0, 0.6, 0.05
            )
            
            self.letter_hold_time = st.slider(
                "Letter Hold Time (s)", 
                0.5, 3.0, 1.5, 0.1
            )
            
            st.divider()
            
            # Info
            st.header("ℹ️ About")
            st.write("""
                This app detects American Sign Language (ASL) 
                alphabets A-Z in real-time using:
                - **OpenCV** for video processing
                - **MediaPipe** for hand tracking
                - **TensorFlow** for prediction
            """)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Live Feed")
            frame_placeholder = st.empty()
        
        with col2:
            st.subheader("🎯 Prediction")
            prediction_placeholder = st.empty()
            confidence_placeholder = st.empty()
            
            st.subheader("📊 Top 3 Predictions")
            top_predictions_placeholder = st.empty()
            
            if st.session_state.word_mode:
                st.subheader("📝 Current Word")
                word_placeholder = st.empty()
        
        # Camera feed
        if st.session_state.detection_active:
            cap = cv2.VideoCapture(0)
            
            try:
                while st.session_state.detection_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access camera")
                        break
                    
                    # Flip frame
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    processed_frame, predicted_label, confidence, top_preds = self.process_frame(frame)
                    
                    # Display frame
                    frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                    
                    # Display prediction
                    if predicted_label and confidence > self.confidence_threshold:
                        prediction_placeholder.markdown(
                            f'<div class="prediction-box">{predicted_label}</div>', 
                            unsafe_allow_html=True
                        )
                        confidence_placeholder.markdown(
                            f'<p class="confidence-text">Confidence: {confidence*100:.1f}%</p>', 
                            unsafe_allow_html=True
                        )
                    else:
                        prediction_placeholder.markdown(
                            '<div class="prediction-box">-</div>', 
                            unsafe_allow_html=True
                        )
                        confidence_placeholder.markdown(
                            '<p class="confidence-text">No detection</p>', 
                            unsafe_allow_html=True
                        )
                    
                    # Display top predictions
                    if top_preds is not None:
                        top_indices = np.argsort(top_preds)[-3:][::-1]
                        top_text = ""
                        for i, idx in enumerate(top_indices):
                            label = st.session_state.labels[idx]
                            conf = top_preds[idx]
                            top_text += f"{i+1}. **{label}**: {conf*100:.1f}%\n\n"
                        top_predictions_placeholder.markdown(top_text)
                    
                    # Display current word
                    if st.session_state.word_mode:
                        word_text = st.session_state.current_word if st.session_state.current_word else "(empty)"
                        word_placeholder.markdown(
                            f'<div class="word-box">{word_text}</div>', 
                            unsafe_allow_html=True
                        )
                    
                    time.sleep(0.03)  # ~30 FPS
                    
            finally:
                cap.release()
        else:
            st.info("👆 Click 'Start/Stop Detection' in the sidebar to begin")


def main():
    app = SignLanguageDetectorApp()
    app.render_ui()


if __name__ == "__main__":
    main()

