"""
Real-time Sign Language Alphabet Detection
Detects and displays sign language alphabets using webcam
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque
import time

class SignLanguageDetector:
    def __init__(self, model_path='models/sign_language_model.keras', 
                 params_path='models/preprocessing_params.json'):
        """Initialize the sign language detector"""
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load preprocessing parameters
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Parameters not found at {params_path}. Please train the model first.")
        
        with open(params_path, 'r') as f:
            params = json.load(f)
            self.mean = np.array(params['mean'])
            self.std = np.array(params['std'])
            self.labels = params['labels']
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.6
        
        # Word spelling mode
        self.word_mode = False
        self.current_word = ""
        self.last_letter = None
        self.last_letter_time = time.time()
        self.letter_hold_time = 1.5  # seconds to hold before adding letter
        
        print("Detector initialized successfully!")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract normalized hand landmarks as features"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def preprocess_landmarks(self, landmarks):
        """Normalize landmarks using training parameters"""
        normalized = (landmarks - self.mean) / self.std
        return normalized.reshape(1, -1)
    
    def predict(self, landmarks):
        """Predict the alphabet from landmarks"""
        # Preprocess
        X = self.preprocess_landmarks(landmarks)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_label = self.labels[predicted_idx]
        
        return predicted_label, confidence, predictions
    
    def smooth_prediction(self, predicted_label, confidence):
        """Smooth predictions over time"""
        if confidence < self.confidence_threshold:
            return None, 0.0
        
        self.prediction_history.append(predicted_label)
        
        # Get most common prediction in history
        if len(self.prediction_history) >= 3:
            from collections import Counter
            most_common = Counter(self.prediction_history).most_common(1)[0]
            if most_common[1] >= 3:  # At least 3 occurrences
                return most_common[0], confidence
        
        return predicted_label, confidence
    
    def update_word(self, letter):
        """Update word in word spelling mode"""
        current_time = time.time()
        
        if letter == self.last_letter:
            # Same letter - check if held long enough
            if current_time - self.last_letter_time >= self.letter_hold_time:
                if len(self.current_word) == 0 or self.current_word[-1] != letter:
                    self.current_word += letter
                    self.last_letter_time = current_time
        else:
            # New letter
            self.last_letter = letter
            self.last_letter_time = current_time
    
    def draw_hand_info(self, frame, hand_landmarks):
        """Draw hand landmarks and bounding box"""
        h, w, c = frame.shape
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        
        # Calculate bounding box
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return (x_min, y_min, x_max, y_max)
    
    def draw_prediction(self, frame, predicted_label, confidence, bbox, top_predictions):
        """Draw prediction on frame"""
        x_min, y_min, x_max, y_max = bbox
        
        # Draw main prediction
        if predicted_label and confidence > self.confidence_threshold:
            # Large letter display
            text = predicted_label
            font_scale = 3
            thickness = 5
            
            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw background rectangle
            text_x = x_min
            text_y = y_min - 20
            cv2.rectangle(frame, 
                         (text_x, text_y - text_h - 10), 
                         (text_x + text_w + 10, text_y + 5),
                         (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (text_x + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # Draw confidence
            conf_text = f"{confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (x_min, y_max + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw top 3 predictions sidebar
        sidebar_x = 10
        sidebar_y = 100
        
        cv2.putText(frame, "Top Predictions:", (sidebar_x, sidebar_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Get top 3 (only from valid label indices)
        num_classes = len(self.labels)
        if len(top_predictions) > num_classes:
            # Model has more outputs than labels - only consider valid indices
            valid_predictions = top_predictions[:num_classes]
        else:
            valid_predictions = top_predictions
        
        top_indices = np.argsort(valid_predictions)[-min(3, num_classes):][::-1]
        for i, idx in enumerate(top_indices):
            if idx < len(self.labels):  # Safety check
                label = self.labels[idx]
                conf = valid_predictions[idx]
                text = f"{i+1}. {label}: {conf*100:.1f}%"
                y = sidebar_y + 30 + i * 25
                cv2.putText(frame, text, (sidebar_x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_ui(self, frame):
        """Draw UI elements"""
        h, w, c = frame.shape
        
        # Draw title
        cv2.putText(frame, "Sign Language Detector", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw word mode indicator
        if self.word_mode:
            mode_text = "Mode: WORD SPELLING"
            cv2.putText(frame, mode_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw current word
            word_bg_height = 80
            cv2.rectangle(frame, (0, h - word_bg_height), (w, h), (50, 50, 50), -1)
            
            cv2.putText(frame, "Current Word:", (10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            word_text = self.current_word if self.current_word else "(empty)"
            cv2.putText(frame, word_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            
            # Progress bar for letter hold
            if self.last_letter:
                progress = min(1.0, (time.time() - self.last_letter_time) / self.letter_hold_time)
                bar_width = 200
                bar_height = 10
                bar_x = w - bar_width - 10
                bar_y = h - 65
                
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height), 
                            (100, 100, 100), -1)
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + int(bar_width * progress), bar_y + bar_height), 
                            (0, 255, 255), -1)
        
        # Draw controls
        controls = [
            "Controls:",
            "W: Toggle Word Mode",
            "SPACE: Add Space (Word Mode)",
            "BACKSPACE: Delete Letter",
            "C: Clear Word",
            "S: Save Word to file",
            "Q: Quit"
        ]
        
        control_x = w - 300
        control_y = 100
        for i, control in enumerate(controls):
            color = (200, 200, 200) if i > 0 else (255, 255, 255)
            cv2.putText(frame, control, (control_x, control_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run(self):
        """Run the detector"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 60)
        print("Sign Language Detector Started!")
        print("=" * 60)
        print("\nControls:")
        print("  W - Toggle Word Spelling Mode")
        print("  SPACE - Add space to word")
        print("  BACKSPACE - Delete last letter")
        print("  C - Clear current word")
        print("  S - Save word to file")
        print("  Q - Quit")
        print("=" * 60 + "\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            predicted_label = None
            confidence = 0.0
            top_predictions = None
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand info
                    bbox = self.draw_hand_info(frame, hand_landmarks)
                    
                    # Extract and predict
                    landmarks = self.extract_landmarks(hand_landmarks)
                    pred_label, conf, top_preds = self.predict(landmarks)
                    
                    # Smooth prediction
                    predicted_label, confidence = self.smooth_prediction(pred_label, conf)
                    top_predictions = top_preds
                    
                    # Update word if in word mode
                    if self.word_mode and predicted_label:
                        self.update_word(predicted_label)
                    
                    # Draw prediction
                    if top_predictions is not None:
                        self.draw_prediction(frame, predicted_label, confidence, bbox, top_predictions)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow('Sign Language Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('w'):
                self.word_mode = not self.word_mode
                print(f"Word mode: {'ON' if self.word_mode else 'OFF'}")
            elif key == ord(' ') and self.word_mode:
                self.current_word += ' '
            elif key == 8 and self.word_mode:  # Backspace
                self.current_word = self.current_word[:-1]
            elif key == ord('c') and self.word_mode:
                self.current_word = ""
                print("Word cleared")
            elif key == ord('s') and self.word_mode:
                if self.current_word:
                    with open('detected_words.txt', 'a') as f:
                        f.write(f"{self.current_word}\n")
                    print(f"Saved: {self.current_word}")
                    self.current_word = ""
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\nDetector closed.")


if __name__ == "__main__":
    try:
        detector = SignLanguageDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease follow these steps:")
        print("1. Run 'python collect_data.py' to collect training data")
        print("2. Run 'python train_model.py' to train the model")
        print("3. Run 'python detect_sign_language.py' to use the detector")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

