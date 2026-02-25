"""
Real-time Sign Language Word/Phrase Detection
Detects complete sign language words using webcam
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque
import time

class SignLanguageWordDetector:
    def __init__(self, model_path='models/sign_language_words_model.keras', 
                 params_path='models/preprocessing_params_words.json'):
        """Initialize the word detector"""
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first using train_words.py")
        
        print("Loading word recognition model...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load preprocessing parameters
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Parameters not found at {params_path}. Please train the model first.")
        
        with open(params_path, 'r') as f:
            params = json.load(f)
            self.mean = np.array(params['mean'])
            self.std = np.array(params['std'])
            self.labels = params['labels']
            self.num_hands = params.get('num_hands', 2)
        
        # Initialize MediaPipe with 2 hands support
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support two-handed signs
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=7)
        self.confidence_threshold = 0.65
        
        # Sentence building mode
        self.sentence_mode = False
        self.current_sentence = []
        self.last_word = None
        self.last_word_time = time.time()
        self.word_hold_time = 2.0  # Hold for 2 seconds to add word
        
        print("Word detector initialized successfully!")
        print(f"Loaded {len(self.labels)} word categories")
    
    def extract_landmarks(self, hand_landmarks_list):
        """Extract normalized hand landmarks from one or two hands"""
        landmarks = []
        
        # Always store features for 2 hands (pad with zeros if only 1 hand)
        for i in range(2):
            if hand_landmarks_list and i < len(hand_landmarks_list):
                for landmark in hand_landmarks_list[i].landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                # No hand detected - pad with zeros
                landmarks.extend([0.0] * 63)
        
        return np.array(landmarks)
    
    def preprocess_landmarks(self, landmarks):
        """Normalize landmarks using training parameters"""
        normalized = (landmarks - self.mean) / self.std
        return normalized.reshape(1, -1)
    
    def predict(self, landmarks):
        """Predict the word from landmarks"""
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
        if len(self.prediction_history) >= 4:
            from collections import Counter
            most_common = Counter(self.prediction_history).most_common(1)[0]
            if most_common[1] >= 4:  # At least 4 occurrences
                return most_common[0], confidence
        
        return predicted_label, confidence
    
    def update_sentence(self, word):
        """Update sentence in sentence building mode"""
        current_time = time.time()
        
        if word == self.last_word:
            # Same word - check if held long enough
            if current_time - self.last_word_time >= self.word_hold_time:
                if len(self.current_sentence) == 0 or self.current_sentence[-1] != word:
                    self.current_sentence.append(word)
                    self.last_word_time = current_time
                    print(f"Added to sentence: {word}")
        else:
            # New word
            self.last_word = word
            self.last_word_time = current_time
    
    def draw_hand_info(self, frame, hand_landmarks_list):
        """Draw hand landmarks and bounding boxes for all hands"""
        h, w, c = frame.shape
        bboxes = []
        
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
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
                bboxes.append((x_min, y_min, x_max, y_max))
        
        return bboxes
    
    def draw_prediction(self, frame, predicted_word, confidence):
        """Draw prediction on frame"""
        h, w, c = frame.shape
        
        if predicted_word and confidence > self.confidence_threshold:
            # Format word for display
            display_word = predicted_word.replace('_', ' ')
            
            # Draw background box
            box_height = 100
            cv2.rectangle(frame, (0, 0), (w, box_height), (50, 50, 50), -1)
            
            # Draw word
            font_scale = 2.0
            thickness = 4
            (text_w, text_h), _ = cv2.getTextSize(display_word, cv2.FONT_HERSHEY_SIMPLEX, 
                                                   font_scale, thickness)
            
            text_x = (w - text_w) // 2
            text_y = (box_height + text_h) // 2
            
            cv2.putText(frame, display_word, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            
            # Draw confidence
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (10, box_height + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def draw_ui(self, frame):
        """Draw UI elements"""
        h, w, c = frame.shape
        
        # Draw sentence mode indicator
        if self.sentence_mode:
            mode_text = "Mode: SENTENCE BUILDING"
            cv2.putText(frame, mode_text, (10, h - 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw current sentence box
            sentence_bg_height = 120
            cv2.rectangle(frame, (0, h - sentence_bg_height), (w, h), (40, 40, 40), -1)
            
            cv2.putText(frame, "Current Sentence:", (10, h - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Format sentence
            sentence_text = ' '.join([word.replace('_', ' ') for word in self.current_sentence])
            if not sentence_text:
                sentence_text = "(empty)"
            
            # Word wrap for long sentences
            max_chars = 60
            if len(sentence_text) > max_chars:
                sentence_text = sentence_text[:max_chars] + "..."
            
            cv2.putText(frame, sentence_text, (10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Progress bar for word hold
            if self.last_word:
                progress = min(1.0, (time.time() - self.last_word_time) / self.word_hold_time)
                bar_width = 300
                bar_height = 15
                bar_x = w - bar_width - 10
                bar_y = h - 25
                
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height), 
                            (100, 100, 100), -1)
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + int(bar_width * progress), bar_y + bar_height), 
                            (0, 255, 255), -1)
                cv2.putText(frame, "Hold to add", (bar_x, bar_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw controls
        controls = [
            "Controls:",
            "S: Toggle Sentence Mode",
            "SPACE: Clear Sentence",
            "BACKSPACE: Remove Last Word",
            "V: Save Sentence",
            "Q: Quit"
        ]
        
        control_x = w - 320
        control_y = 150
        for i, control in enumerate(controls):
            color = (255, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(frame, control, (control_x, control_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run(self):
        """Run the word detector"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 70)
        print("Sign Language WORD Detector Started!")
        print("=" * 70)
        print("\nControls:")
        print("  S - Toggle Sentence Building Mode")
        print("  SPACE - Clear sentence")
        print("  BACKSPACE - Remove last word")
        print("  V - Save sentence to file")
        print("  Q - Quit")
        print("=" * 70 + "\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            predicted_word = None
            confidence = 0.0
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                # Draw hand info
                bboxes = self.draw_hand_info(frame, results.multi_hand_landmarks)
                
                # Extract and predict
                landmarks = self.extract_landmarks(results.multi_hand_landmarks)
                pred_word, conf, _ = self.predict(landmarks)
                
                # Smooth prediction
                predicted_word, confidence = self.smooth_prediction(pred_word, conf)
                
                # Update sentence if in sentence mode
                if self.sentence_mode and predicted_word:
                    self.update_sentence(predicted_word)
            
            # Draw prediction
            self.draw_prediction(frame, predicted_word, confidence)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow('Sign Language Word Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.sentence_mode = not self.sentence_mode
                print(f"Sentence mode: {'ON' if self.sentence_mode else 'OFF'}")
            elif key == ord(' ') and self.sentence_mode:
                self.current_sentence = []
                print("Sentence cleared")
            elif key == 8 and self.sentence_mode:  # Backspace
                if self.current_sentence:
                    removed = self.current_sentence.pop()
                    print(f"Removed: {removed}")
            elif key == ord('v') and self.sentence_mode:
                if self.current_sentence:
                    sentence_text = ' '.join([word.replace('_', ' ') for word in self.current_sentence])
                    with open('detected_sentences.txt', 'a') as f:
                        f.write(f"{sentence_text}\n")
                    print(f"Saved: {sentence_text}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\nWord detector closed.")


if __name__ == "__main__":
    try:
        detector = SignLanguageWordDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease follow these steps:")
        print("1. Run 'python collect_words.py' to collect word gesture data")
        print("2. Run 'python train_words.py' to train the word recognition model")
        print("3. Run 'python detect_words.py' to use the detector")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

