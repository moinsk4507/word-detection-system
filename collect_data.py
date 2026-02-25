"""
Data Collection Script for Sign Language Alphabet Recognition
Captures hand gesture samples for training the model
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Alphabets to collect
        self.alphabets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.current_alphabet_idx = 0
        self.samples_per_alphabet = 100
        self.current_samples = 0
        
    def extract_landmarks(self, hand_landmarks):
        """Extract normalized hand landmarks as features"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def save_sample(self, landmarks, label):
        """Save a single sample to disk"""
        sample = {
            'landmarks': landmarks.tolist(),
            'label': label,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create alphabet directory
        alphabet_dir = os.path.join(self.data_dir, label)
        os.makedirs(alphabet_dir, exist_ok=True)
        
        # Save sample
        filename = os.path.join(alphabet_dir, f'sample_{self.current_samples:04d}.json')
        with open(filename, 'w') as f:
            json.dump(sample, f)
    
    def run(self):
        """Run the data collection process"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        collecting = False
        
        print("=" * 50)
        print("Sign Language Data Collection Tool")
        print("=" * 50)
        print("\nInstructions:")
        print("- Press SPACE to start/stop collecting samples")
        print("- Press 'n' to move to next alphabet")
        print("- Press 'p' to go to previous alphabet")
        print("- Press 'q' to quit")
        print("=" * 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # If collecting, save the sample
                    if collecting:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        current_alphabet = self.alphabets[self.current_alphabet_idx]
                        self.save_sample(landmarks, current_alphabet)
                        self.current_samples += 1
                        
                        # Check if we've collected enough samples
                        if self.current_samples >= self.samples_per_alphabet:
                            print(f"\n✓ Completed collecting {self.samples_per_alphabet} samples for '{current_alphabet}'")
                            collecting = False
                            self.current_samples = 0
                            
                            # Auto advance to next alphabet
                            if self.current_alphabet_idx < len(self.alphabets) - 1:
                                self.current_alphabet_idx += 1
                                print(f"Moving to next alphabet: '{self.alphabets[self.current_alphabet_idx]}'")
            
            # Display information on frame
            current_alphabet = self.alphabets[self.current_alphabet_idx]
            info_y = 30
            
            # Status
            status = "COLLECTING" if collecting else "READY"
            status_color = (0, 255, 0) if collecting else (0, 165, 255)
            cv2.putText(frame, f"Status: {status}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Current alphabet
            cv2.putText(frame, f"Alphabet: {current_alphabet}", (10, info_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            
            # Progress
            progress_text = f"Progress: {self.current_samples}/{self.samples_per_alphabet}"
            cv2.putText(frame, progress_text, (10, info_y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Overall progress
            total_alphabets = len(self.alphabets)
            overall_text = f"Alphabet: {self.current_alphabet_idx + 1}/{total_alphabets}"
            cv2.putText(frame, overall_text, (10, info_y + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            instructions = [
                "SPACE: Start/Stop",
                "N: Next alphabet",
                "P: Previous alphabet",
                "Q: Quit"
            ]
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, h - 120 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('Sign Language Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                if results.multi_hand_landmarks:
                    collecting = not collecting
                    if collecting:
                        print(f"\nStarting collection for '{current_alphabet}'...")
                    else:
                        print(f"\nPaused at {self.current_samples} samples")
                else:
                    print("\nNo hand detected! Please show your hand to the camera.")
            elif key == ord('n'):
                if self.current_alphabet_idx < len(self.alphabets) - 1:
                    self.current_alphabet_idx += 1
                    self.current_samples = 0
                    collecting = False
                    print(f"\nMoved to alphabet: '{self.alphabets[self.current_alphabet_idx]}'")
            elif key == ord('p'):
                if self.current_alphabet_idx > 0:
                    self.current_alphabet_idx -= 1
                    self.current_samples = 0
                    collecting = False
                    print(f"\nMoved back to alphabet: '{self.alphabets[self.current_alphabet_idx]}'")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\n" + "=" * 50)
        print("Data collection completed!")
        print(f"Samples saved in: {os.path.abspath(self.data_dir)}")
        print("=" * 50)


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()

