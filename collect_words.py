"""
Word/Phrase Data Collection for Sign Language
Collects complete sign language words and phrases (not letter-by-letter)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class WordDataCollector:
    def __init__(self, data_dir='data_words'):
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Some words use two hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Common sign language words/phrases
        self.words = [
            'HELLO',
            'THANK_YOU',
            'PLEASE',
            'SORRY',
            'YES',
            'NO',
            'HELP',
            'STOP',
            'MORE',
            'BATHROOM',
            'EAT',
            'DRINK',
            'WATER',
            'FOOD',
            'GOOD',
            'BAD',
            'HOW',
            'WHAT',
            'WHERE',
            'WHEN',
            'WHO',
            'WHY',
            'I_LOVE_YOU',
            'MY_NAME_IS',
            'NICE_TO_MEET_YOU',
            'GOODBYE',
            'SEE_YOU_LATER',
            'MORNING',
            'AFTERNOON',
            'NIGHT',
            'TODAY',
            'TOMORROW',
            'YESTERDAY',
            'FRIEND',
            'FAMILY',
            'MOTHER',
            'FATHER',
            'PAIN',
            'HAPPY',
            'SAD',
            'ANGRY',
            'SCARED',
            'SICK',
            'TIRED',
            'HUNGRY',
            'THIRSTY',
        ]
        
        self.current_word_idx = 0
        self.samples_per_word = 100
        self.current_samples = 0
        
    def extract_landmarks(self, hand_landmarks_list):
        """Extract normalized hand landmarks from one or two hands"""
        landmarks = []
        
        # Always store features for 2 hands (pad with zeros if only 1 hand)
        for i in range(2):
            if i < len(hand_landmarks_list):
                for landmark in hand_landmarks_list[i].landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                # No hand detected - pad with zeros
                landmarks.extend([0.0] * 63)  # 21 landmarks * 3 coords
        
        return np.array(landmarks)
    
    def save_sample(self, landmarks, label):
        """Save a single sample to disk"""
        sample = {
            'landmarks': landmarks.tolist(),
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'num_hands': 2  # Always store 2-hand data
        }
        
        # Create word directory
        word_dir = os.path.join(self.data_dir, label)
        os.makedirs(word_dir, exist_ok=True)
        
        # Save sample
        filename = os.path.join(word_dir, f'sample_{self.current_samples:04d}.json')
        with open(filename, 'w') as f:
            json.dump(sample, f)
    
    def run(self):
        """Run the data collection process"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        collecting = False
        
        print("=" * 60)
        print("Sign Language WORD/PHRASE Collection Tool")
        print("=" * 60)
        print("\nInstructions:")
        print("- These are COMPLETE words, not letter-by-letter")
        print("- Learn the signs from ASL resources before collecting")
        print("- Press SPACE to start/stop collecting samples")
        print("- Press 'n' to move to next word")
        print("- Press 'p' to go to previous word")
        print("- Press 's' to skip current word")
        print("- Press 'q' to quit")
        print("=" * 60)
        
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
            hands_detected = 0
            if results.multi_hand_landmarks:
                hands_detected = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                # If collecting, save the sample
                if collecting:
                    landmarks = self.extract_landmarks(results.multi_hand_landmarks)
                    current_word = self.words[self.current_word_idx]
                    self.save_sample(landmarks, current_word)
                    self.current_samples += 1
                    
                    # Check if we've collected enough samples
                    if self.current_samples >= self.samples_per_word:
                        print(f"\n✓ Completed collecting {self.samples_per_word} samples for '{current_word}'")
                        collecting = False
                        self.current_samples = 0
                        
                        # Auto advance to next word
                        if self.current_word_idx < len(self.words) - 1:
                            self.current_word_idx += 1
                            print(f"Moving to next word: '{self.words[self.current_word_idx]}'")
            
            # Display information on frame
            current_word = self.words[self.current_word_idx]
            info_y = 30
            
            # Status
            status = "COLLECTING" if collecting else "READY"
            status_color = (0, 255, 0) if collecting else (0, 165, 255)
            cv2.putText(frame, f"Status: {status}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Current word (large and prominent)
            word_display = current_word.replace('_', ' ')
            cv2.rectangle(frame, (10, info_y + 10), (w - 10, info_y + 90), (50, 50, 50), -1)
            cv2.putText(frame, word_display, (20, info_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            # Progress
            progress_text = f"Progress: {self.current_samples}/{self.samples_per_word}"
            cv2.putText(frame, progress_text, (10, info_y + 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Overall progress
            total_words = len(self.words)
            overall_text = f"Word: {self.current_word_idx + 1}/{total_words}"
            cv2.putText(frame, overall_text, (10, info_y + 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Hands detected
            hands_text = f"Hands: {hands_detected}"
            hands_color = (0, 255, 0) if hands_detected > 0 else (0, 0, 255)
            cv2.putText(frame, hands_text, (10, info_y + 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, hands_color, 2)
            
            # Instructions
            instructions = [
                "SPACE: Start/Stop",
                "N: Next word",
                "P: Previous word",
                "S: Skip word",
                "Q: Quit"
            ]
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, h - 150 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Reference note
            cv2.putText(frame, "Learn signs from ASL dictionaries/videos first!", 
                       (w - 550, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Sign Language Word Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                collecting = not collecting
                if collecting:
                    print(f"\nStarting collection for '{current_word}'...")
                    if hands_detected == 0:
                        print("⚠️  No hands detected! Please show your hands to the camera.")
                        collecting = False
                else:
                    print(f"\nPaused at {self.current_samples} samples")
            elif key == ord('n'):
                if self.current_word_idx < len(self.words) - 1:
                    self.current_word_idx += 1
                    self.current_samples = 0
                    collecting = False
                    print(f"\nMoved to word: '{self.words[self.current_word_idx]}'")
            elif key == ord('p'):
                if self.current_word_idx > 0:
                    self.current_word_idx -= 1
                    self.current_samples = 0
                    collecting = False
                    print(f"\nMoved back to word: '{self.words[self.current_word_idx]}'")
            elif key == ord('s'):
                # Skip current word
                if self.current_word_idx < len(self.words) - 1:
                    print(f"\nSkipped: '{self.words[self.current_word_idx]}'")
                    self.current_word_idx += 1
                    self.current_samples = 0
                    collecting = False
                    print(f"Moved to word: '{self.words[self.current_word_idx]}'")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\n" + "=" * 60)
        print("Word data collection completed!")
        print(f"Samples saved in: {os.path.abspath(self.data_dir)}")
        print("=" * 60)


if __name__ == "__main__":
    collector = WordDataCollector()
    collector.run()

