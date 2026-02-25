"""
Model Training Script for Sign Language Alphabet Recognition
Trains a neural network using collected hand landmark data
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime

class SignLanguageModelTrainer:
    def __init__(self, data_dir='data', model_dir='models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.alphabets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        self.X = []
        self.y = []
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load all collected samples from disk"""
        print("Loading data...")
        
        for alphabet in self.alphabets:
            alphabet_dir = os.path.join(self.data_dir, alphabet)
            
            if not os.path.exists(alphabet_dir):
                print(f"Warning: No data found for alphabet '{alphabet}'")
                continue
            
            sample_files = [f for f in os.listdir(alphabet_dir) if f.endswith('.json')]
            
            for sample_file in sample_files:
                filepath = os.path.join(alphabet_dir, sample_file)
                try:
                    with open(filepath, 'r') as f:
                        sample = json.load(f)
                        self.X.append(sample['landmarks'])
                        self.y.append(sample['label'])
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        print(f"Loaded {len(self.X)} samples")
        print(f"Feature shape: {self.X.shape}")
        
        # Check distribution
        unique, counts = np.unique(self.y, return_counts=True)
        print("\nData distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} samples")
        
        return len(self.X) > 0
    
    def preprocess_data(self):
        """Normalize and encode the data"""
        print("\nPreprocessing data...")
        
        # Normalize landmarks (they're already normalized by MediaPipe, but ensure consistency)
        # Calculate mean and std for each feature
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
        
        X_normalized = (self.X - self.mean) / self.std
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(self.y)
        num_classes = len(self.label_encoder.classes_)
        y_categorical = keras.utils.to_categorical(y_encoded, num_classes=num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y_categorical, 
            test_size=0.2, 
            random_state=42,
            stratify=y_encoded
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of classes: {num_classes}")
        
        return X_train, X_test, y_train, y_test, num_classes
    
    def build_model(self, input_shape, num_classes):
        """Build the neural network model"""
        print("\nBuilding model...")
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            
            # First dense block
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Second dense block
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Third dense block
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        return model
    
    def train(self, X_train, X_test, y_train, y_test, num_classes, epochs=100):
        """Train the model"""
        print("\nTraining model...")
        
        model = self.build_model(X_train.shape[1], num_classes)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the trained model"""
        print("\nEvaluating model...")
        
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for i, alphabet in enumerate(self.alphabets):
            mask = y_true_classes == i
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred_classes[mask] == y_true_classes[mask])
                print(f"  {alphabet}: {class_acc:.4f}")
        
        return accuracy
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, 'training_history.png')
        plt.savefig(plot_path)
        print(f"\nTraining history plot saved to: {plot_path}")
        plt.close()
    
    def save_model(self, model):
        """Save the trained model and preprocessing parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.model_dir, 'sign_language_model.keras')
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save preprocessing parameters
        params = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'labels': self.label_encoder.classes_.tolist()
        }
        
        params_path = os.path.join(self.model_dir, 'preprocessing_params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f)
        print(f"Preprocessing parameters saved to: {params_path}")
    
    def run(self):
        """Run the complete training pipeline"""
        print("=" * 60)
        print("Sign Language Model Training")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            print("\nError: No data found!")
            print("Please run collect_data.py first to collect training samples.")
            return
        
        # Check if we have enough data
        if len(self.X) < 100:
            print(f"\nWarning: Only {len(self.X)} samples found.")
            print("For better results, collect at least 100 samples per alphabet.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Preprocess
        X_train, X_test, y_train, y_test, num_classes = self.preprocess_data()
        
        # Train
        model, history = self.train(X_train, X_test, y_train, y_test, num_classes)
        
        # Evaluate
        accuracy = self.evaluate_model(model, X_test, y_test)
        
        # Plot
        self.plot_training_history(history)
        
        # Save
        self.save_model(model)
        
        print("\n" + "=" * 60)
        print(f"Training completed! Final accuracy: {accuracy:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    trainer = SignLanguageModelTrainer()
    trainer.run()

