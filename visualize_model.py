"""
Model Visualization and Analysis Script
Provides insights into model performance and predictions
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


class ModelVisualizer:
    def __init__(self, model_path='models/sign_language_model.keras',
                 params_path='models/preprocessing_params.json',
                 data_dir='data'):
        """Initialize the visualizer"""
        self.model_path = model_path
        self.params_path = params_path
        self.data_dir = data_dir
        
        # Load model and parameters
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                self.mean = np.array(params['mean'])
                self.std = np.array(params['std'])
                self.labels = params['labels']
        else:
            raise FileNotFoundError(f"Parameters not found at: {params_path}")
    
    def load_test_data(self):
        """Load data for evaluation"""
        X = []
        y = []
        
        for alphabet in self.labels:
            alphabet_dir = os.path.join(self.data_dir, alphabet)
            
            if not os.path.exists(alphabet_dir):
                continue
            
            sample_files = [f for f in os.listdir(alphabet_dir) if f.endswith('.json')]
            
            for sample_file in sample_files:
                filepath = os.path.join(alphabet_dir, sample_file)
                try:
                    with open(filepath, 'r') as f:
                        sample = json.load(f)
                        X.append(sample['landmarks'])
                        y.append(sample['label'])
                except Exception as e:
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize
        X_normalized = (X - self.mean) / self.std
        
        return X_normalized, y
    
    def plot_model_architecture(self):
        """Visualize model architecture"""
        print("\n" + "=" * 60)
        print("MODEL ARCHITECTURE")
        print("=" * 60)
        
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        
        # Create architecture diagram
        fig, ax = plt.subplots(figsize=(10, 8))
        
        layer_info = []
        for layer in self.model.layers:
            layer_type = layer.__class__.__name__
            output_shape = layer.output_shape
            params = layer.count_params()
            layer_info.append(f"{layer_type}\n{output_shape}\n{params:,} params")
        
        y_positions = np.linspace(0.9, 0.1, len(layer_info))
        
        for i, (info, y_pos) in enumerate(zip(layer_info, y_positions)):
            ax.text(0.5, y_pos, info, 
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                   fontsize=10)
            
            if i < len(layer_info) - 1:
                ax.arrow(0.5, y_pos - 0.03, 0, -0.03,
                        head_width=0.05, head_length=0.01,
                        fc='gray', ec='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Model Architecture', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/model_architecture.png', dpi=150, bbox_inches='tight')
        print("\n✓ Architecture diagram saved: models/model_architecture.png")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Generate confusion matrix"""
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)
        
        X, y = self.load_test_data()
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        y_pred = np.array([self.labels[i] for i in np.argmax(predictions, axis=1)])
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred, labels=self.labels)
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 14))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.labels,
                   yticklabels=self.labels,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix - Sign Language Recognition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("✓ Confusion matrix saved: models/confusion_matrix.png")
        plt.close()
        
        return cm
    
    def plot_per_class_accuracy(self):
        """Plot accuracy for each class"""
        print("\n" + "=" * 60)
        print("PER-CLASS ACCURACY")
        print("=" * 60)
        
        X, y = self.load_test_data()
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        y_pred_idx = np.argmax(predictions, axis=1)
        y_pred = np.array([self.labels[i] for i in y_pred_idx])
        
        # Calculate per-class accuracy
        accuracies = {}
        for label in self.labels:
            mask = y == label
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == y[mask])
                accuracies[label] = acc
            else:
                accuracies[label] = 0
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        labels = list(accuracies.keys())
        values = list(accuracies.values())
        colors = ['green' if v >= 0.9 else 'orange' if v >= 0.7 else 'red' for v in values]
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2%}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Alphabet', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.3, label='90% threshold')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, label='70% threshold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('models/per_class_accuracy.png', dpi=150, bbox_inches='tight')
        print("✓ Per-class accuracy plot saved: models/per_class_accuracy.png")
        plt.close()
        
        # Print statistics
        avg_accuracy = np.mean(list(accuracies.values()))
        print(f"\nAverage Accuracy: {avg_accuracy:.2%}")
        
        print("\nTop 5 Best Performing:")
        sorted_acc = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
        for label, acc in sorted_acc[:5]:
            print(f"  {label}: {acc:.2%}")
        
        print("\nTop 5 Worst Performing:")
        for label, acc in sorted_acc[-5:]:
            print(f"  {label}: {acc:.2%}")
    
    def plot_confidence_distribution(self):
        """Plot distribution of prediction confidences"""
        print("\n" + "=" * 60)
        print("CONFIDENCE DISTRIBUTION")
        print("=" * 60)
        
        X, y = self.load_test_data()
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        confidences = np.max(predictions, axis=1)
        correct = np.array([self.labels[i] for i in np.argmax(predictions, axis=1)]) == y
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall distribution
        ax1.hist(confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.axvline(np.median(confidences), color='green', linestyle='--', 
                   label=f'Median: {np.median(confidences):.3f}')
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Overall Confidence Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Correct vs Incorrect
        ax2.hist(confidences[correct], bins=30, alpha=0.7, color='green', 
                label='Correct', edgecolor='black')
        ax2.hist(confidences[~correct], bins=30, alpha=0.7, color='red', 
                label='Incorrect', edgecolor='black')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Confidence: Correct vs Incorrect', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/confidence_distribution.png', dpi=150, bbox_inches='tight')
        print("✓ Confidence distribution plot saved: models/confidence_distribution.png")
        plt.close()
        
        print(f"\nMean Confidence: {np.mean(confidences):.3f}")
        print(f"Median Confidence: {np.median(confidences):.3f}")
        print(f"Min Confidence: {np.min(confidences):.3f}")
        print(f"Max Confidence: {np.max(confidences):.3f}")
    
    def generate_report(self):
        """Generate comprehensive classification report"""
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        
        X, y = self.load_test_data()
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        y_pred = np.array([self.labels[i] for i in np.argmax(predictions, axis=1)])
        
        # Generate report
        report = classification_report(y, y_pred, labels=self.labels, zero_division=0)
        print("\n" + report)
        
        # Save report
        with open('models/classification_report.txt', 'w') as f:
            f.write("SIGN LANGUAGE RECOGNITION - CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        
        print("\n✓ Report saved: models/classification_report.txt")
    
    def run_all_visualizations(self):
        """Run all visualization functions"""
        print("\n" + "=" * 60)
        print("MODEL VISUALIZATION AND ANALYSIS")
        print("=" * 60)
        
        self.plot_model_architecture()
        self.plot_confusion_matrix()
        self.plot_per_class_accuracy()
        self.plot_confidence_distribution()
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETE!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - models/model_architecture.png")
        print("  - models/confusion_matrix.png")
        print("  - models/per_class_accuracy.png")
        print("  - models/confidence_distribution.png")
        print("  - models/classification_report.txt")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        visualizer = ModelVisualizer()
        visualizer.run_all_visualizations()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease make sure you have:")
        print("  1. Collected training data (python collect_data.py)")
        print("  2. Trained the model (python train_model.py)")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

