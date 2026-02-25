"""
Utility functions for Sign Language Detection project
"""

import os
import json
import numpy as np
import cv2
from datetime import datetime


def count_samples(data_dir='data'):
    """Count number of samples for each alphabet"""
    alphabets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    counts = {}
    
    for alphabet in alphabets:
        alphabet_dir = os.path.join(data_dir, alphabet)
        if os.path.exists(alphabet_dir):
            sample_files = [f for f in os.listdir(alphabet_dir) if f.endswith('.json')]
            counts[alphabet] = len(sample_files)
        else:
            counts[alphabet] = 0
    
    return counts


def print_data_statistics(data_dir='data'):
    """Print statistics about collected data"""
    counts = count_samples(data_dir)
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION STATISTICS")
    print("=" * 60)
    
    total_samples = sum(counts.values())
    collected_letters = sum(1 for c in counts.values() if c > 0)
    
    print(f"\nTotal Samples: {total_samples}")
    print(f"Letters Collected: {collected_letters}/26")
    print(f"Average per Letter: {total_samples/26:.1f}")
    
    print("\nPer-Letter Breakdown:")
    print("-" * 60)
    
    for alphabet, count in counts.items():
        bar_length = min(50, count // 2)  # Scale for display
        bar = "█" * bar_length
        status = "✓" if count >= 100 else "⚠" if count > 0 else "✗"
        print(f"{status} {alphabet}: {count:3d} samples {bar}")
    
    print("=" * 60)
    
    # Check for imbalanced data
    if total_samples > 0:
        max_count = max(counts.values())
        min_count = min(c for c in counts.values() if c > 0) if collected_letters > 0 else 0
        
        if max_count > 0 and min_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 2:
                print(f"\n⚠️  Warning: Data is imbalanced (ratio: {imbalance_ratio:.2f})")
                print("   Consider collecting more samples for underrepresented letters.")
    
    return counts


def check_model_exists(model_dir='models'):
    """Check if trained model exists"""
    model_path = os.path.join(model_dir, 'sign_language_model.keras')
    params_path = os.path.join(model_dir, 'preprocessing_params.json')
    
    model_exists = os.path.exists(model_path)
    params_exists = os.path.exists(params_path)
    
    if model_exists and params_exists:
        # Get model info
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        
        print("\n" + "=" * 60)
        print("MODEL STATUS")
        print("=" * 60)
        print(f"✓ Model found: {model_path}")
        print(f"  Size: {model_size:.2f} MB")
        print(f"  Last modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return True
    else:
        print("\n" + "=" * 60)
        print("MODEL STATUS")
        print("=" * 60)
        print("✗ Model not found")
        print("  Run 'python train_model.py' to train a model")
        print("=" * 60)
        
        return False


def validate_environment():
    """Validate that all dependencies are installed"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT VALIDATION")
    print("=" * 60)
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('tensorflow', 'tensorflow'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('streamlit', 'streamlit'),
    ]
    
    all_installed = True
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - NOT INSTALLED")
            all_installed = False
    
    print("=" * 60)
    
    if all_installed:
        print("✓ All dependencies installed!")
    else:
        print("✗ Some dependencies missing. Run: pip install -r requirements.txt")
    
    print("=" * 60)
    
    return all_installed


def test_camera(camera_index=0):
    """Test if camera is accessible"""
    print("\n" + "=" * 60)
    print("CAMERA TEST")
    print("=" * 60)
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"✗ Camera {camera_index} not accessible")
        print("  Possible issues:")
        print("  - Camera is being used by another application")
        print("  - No camera available")
        print("  - Permission denied")
        print("=" * 60)
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        h, w, c = frame.shape
        print(f"✓ Camera {camera_index} is working!")
        print(f"  Resolution: {w}x{h}")
        print(f"  Channels: {c}")
        print("=" * 60)
        return True
    else:
        print(f"✗ Camera {camera_index} opened but cannot read frames")
        print("=" * 60)
        return False


def export_data_summary(data_dir='data', output_file='data_summary.json'):
    """Export data collection summary to JSON"""
    counts = count_samples(data_dir)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': sum(counts.values()),
        'letters_collected': sum(1 for c in counts.values() if c > 0),
        'per_letter_counts': counts
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Data summary exported to: {output_file}")
    return summary


def clean_empty_directories(data_dir='data'):
    """Remove empty alphabet directories"""
    alphabets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    removed = []
    
    for alphabet in alphabets:
        alphabet_dir = os.path.join(data_dir, alphabet)
        if os.path.exists(alphabet_dir):
            files = [f for f in os.listdir(alphabet_dir) if f.endswith('.json')]
            if len(files) == 0:
                os.rmdir(alphabet_dir)
                removed.append(alphabet)
    
    if removed:
        print(f"\n✓ Removed empty directories: {', '.join(removed)}")
    else:
        print("\n✓ No empty directories found")
    
    return removed


def run_diagnostics():
    """Run complete system diagnostics"""
    print("\n" + "=" * 60)
    print("SIGN LANGUAGE DETECTION - SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    # Validate environment
    env_ok = validate_environment()
    
    # Test camera
    camera_ok = test_camera()
    
    # Check data
    counts = count_samples()
    data_ok = sum(counts.values()) > 0
    
    # Check model
    model_ok = check_model_exists()
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Environment: {'✓ OK' if env_ok else '✗ ISSUES FOUND'}")
    print(f"Camera: {'✓ OK' if camera_ok else '✗ ISSUES FOUND'}")
    print(f"Training Data: {'✓ OK' if data_ok else '✗ NO DATA'}")
    print(f"Model: {'✓ OK' if model_ok else '✗ NOT TRAINED'}")
    print("=" * 60)
    
    if env_ok and camera_ok and data_ok and model_ok:
        print("\n✓ System is ready for detection!")
    elif env_ok and camera_ok and data_ok:
        print("\n⚠️  Ready for training. Run: python train_model.py")
    elif env_ok and camera_ok:
        print("\n⚠️  Ready for data collection. Run: python collect_data.py")
    else:
        print("\n✗ Please resolve the issues above")
    
    print("=" * 60 + "\n")
    
    return env_ok and camera_ok


if __name__ == "__main__":
    """Run diagnostics when executed directly"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'stats':
            print_data_statistics()
        elif command == 'check':
            check_model_exists()
        elif command == 'validate':
            validate_environment()
        elif command == 'camera':
            test_camera()
        elif command == 'clean':
            clean_empty_directories()
        elif command == 'export':
            export_data_summary()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  stats    - Show data collection statistics")
            print("  check    - Check if model exists")
            print("  validate - Validate environment")
            print("  camera   - Test camera")
            print("  clean    - Clean empty directories")
            print("  export   - Export data summary")
    else:
        run_diagnostics()

