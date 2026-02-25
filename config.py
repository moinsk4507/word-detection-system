"""
Configuration file for Sign Language Detection project
Adjust these parameters to customize behavior
"""

# ========================================
# Paths
# ========================================
DATA_DIR = 'data'
MODEL_DIR = 'models'
MODEL_PATH = 'models/sign_language_model.keras'
PARAMS_PATH = 'models/preprocessing_params.json'
OUTPUT_FILE = 'detected_words.txt'

# ========================================
# Data Collection Settings
# ========================================
SAMPLES_PER_ALPHABET = 100  # Number of samples to collect per letter
ALPHABETS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # Letters to collect

# ========================================
# Camera Settings
# ========================================
CAMERA_INDEX = 0  # Change if you have multiple cameras
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ========================================
# MediaPipe Hand Detection Settings
# ========================================
# Detection confidence (0.0 - 1.0)
# Higher = more strict detection (fewer false positives)
MIN_DETECTION_CONFIDENCE = 0.7

# Tracking confidence (0.0 - 1.0)
# Higher = more stable tracking (less jitter)
MIN_TRACKING_CONFIDENCE = 0.7

# Maximum number of hands to detect
MAX_NUM_HANDS = 1

# ========================================
# Model Architecture Settings
# ========================================
# Neural network layer sizes
LAYER_SIZES = [256, 128, 64]

# Dropout rates for each layer
DROPOUT_RATES = [0.3, 0.3, 0.2]

# Use batch normalization
USE_BATCH_NORM = True

# ========================================
# Training Settings
# ========================================
# Number of training epochs
EPOCHS = 100

# Batch size for training
BATCH_SIZE = 32

# Initial learning rate
LEARNING_RATE = 0.001

# Test/validation split ratio
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# Early stopping patience (epochs)
EARLY_STOPPING_PATIENCE = 15

# Learning rate reduction patience (epochs)
LR_REDUCTION_PATIENCE = 5

# Learning rate reduction factor
LR_REDUCTION_FACTOR = 0.5

# Minimum learning rate
MIN_LEARNING_RATE = 1e-6

# ========================================
# Detection/Inference Settings
# ========================================
# Confidence threshold for displaying predictions (0.0 - 1.0)
# Only predictions above this will be shown
CONFIDENCE_THRESHOLD = 0.6

# Number of predictions to keep for smoothing
PREDICTION_HISTORY_SIZE = 5

# Minimum number of consistent predictions needed
MIN_CONSISTENT_PREDICTIONS = 3

# ========================================
# Word Spelling Mode Settings
# ========================================
# Time (seconds) to hold a sign before adding to word
LETTER_HOLD_TIME = 1.5

# Enable word mode by default
DEFAULT_WORD_MODE = False

# ========================================
# Visualization Settings
# ========================================
# Hand landmark colors (B, G, R)
LANDMARK_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (255, 0, 0)  # Red/Blue

# Landmark drawing specs
LANDMARK_THICKNESS = 2
LANDMARK_RADIUS = 2
CONNECTION_THICKNESS = 2

# Bounding box color and thickness
BBOX_COLOR = (0, 255, 0)  # Green
BBOX_THICKNESS = 2
BBOX_PADDING = 20

# Text display settings
# Note: OpenCV doesn't have FONT_HERSHEY_BOLD - use FONT_HERSHEY_SIMPLEX with higher thickness
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE = 3
FONT_SCALE_MEDIUM = 1.2
FONT_SCALE_SMALL = 0.7
FONT_THICKNESS_LARGE = 5
FONT_THICKNESS_MEDIUM = 3
FONT_THICKNESS_SMALL = 2

# Colors
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (200, 200, 200)

# ========================================
# Streamlit Settings
# ========================================
# Page title
PAGE_TITLE = "Sign Language Detector"

# Page icon
PAGE_ICON = "🤟"

# Layout
LAYOUT = "wide"

# Frame rate for video display (FPS)
STREAMLIT_FPS = 30

# ========================================
# Feature Flags
# ========================================
# Enable/disable features
ENABLE_WORD_MODE = True
ENABLE_TOP_PREDICTIONS = True
ENABLE_CONFIDENCE_DISPLAY = True
ENABLE_HAND_LANDMARKS = True
ENABLE_BOUNDING_BOX = True

# Development mode (more verbose logging)
DEBUG_MODE = False

# Save prediction history for analysis
SAVE_PREDICTIONS = False
PREDICTIONS_LOG_FILE = 'predictions.log'

