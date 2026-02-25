# 🤟 Sign Language Alphabet Recognition

A real-time sign language alphabet (A-Z) recognition system using computer vision and deep learning. This project uses OpenCV, MediaPipe, and TensorFlow to detect and classify American Sign Language (ASL) hand gestures through your webcam.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **Real-time Detection**: Recognizes sign language alphabets from live webcam feed
- **High Accuracy**: Uses MediaPipe hand landmarks and deep neural networks
- **Visual Feedback**: 
  - Hand landmark visualization
  - Bounding boxes around detected hands
  - Confidence scores for predictions
  - Top-3 prediction display
- **Word Spelling Mode**: Spell out complete words by continuously detecting letters
- **Save Functionality**: Save detected words to a text file
- **Two Interfaces**:
  - OpenCV-based standalone application
  - Beautiful Streamlit web interface
- **Data Collection Tool**: Easy-to-use tool to collect your own training data

## 📋 Requirements

- Python 3.8 or higher
- Webcam
- Good lighting conditions for better detection

## 🚀 Installation

1. **Clone the repository**:
```bash
cd C:\Users\moins\OneDrive\Desktop\word-detection
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📦 Project Structure

```
sign-language/
├── app.py                      # Streamlit web interface
├── detect_sign_language.py     # OpenCV standalone detector
├── collect_data.py             # Data collection tool
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── models/                     # Trained models directory
│   ├── sign_language_model.keras
│   └── preprocessing_params.json
├── data/                       # Training data directory
│   ├── A/
│   ├── B/
│   └── ...
└── README.md                   # This file
```

## 🎯 Usage

### Step 1: Collect Training Data

First, collect hand gesture samples for each alphabet (A-Z):

```bash
python collect_data.py
```

**Instructions**:
- Position your hand clearly in front of the camera
- Press `SPACE` to start/stop collecting samples for current alphabet
- The tool will collect 100 samples per alphabet
- Press `N` to move to next alphabet, `P` for previous
- Press `Q` to quit

**Tips for better data**:
- Use good lighting
- Keep hand in center of frame
- Vary hand positions slightly (different angles, distances)
- Make sure the hand gesture is clearly visible

### Step 2: Train the Model

Once you've collected enough data (recommended: at least 100 samples per alphabet):

```bash
python train_model.py
```

This will:
- Load all collected samples
- Preprocess the data
- Train a neural network
- Evaluate the model
- Save the trained model and parameters
- Generate training history plots

**Training outputs**:
- `models/sign_language_model.keras` - Trained model
- `models/preprocessing_params.json` - Normalization parameters
- `models/training_history.png` - Training/validation curves

### Step 3: Run Detection

You have two options:

#### Option A: Streamlit Web Interface (Recommended)

```bash
streamlit run app.py
```

**Features**:
- User-friendly web interface
- Start/stop detection with a button
- Visual feedback with confidence scores
- Word spelling mode with real-time updates
- Easy controls for word manipulation

#### Option B: OpenCV Standalone Application

```bash
python detect_sign_language.py
```

**Keyboard Controls**:
- `W` - Toggle word spelling mode
- `SPACE` - Add space to word (in word mode)
- `BACKSPACE` - Delete last letter
- `C` - Clear current word
- `S` - Save word to file
- `Q` - Quit

## 🎨 How It Works

### Architecture

1. **Hand Detection**: MediaPipe Hands detects hand landmarks (21 points per hand)
2. **Feature Extraction**: Extracts 3D coordinates (x, y, z) of all landmarks (63 features)
3. **Preprocessing**: Normalizes features using training statistics
4. **Classification**: Deep neural network predicts the alphabet
5. **Smoothing**: Temporal smoothing for stable predictions

### Model Architecture

```
Input (63 features)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + Dropout(0.2)
    ↓
Dense(26, softmax) → Output (A-Z)
```

### MediaPipe Hand Landmarks

The system tracks 21 hand landmarks:

```
0: Wrist
1-4: Thumb (CMC, MCP, IP, Tip)
5-8: Index finger (MCP, PIP, DIP, Tip)
9-12: Middle finger (MCP, PIP, DIP, Tip)
13-16: Ring finger (MCP, PIP, DIP, Tip)
17-20: Pinky (MCP, PIP, DIP, Tip)
```

## 🔧 Configuration

### Adjust Detection Sensitivity

Edit detection parameters in the code:

```python
# In detect_sign_language.py or app.py
self.hands = self.mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Adjust this (0.0-1.0)
    min_tracking_confidence=0.7    # Adjust this (0.0-1.0)
)

self.confidence_threshold = 0.6  # Minimum confidence for display
```

### Modify Training Parameters

Edit training settings in `train_model.py`:

```python
epochs=100,           # Number of training epochs
batch_size=32,        # Batch size
learning_rate=0.001   # Initial learning rate
```

## 📊 Performance Tips

### For Better Detection:
- Ensure good lighting conditions
- Keep hand centered in frame
- Maintain consistent distance from camera
- Avoid cluttered backgrounds
- Keep hand steady for 1-2 seconds

### For Better Model:
- Collect more diverse training samples
- Include variations in lighting, angle, distance
- Balance dataset (equal samples per alphabet)
- Augment data if needed
- Fine-tune model architecture

## 🎓 ASL Alphabet Reference

For reference on ASL hand signs, see:
- [ASL Alphabet Chart](https://www.startasl.com/american-sign-language-alphabet/)
- Note: Letters J and Z require motion and may be harder to detect

## 🐛 Troubleshooting

### Camera Not Opening
- Check if another application is using the camera
- Try changing camera index in code: `cv2.VideoCapture(1)`
- Verify camera permissions on your system

### Low Accuracy
- Collect more training data
- Ensure training data quality (clear gestures, good lighting)
- Check if hand is fully visible in frame
- Verify all 21 landmarks are detected

### Model Not Found Error
- Make sure you've completed Steps 1 and 2
- Check if model files exist in `models/` directory
- Verify file paths in code

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Try upgrading pip: `pip install --upgrade pip`

## 🔮 Future Enhancements

- [ ] Support for more sign languages (BSL, ISL, etc.)
- [ ] Word prediction and autocorrect
- [ ] Sentence formation with grammar
- [ ] Mobile app version
- [ ] Two-hand gesture support
- [ ] Motion-based letters (J, Z)
- [ ] Real-time translation mode
- [ ] Cloud-based model serving
- [ ] Export to different formats (JSON, CSV)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Areas for contribution:
- Improve model architecture
- Add data augmentation
- Support for more languages
- Better UI/UX
- Documentation improvements
- Bug fixes

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe**: Google's hand tracking solution
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library
- **Streamlit**: Web app framework
- ASL community for sign language resources

## 📧 Contact

For questions, suggestions, or issues:
- Open an issue on GitHub
- Submit a pull request
- Contact the maintainer

## 🌟 Show Your Support

If you find this project helpful:
- ⭐ Star the repository
- 🐛 Report bugs
- 💡 Suggest new features
- 📢 Share with others

---

**Happy Signing! 🤟**

Made with ❤️ for the deaf and hard-of-hearing community

