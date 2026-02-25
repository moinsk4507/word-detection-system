# 🚀 Quick Start Guide

Get started with Sign Language Detection in 5 minutes!

## Prerequisites

- Python 3.8+
- Webcam
- 15-20 minutes for data collection
- 5-10 minutes for training

## Installation (2 minutes)

```bash
# Navigate to project directory
cd /Users/ansari.a/Python/sign-language

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step-by-Step Setup

### 1️⃣ Collect Training Data (15-20 min)

```bash
python collect_data.py
```

**Quick Tips**:
- Position your hand clearly in camera view
- Press `SPACE` to start collecting samples
- Hold each sign steady - it collects 100 samples automatically
- Press `N` to move to next letter
- You can skip letters you don't want to train on

**Minimum Recommended**:
- At least 5-10 letters to start
- 100 samples per letter (automatic)

### 2️⃣ Train the Model (5-10 min)

```bash
python train_model.py
```

This will:
- Load your collected data
- Train a neural network
- Save the model automatically
- Show accuracy metrics

**Expected Results**:
- Training accuracy: 90%+ (with good data)
- Training time: 2-5 minutes on CPU

### 3️⃣ Start Detecting! (30 sec)

**Option A - Web Interface** (Easiest):
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser

**Option B - Desktop App**:
```bash
python detect_sign_language.py
```

## First Detection Test

1. Launch the detector (either interface)
2. Show your hand to the camera
3. Make a sign you trained on
4. See the prediction appear!

## Common Issues

**"No module named 'cv2'"**
```bash
pip install opencv-python
```

**"Camera not opening"**
- Close other apps using the camera
- Check camera permissions in System Settings

**"Model not found"**
- Make sure you completed steps 1 and 2
- Check that `models/` directory exists

**"Low accuracy"**
- Collect more training samples
- Use better lighting
- Keep hand in center of frame

## Next Steps

Once you're comfortable:

1. **Improve Accuracy**:
   - Collect more diverse samples
   - Train with all 26 letters
   - Add variations (different angles, lighting)

2. **Try Word Mode**:
   - Press `W` in the detector
   - Hold signs to spell words
   - Save words with `S`

3. **Customize**:
   - Adjust confidence threshold
   - Modify hold time for word mode
   - Change model architecture

## Need Help?

- Check the full README.md for detailed documentation
- Open an issue on GitHub
- Review troubleshooting section in README

---

**Happy Detecting! 🤟**

