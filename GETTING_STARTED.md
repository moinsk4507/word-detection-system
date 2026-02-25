# 🚀 Getting Started with Sign Language Detection

Welcome! This guide will get you up and running in just a few steps.

## 📋 What You'll Need

- ✅ Python 3.8 or higher
- ✅ A working webcam
- ✅ 15-20 minutes for setup and data collection
- ✅ Good lighting for best results

## 🎯 Three Simple Steps

### Step 1: Install Everything (5 minutes)

Run the automated setup:

```bash
cd /Users/ansari.a/Python/sign-language
python setup.py
```

This will:
- Check your Python version
- Create necessary directories
- Install all dependencies
- Test your camera

**Alternative (Manual Setup):**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Collect Training Data (15 minutes)

```bash
python collect_data.py
```

**How to use:**
1. Position your hand clearly in the camera view
2. Press `SPACE` to start collecting samples for current letter
3. Hold the sign steady - it will collect 100 samples automatically
4. Press `N` to move to next letter
5. Repeat for as many letters as you want (minimum 5-10 recommended)

**Quick Tips:**
- ✅ Use good lighting
- ✅ Keep hand centered
- ✅ Hold sign steady
- ✅ Fill the frame well
- ❌ Avoid shadows
- ❌ Avoid cluttered background

**You can collect all 26 letters or just start with a few to test!**

### Step 3: Train & Detect (10 minutes)

**Train the model:**
```bash
python train_model.py
```

Wait 5-10 minutes while it trains. You'll see:
- Data loading progress
- Training progress bar
- Accuracy metrics
- Saved model confirmation

**Start detecting:**
```bash
# Option A: Beautiful web interface (recommended)
streamlit run app.py

# Option B: Lightweight desktop app
python detect_sign_language.py
```

## 🎉 You're Done!

Now you can:
- ✨ Show signs to your webcam
- 📺 See predictions in real-time
- 📝 Spell words using word mode
- 💾 Save detected words

## 🎮 Using the Detector

### Web Interface (Streamlit)

After running `streamlit run app.py`:

1. **Start Detection:**
   - Click "Start/Stop Detection" in sidebar
   - Your webcam will activate

2. **See Predictions:**
   - Main area shows video feed with hand landmarks
   - Right panel shows current prediction in large text
   - Confidence score displayed below
   - Top 3 predictions listed

3. **Word Mode:**
   - Toggle "Enable Word Mode" in sidebar
   - Hold each sign for 1.5 seconds
   - Letters appear in the word box
   - Use buttons to edit or save

4. **Adjust Settings:**
   - Confidence Threshold: Higher = more strict
   - Letter Hold Time: Longer = fewer mistakes

### Desktop App (OpenCV)

After running `python detect_sign_language.py`:

**Keyboard Shortcuts:**
- `W` - Toggle word spelling mode
- `SPACE` - Add space to word
- `BACKSPACE` - Delete last letter
- `C` - Clear word
- `S` - Save word to file
- `Q` - Quit

## 📊 Check Your System

Run diagnostics anytime:

```bash
python utils.py
```

This shows:
- ✓ Environment status
- ✓ Camera status
- ✓ Data collection progress
- ✓ Model status

## 🐛 Troubleshooting

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "Camera not accessible"
- Close other apps using camera (Zoom, Skype, etc.)
- Check system camera permissions
- Try a different camera: Change `CAMERA_INDEX` in config.py

### "Model not found"
Make sure you've completed both Step 2 (collect data) and Step 3 (train model).

### "Low accuracy"
- Collect more training samples (100+ per letter)
- Improve lighting conditions
- Ensure hand is fully visible
- Keep background uncluttered

## 💡 Tips for Best Results

### During Data Collection:
1. **Vary slightly** - Different hand positions, angles, distances
2. **Stay consistent** - Same general lighting and setup
3. **Be clear** - Make sure gesture is recognizable
4. **Fill frame** - Hand should be visible but not too small

### During Detection:
1. **Good lighting** - Bright, even lighting is best
2. **Clear background** - Avoid complex backgrounds
3. **Center hand** - Keep hand in center of frame
4. **Hold steady** - Reduce motion blur
5. **Correct distance** - Similar to training data

## 📚 Next Steps

Once you're comfortable with basic usage:

### Improve Your Model
```bash
# Collect more data for specific letters
python collect_data.py

# Retrain with new data
python train_model.py

# Analyze performance
python visualize_model.py
```

### Experiment
- Try different hand positions during collection
- Train on all 26 letters
- Adjust confidence thresholds
- Customize the UI colors and layout

### Learn More
- Read `README.md` for complete documentation
- Check `PROJECT_OVERVIEW.md` for technical details
- Explore `config.py` for customization options

## 🎓 ASL Alphabet Reference

Need help remembering the signs?
- Visit: https://www.startasl.com/american-sign-language-alphabet/
- Print a reference chart
- Practice with videos

## 🆘 Need Help?

1. **Check documentation:**
   - README.md - Complete reference
   - QUICKSTART.md - Fast setup
   - PROJECT_OVERVIEW.md - Technical details

2. **Run diagnostics:**
   ```bash
   python utils.py
   ```

3. **Common fixes:**
   ```bash
   # Reinstall packages
   pip install -r requirements.txt --force-reinstall
   
   # Check data
   python utils.py stats
   
   # Verify model
   python utils.py check
   ```

4. **Still stuck?**
   - Check error messages carefully
   - Read the troubleshooting section in README.md
   - Open an issue on GitHub

## 🌟 What's Possible

With this system, you can:
- Learn sign language alphabets
- Build a communication tool
- Create educational applications
- Develop accessibility features
- Extend to full words and sentences
- Integrate with other systems

## 🎯 Quick Command Reference

```bash
# Setup
python setup.py              # Automated setup
python utils.py              # Run diagnostics

# Data & Training
python collect_data.py       # Collect training data
python train_model.py        # Train the model
python visualize_model.py    # Analyze model

# Detection
streamlit run app.py         # Web interface
python detect_sign_language.py  # Desktop app

# Utilities
python utils.py stats        # Show data statistics
python utils.py check        # Check model status
python utils.py camera       # Test camera
```

## 🎊 Have Fun!

Remember, this is a learning tool. Don't worry about perfection:
- Start with a few letters
- Practice makes better
- Experiment and explore
- Build something cool!

---

**Happy signing! 🤟**

Questions? Check README.md or run `python utils.py` for diagnostics.

