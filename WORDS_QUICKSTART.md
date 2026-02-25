# 🚀 Word Detection Quick Start

Get started with sign language **word detection** in 3 steps!

## What's the Difference?

### Alphabet Detection (Already Set Up)
- ✅ Detects letters A-Z
- ✅ Used for spelling words letter-by-letter
- ✅ You've already collected: C, D, E, F, G, I, L

### Word Detection (New!)
- ✨ Detects complete words/phrases
- ✨ Recognizes whole gestures like "HELLO", "THANK YOU"
- ✨ Faster communication than spelling
- ✨ Supports one-handed and two-handed signs

## Quick Start (30 minutes)

### Step 1: Learn 5 Basic Signs (10 min)

Visit https://www.lifeprint.com/ and learn these 5 signs:

1. **HELLO** - Simple wave
2. **THANK_YOU** - Hand from chin forward
3. **YES** - Fist nods
4. **NO** - Two fingers close
5. **HELP** - Fist on palm, lift up

**Important**: Watch videos multiple times and practice!

### Step 2: Collect Data (15 min)

```bash
python collect_words.py
```

**Process**:
1. Program shows "HELLO" on screen
2. Make the HELLO sign
3. Press SPACE to start collecting
4. Hold sign steady (collects 100 samples automatically)
5. Press N for next word
6. Repeat for all 5 words

**Tips**:
- Keep both hands visible
- Good lighting
- Center yourself in frame
- Press S to skip unknown words

### Step 3: Train & Test (5 min)

Train the model:
```bash
python train_words.py
```

Test it:
```bash
python detect_words.py
```

## Controls

### During Data Collection (`collect_words.py`)
- `SPACE` - Start/stop collecting
- `N` - Next word
- `P` - Previous word  
- `S` - Skip word
- `Q` - Quit

### During Detection (`detect_words.py`)
- `S` - Toggle sentence building mode
- `SPACE` - Clear sentence
- `BACKSPACE` - Remove last word
- `V` - Save sentence to file
- `Q` - Quit

## What You Can Do

### Basic Detection
Just show a sign → See it detected with confidence score

### Sentence Building
1. Press `S` to enable sentence mode
2. Show first sign, hold for 2 seconds → Added to sentence
3. Show next sign, hold for 2 seconds → Added to sentence
4. Press `V` to save complete sentence

Example: HELLO → MY_NAME_IS → [fingerspell name]

## Recommended Learning Path

### Day 1: Basic Words (5 words)
- HELLO
- THANK_YOU
- YES
- NO
- HELP

### Day 2: Needs (5 words)
- PLEASE
- SORRY
- BATHROOM
- WATER
- STOP

### Day 3: Questions (5 words)
- WHAT
- WHERE
- WHEN
- WHO
- HOW

### Day 4: Emotions (5 words)
- HAPPY
- SAD
- GOOD
- BAD
- I_LOVE_YOU

## Common Questions

### Q: Do I need to learn all 40+ words at once?
**A**: No! Start with 5-10 words. You can always add more later.

### Q: What if I make the sign wrong during collection?
**A**: The model will learn your version. Best to learn correctly first!

### Q: Can I mix alphabets and words?
**A**: Yes! Use alphabets for spelling names, words for common phrases.

### Q: How long does training take?
**A**: 5-15 minutes for 5-10 words. Longer for more words.

### Q: What accuracy should I expect?
**A**: 
- 5 words: 90%+ accuracy
- 10 words: 85%+ accuracy
- 20+ words: 80%+ accuracy

## Troubleshooting

### "No hands detected"
- Ensure camera is working
- Check lighting
- Move closer to camera
- Keep hands in frame

### "Low accuracy"
- Re-learn signs correctly
- Collect more samples
- Use consistent lighting
- Avoid similar signs initially

### "Model not found"
- Run `train_words.py` after collecting data
- Check `models/` directory exists

## Resources

### Learn Signs
- **ASL University**: https://www.lifeprint.com/
- **SignASL**: https://www.signasl.org/
- **YouTube**: Search "ASL [word name]"

### Documentation
- `WORD_GUIDE.md` - Complete word detection guide
- `README.md` - Project overview
- `GETTING_STARTED.md` - Alphabet detection guide

## Tips for Success

### Before Collection
✅ Learn signs correctly from ASL resources
✅ Practice until confident  
✅ Understand hand shapes and positions
✅ Know if sign uses one or two hands

### During Collection
✅ Good, even lighting
✅ Plain background
✅ Both hands visible (if needed)
✅ Vary position slightly (not too much)
✅ Hold sign steady

### After Training
✅ Test with each word individually
✅ Check confidence scores
✅ Practice challenging signs more
✅ Add more words gradually

## Example Session

```bash
# 1. Learn 5 signs (10 minutes on lifeprint.com)
# Watch videos for: HELLO, THANK_YOU, YES, NO, HELP

# 2. Collect data (15 minutes)
python collect_words.py
# Collect 100 samples each for your 5 words

# 3. Train model (5 minutes)
python train_words.py
# Wait for training to complete

# 4. Test it! (play around)
python detect_words.py
# Try each sign, test sentence building mode
```

## What's Next?

After mastering basics:
1. Add more words (5 at a time)
2. Practice sentence building
3. Combine with alphabet detection for names
4. Share your experience!

---

**Ready to start? Run `python collect_words.py`! 🤟**

*Remember: Quality over quantity. 5 well-learned signs are better than 20 poorly-learned ones!*

