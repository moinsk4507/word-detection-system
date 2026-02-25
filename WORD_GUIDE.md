# 🤟 Sign Language Word/Phrase Detection Guide

## Overview

This guide covers the **word/phrase detection** system, which recognizes complete ASL signs (not letter-by-letter spelling).

## Difference: Words vs. Alphabets

| Alphabet Detection | Word Detection |
|-------------------|----------------|
| Individual letters A-Z | Complete words/phrases |
| One hand | One or two hands |
| Finger spelling | Whole gestures |
| Example: C-A-T | Example: "HELLO" gesture |

## Supported Words/Phrases

The system currently supports these common ASL words:

### Greetings & Social
- **HELLO** - Wave hand
- **GOODBYE** - Wave goodbye
- **THANK_YOU** - Hand from chin outward
- **PLEASE** - Circle on chest
- **SORRY** - Fist circles on chest
- **NICE_TO_MEET_YOU** - Combined gesture
- **MY_NAME_IS** - Combined gesture

### Yes/No Questions
- **YES** - Fist nods like head
- **NO** - Index and middle finger close like mouth
- **MAYBE** / **SORT_OF**

### Emotions & States
- **HAPPY** - Brush hands up chest
- **SAD** - Hands slide down face
- **ANGRY** - Claw hand from face
- **SCARED** - Hands open in front of body
- **TIRED** - Hands drop on chest
- **SICK** - Hand on forehead and stomach
- **PAIN** - Index fingers twist together

### Needs & Requests
- **HELP** - Fist on flat palm, lift up
- **STOP** - Flat hand chops down
- **MORE** - Fingertips tap together
- **BATHROOM** - Shake "T" hand
- **WATER** - "W" hand to mouth
- **FOOD** / **EAT** - Fingers to mouth
- **DRINK** - "C" hand to mouth
- **HUNGRY** - "C" hand down chest
- **THIRSTY** - Finger down throat

### Question Words
- **HOW** - Hands roll around each other
- **WHAT** - Shake hands palms up
- **WHERE** - Point and shake
- **WHEN** - Point and circle
- **WHO** - Circle around mouth
- **WHY** - Wiggle middle finger on forehead

### Time
- **TODAY** - "Now" + "day"
- **TOMORROW** - Thumb forward from cheek
- **YESTERDAY** - Thumb backward from cheek
- **MORNING** - Arm rises like sun
- **AFTERNOON** - Arm horizontal
- **NIGHT** - Arm sets like sun

### Family
- **FAMILY** - "F" hands circle around
- **MOTHER** - "5" hand to chin
- **FATHER** - "5" hand to forehead
- **FRIEND** - Hook index fingers together

### Common Words
- **GOOD** - Flat hand from mouth outward
- **BAD** - Hand from mouth, flip down
- **I_LOVE_YOU** - ILY hand shape (thumb, index, pinky up)

## Learning Resources

### Online Resources
1. **ASL University** - https://www.lifeprint.com/
   - Free video dictionary
   - Complete sign database

2. **SignASL.org** - https://www.signasl.org/
   - Sign videos with descriptions
   - Mobile-friendly

3. **Handspeak** - https://www.handspeak.com/
   - ASL dictionary
   - Sign language resources

4. **YouTube Channels**:
   - "Bill Vicars" (ASL University)
   - "Learn How to Sign"
   - "Sign Language 101"

### Books
- "The American Sign Language Phrase Book" by Lou Fant
- "Signing Naturally" (Student Workbook and DVD)
- "Master ASL!" by Jason E. Zinza

## Data Collection Tips

### Before You Start
1. **Learn the signs correctly first!**
   - Watch multiple videos
   - Practice until confident
   - Verify with ASL dictionary

2. **Understand the motion**:
   - Some signs are static (hold position)
   - Some signs have movement (this system captures static poses)
   - Focus on the end position of the sign

3. **Check hand requirements**:
   - One hand or two hands?
   - Dominant hand matters?
   - Hand orientation important?

### During Collection

#### Setup
- ✅ Good, even lighting
- ✅ Plain background
- ✅ Centered in frame
- ✅ Both hands visible (if two-handed sign)

#### Variety
For each word, vary:
- **Position**: Slight left/right, up/down
- **Distance**: Closer and farther from camera
- **Angle**: Slight rotations
- **Hand size**: Different people if possible

#### What to Avoid
- ❌ Shadows on hands
- ❌ Hands cut off by frame
- ❌ Cluttered background
- ❌ Poor lighting
- ❌ Motion blur
- ❌ Wrong hand shape

### Quality Checks
After collecting 100 samples for a word:
1. Review if you're still making the sign correctly
2. Ensure variety in your samples
3. Check that all samples are visible and clear
4. Verify you haven't confused similar signs

## Usage Workflow

### 1. Learn Signs (1-2 hours per word group)
```bash
# Visit ASL learning resources
# Watch videos for words you want to collect
# Practice until confident
```

### 2. Collect Data (15-20 minutes per word)
```bash
python collect_words.py
```

**Tips**:
- Start with 5-10 common words
- Collect 100 samples per word (automatic)
- Press `S` to skip words you don't know yet
- Take breaks between words

### 3. Train Model (10-20 minutes)
```bash
python train_words.py
```

**Expected results**:
- 5 words: 85-90% accuracy
- 10 words: 80-85% accuracy
- 20+ words: 75-85% accuracy

### 4. Detect Words (Real-time)
```bash
python detect_words.py
```

**Features**:
- Real-time word recognition
- Confidence scores
- Sentence building mode
- Save sentences to file

## Common Signs Confusion

### Similar Signs to Watch For
- **GOOD** vs **THANK YOU**: Direction matters
- **PLEASE** vs **SORRY**: Circle size and location
- **HELLO** vs **GOODBYE**: Wave direction
- **MOTHER** vs **FATHER**: Forehead vs chin
- **MORNING** vs **AFTERNOON** vs **NIGHT**: Arm angle

### Best Practices
1. Collect very distinct words first
2. Add similar signs later after model is trained
3. Collect extra samples for easily confused signs
4. Test frequently to verify accuracy

## Sentence Building

The detector has a **sentence building mode** that lets you create complete sentences:

### How It Works
1. Press `S` to enable sentence mode
2. Make a sign and hold it for 2 seconds
3. Word automatically adds to sentence
4. Repeat for next word
5. Press `V` to save sentence to file

### Example Sentences
- "HELLO MY NAME IS [fingerspell]"
- "I NEED HELP"
- "THANK YOU VERY MUCH"
- "WHERE BATHROOM"
- "I HUNGRY"
- "SORRY I LATE"

## Limitations

### Current System
- ✅ Detects static hand positions
- ❌ Does NOT capture motion (yet)
- ❌ Some signs require motion to be accurate

### Signs That May Not Work Well
Signs with essential motion components:
- Movement-based signs
- Signs requiring facial expressions
- Signs with specific timing
- Continuous motion signs

### Future Improvements
- Motion tracking (sequence of frames)
- Facial expression recognition
- Non-manual markers
- Direction and speed of movement

## Troubleshooting

### Low Accuracy
**Problem**: Model often incorrect

**Solutions**:
1. Verify you learned signs correctly
2. Collect more diverse samples
3. Use better lighting
4. Check if hands are fully visible
5. Start with fewer, distinct words

### Signs Not Detected
**Problem**: No prediction showing

**Solutions**:
1. Check confidence threshold (default 0.65)
2. Ensure both hands visible (if two-handed)
3. Hold sign steadily for 2-3 seconds
4. Move closer or farther from camera
5. Improve lighting

### Wrong Word Detected
**Problem**: Always shows wrong word

**Solutions**:
1. Re-learn the correct sign
2. Collect more samples for correct word
3. Remove samples for incorrect word if collected wrong
4. Train model again

## Performance Tips

### For Better Recognition
1. **Lighting**: Bright, even, no shadows
2. **Background**: Plain, uncluttered
3. **Position**: Center of frame, both hands visible
4. **Stability**: Hold sign steady, no motion
5. **Consistency**: Match training data conditions

### For Faster Detection
1. Use fewer words in model (5-10 is ideal)
2. Collect more samples per word (150-200)
3. Ensure high-quality training data
4. Use good hardware (better camera)

## Next Steps

### Beginner Path
1. Start with 5 common words:
   - HELLO
   - THANK_YOU
   - YES
   - NO
   - HELP

2. Collect 100 samples each (2 hours)
3. Train model (15 minutes)
4. Test and practice

### Intermediate Path
1. Add 10-15 common words
2. Include question words (WHO, WHAT, WHERE, WHEN, WHY, HOW)
3. Add emotions (HAPPY, SAD, ANGRY)
4. Practice sentence building

### Advanced Path
1. Collect 20-30 words
2. Create themed vocabularies (food, family, time)
3. Build complex sentences
4. Contribute signs to community

## Contributing

Want to help improve word detection?

1. **Collect diverse data**:
   - Different lighting conditions
   - Different people
   - Different camera angles

2. **Document signs**:
   - Create video tutorials
   - Share tips and tricks
   - Report confusion pairs

3. **Improve code**:
   - Add motion detection
   - Improve accuracy
   - Better visualization

## Resources Summary

### Quick Links
- **Learn**: https://www.lifeprint.com/
- **Practice**: https://www.signasl.org/
- **Verify**: https://www.handspeak.com/

### Documentation
- `README.md` - Project overview
- `GETTING_STARTED.md` - Quick start for alphabets
- `WORD_GUIDE.md` - This file (words)
- `WORKFLOW.txt` - Technical workflow

### Scripts
- `collect_words.py` - Collect word data
- `train_words.py` - Train word model
- `detect_words.py` - Detect words in real-time

---

**Happy Signing! 🤟**

*Remember: Learning sign language is a journey. Start small, practice often, and have fun!*

