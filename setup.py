"""
Setup script for Sign Language Detection project
Quick setup and verification tool
"""

import os
import sys
import subprocess


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python 3.8 or higher is required")
        print(f"  Current version: {version.major}.{version.minor}.{version.micro}")
        return False


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    dirs = ['data', 'models']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ Created directory: {dir_name}/")
        else:
            print(f"✓ Directory exists: {dir_name}/")
    
    return True


def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    print("Installing packages from requirements.txt...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Failed to install dependencies")
        print("  Try manually: pip install -r requirements.txt")
        return False


def verify_installation():
    """Verify that all packages are installed"""
    print_header("Verifying Installation")
    
    required_packages = [
        'cv2',
        'mediapipe',
        'tensorflow',
        'numpy',
        'sklearn',
        'streamlit',
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            all_installed = False
    
    return all_installed


def test_camera():
    """Quick camera test"""
    print_header("Testing Camera")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                h, w, _ = frame.shape
                print(f"✓ Camera is working!")
                print(f"  Resolution: {w}x{h}")
                return True
            else:
                print("✗ Camera opened but cannot read frames")
                return False
        else:
            print("✗ Cannot access camera")
            print("  Make sure camera is not being used by another application")
            return False
    except Exception as e:
        print(f"✗ Error testing camera: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print("Next steps to get started:\n")
    print("1. Collect Training Data:")
    print("   python collect_data.py")
    print("   - Collect at least 100 samples per letter")
    print("   - Takes about 15-20 minutes\n")
    
    print("2. Train the Model:")
    print("   python train_model.py")
    print("   - Trains on your collected data")
    print("   - Takes about 5-10 minutes\n")
    
    print("3. Run Detection:")
    print("   Option A - Web Interface (Recommended):")
    print("   streamlit run app.py")
    print()
    print("   Option B - Desktop Application:")
    print("   python detect_sign_language.py\n")
    
    print("For more information, see:")
    print("  - README.md for detailed documentation")
    print("  - QUICKSTART.md for quick start guide")
    print()
    print("Run diagnostics anytime with:")
    print("  python utils.py")
    print()


def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("SIGN LANGUAGE DETECTION - SETUP".center(60))
    print("=" * 60)
    
    print("\nThis script will:")
    print("  1. Check Python version")
    print("  2. Create necessary directories")
    print("  3. Install dependencies")
    print("  4. Verify installation")
    print("  5. Test camera")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    # Run setup steps
    steps = [
        ("Python Version Check", check_python_version),
        ("Directory Creation", create_directories),
        ("Dependency Installation", install_dependencies),
        ("Installation Verification", verify_installation),
        ("Camera Test", test_camera),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"\n✗ Error during {step_name}: {e}")
            results.append((step_name, False))
    
    # Print summary
    print_header("Setup Summary")
    
    for step_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{step_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ Setup completed successfully!")
        print_next_steps()
    else:
        print("\n✗ Setup completed with errors")
        print("Please resolve the issues above before proceeding.")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

