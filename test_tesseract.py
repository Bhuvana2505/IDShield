#!/usr/bin/env python3
"""
Tesseract OCR Test Script
This script verifies that Tesseract OCR is properly installed and configured.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    print("✅ Required libraries are installed.")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install required libraries using: pip install -r requirements.txt")
    sys.exit(1)

# Check if Tesseract is installed via command line
def check_tesseract_command():
    try:
        result = subprocess.run(['tesseract', '--version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True, 
                               check=False)
        if result.returncode == 0:
            print(f"✅ Tesseract is installed and accessible from command line.")
            print(f"Version info: {result.stdout.splitlines()[0]}")
            return True
        else:
            print("❌ Tesseract command failed with error:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running tesseract command: {e}")
        return False

# Check Tesseract configuration in pytesseract
def check_pytesseract_config():
    try:
        # Print current Tesseract path
        print(f"Current pytesseract configuration:")
        print(f"Tesseract command: {pytesseract.pytesseract.tesseract_cmd}")
        
        # Try to find Tesseract in common installation locations
        if os.name == 'nt':  # Windows
            tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                os.environ.get('TESSERACT_PATH', '')
            ]
            
            for path in tesseract_paths:
                if path and os.path.exists(path):
                    print(f"✅ Found Tesseract at: {path}")
                    if path != pytesseract.pytesseract.tesseract_cmd:
                        print(f"ℹ️ Consider setting pytesseract.pytesseract.tesseract_cmd to this path")
                    return True
            
            print("❌ Tesseract not found in common installation locations")
            return False
        else:  # Unix-like systems
            return True  # Assume it's in PATH
    except Exception as e:
        print(f"❌ Error checking pytesseract configuration: {e}")
        return False

# Test OCR functionality with a simple image
def test_ocr_functionality():
    try:
        # Create a simple test image with text
        img = np.zeros((100, 300), dtype=np.uint8) + 255  # White background
        cv2.putText(img, "Hello World", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text
        
        # Save temporary image
        temp_img_path = os.path.join(project_root, "temp", "test_ocr.png")
        os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
        cv2.imwrite(temp_img_path, img)
        
        # Perform OCR
        text = pytesseract.image_to_string(Image.open(temp_img_path))
        
        # Check result
        if "Hello" in text and "World" in text:
            print(f"✅ OCR test successful! Detected text: {text.strip()}")
            return True
        else:
            print(f"❌ OCR test failed. Detected text: {text.strip()}")
            return False
    except Exception as e:
        print(f"❌ Error testing OCR functionality: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

def main():
    print("=== Tesseract OCR Configuration Test ===")
    
    # Run tests
    cmd_check = check_tesseract_command()
    config_check = check_pytesseract_config()
    
    if cmd_check and config_check:
        print("\n🔍 Testing OCR functionality...")
        ocr_check = test_ocr_functionality()
        
        if ocr_check:
            print("\n✅ All tests passed! Tesseract OCR is properly configured.")
        else:
            print("\n⚠️ OCR functionality test failed. Please check Tesseract installation.")
    else:
        print("\n⚠️ Tesseract configuration issues detected. Please fix before testing OCR functionality.")
    
    print("\n=== Recommendations ===")
    if not cmd_check:
        print("1. Ensure Tesseract is installed correctly")
        print("2. Add Tesseract to your system PATH")
        print("3. Restart your terminal/IDE after installation")
    
    if not config_check:
        print("1. Set pytesseract.pytesseract.tesseract_cmd to the correct path in your code")
        print("2. Check if Tesseract is installed in a non-standard location")

if __name__ == "__main__":
    main()