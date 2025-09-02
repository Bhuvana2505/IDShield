
import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('streamlit', 'streamlit'),
        ('opencv-python', 'cv2'), 
        ('pytesseract', 'pytesseract'),
        ('pillow', 'PIL'), 
        ('numpy', 'numpy'), 
        ('pandas', 'pandas')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        print(f"❌ Tesseract OCR not found: {e}")
        print("💡 Please install Tesseract OCR:")
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   macOS: brew install tesseract")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def create_required_directories():
    """Create necessary directories if they don't exist"""
    directories = ['output', 'logs', 'temp']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Required directories created/verified")

def main():
    """Main launcher function"""
    print("🛡️  Starting IDShield Document Redaction System")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("❌ main.py not found. Please run this script from the IDShield directory.")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check Tesseract
    print("🔍 Checking Tesseract OCR...")
    if not check_tesseract():
        sys.exit(1)
    
    # Create directories
    print("📁 Setting up directories...")
    create_required_directories()
    
    print("✅ All checks passed!")
    print("🚀 Launching IDShield...")
    print("\n" + "="*55)
    print("📱 IDShield will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("="*55 + "\n")
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 IDShield stopped. Thank you for using IDShield!")
    except Exception as e:
        print(f"\n❌ Error starting IDShield: {e}")
        print("💡 Try running manually: streamlit run main.py")

if __name__ == "__main__":
    main()