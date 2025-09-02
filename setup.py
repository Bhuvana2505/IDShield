import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_tesseract():
    """Install Tesseract OCR based on operating system"""
    system = platform.system().lower()
    
    if system == "linux":
        print("📦 Installing Tesseract OCR for Linux...")
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "tesseract-ocr"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "libtesseract-dev"], check=True)
            print("✅ Tesseract OCR installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Tesseract OCR. Please install manually:")
            print("   sudo apt-get install tesseract-ocr libtesseract-dev")
            
    elif system == "darwin":  # macOS
        print("📦 Installing Tesseract OCR for macOS...")
        try:
            subprocess.run(["brew", "install", "tesseract"], check=True)
            print("✅ Tesseract OCR installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Tesseract OCR. Please install manually:")
            print("   brew install tesseract")
            
    elif system == "windows":
        print("📦 For Windows, please install Tesseract OCR manually:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install and add to PATH")
        print("   3. Restart your terminal")
        
    else:
        print(f"❌ Unsupported operating system: {system}")

def install_python_dependencies():
    """Install Python package dependencies"""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "ocr_engine",
        "redactor", 
        "utils",
        "output",
        "logs",
        "temp"
    ]
    
    print("📁 Creating directory structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory in ['ocr_engine', 'redactor', 'utils']:
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write(f'"""IDShield {directory} module"""\n')
    
    print("✅ Directory structure created")

def verify_installation():
    """Verify that all components are properly installed"""
    print("🔍 Verifying installation...")
    
    try:
        # Test Tesseract
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR working")
    except Exception as e:
        print(f"❌ Tesseract OCR issue: {e}")
        return False
    
    try:
        # Test OpenCV
        import cv2
        print(f"✅ OpenCV {cv2.__version__} working")
    except Exception as e:
        print(f"❌ OpenCV issue: {e}")
        return False
    
    try:
        # Test other key dependencies
        import streamlit
        import PIL
        import fitz
        print("✅ All core dependencies working")
    except Exception as e:
        print(f"❌ Dependency issue: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🛡️  IDShield Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directory_structure()
    
    # Install Tesseract OCR
    install_tesseract()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Verify installation
    if verify_installation():
        print("\n🎉 IDShield setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run: streamlit run main.py")
        print("   2. Open your browser to the displayed URL")
        print("   3. Upload a document to start redacting")
        print("\n📖 For detailed usage instructions, see README.md")
    else:
        print("\n❌ Setup completed with errors. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()