import os

# Application Settings
APP_NAME = "IDShield"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Secure Document Redaction System"

# OCR Settings
TESSERACT_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/-: '
OCR_CONFIDENCE_THRESHOLD = 30
DPI_SETTING = 200

# Entity Detection Settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
MINIMUM_FACE_SIZE = (50, 50)
MINIMUM_SIGNATURE_AREA = 500
MAXIMUM_SIGNATURE_AREA = 10000

# Redaction Settings
DEFAULT_REDACTION_MODE = "blackout"
PIXELATION_BLOCK_SIZE = 8
BLUR_KERNEL_SIZE = 15

# File Handling Settings
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
SUPPORTED_DOCUMENT_FORMATS = ['pdf']
MAX_FILE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 4000

# Directory Settings
OUTPUT_DIR = "output"
LOGS_DIR = "logs"
TEMP_DIR = "temp"

# Logging Settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Privacy Settings
ENABLE_AUDIT_LOGGING = True
AUTO_CLEANUP_TEMP_FILES = True
DOCUMENT_RETENTION_HOURS = 0  # Never retain documents

# Performance Settings
ENABLE_GPU_ACCELERATION = False  # Set to True if CUDA available
MAX_CONCURRENT_PROCESSES = 2
IMAGE_RESIZE_THRESHOLD = 2048

# Indian Document Specific Settings
AADHAAR_PATTERN = r'\b\d{4}\s*\d{4}\s*\d{4}\b'
PAN_PATTERN = r'\b[A-Z]{5}\d{4}[A-Z]{1}\b'
PASSPORT_PATTERN = r'\b[A-Z]\d{7}\b'
VOTER_ID_PATTERN = r'\b[A-Z]{3}\d{7}\b'

# Common Indian Address Keywords
INDIAN_CITIES = [
    'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata',
    'pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
    'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna'
]

# Language Support (Future Enhancement)
SUPPORTED_LANGUAGES = ['eng']  # English only for now
PLANNED_LANGUAGES = ['hin', 'tam', 'kan', 'tel']  # Hindi, Tamil, Kannada, Telugu