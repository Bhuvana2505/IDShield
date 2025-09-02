


import cv2
import pytesseract
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import re
import io
import os

class TextExtractor:
    def __init__(self):
        """Initialize OCR engine with optimized settings for Indian documents"""
        # Set Tesseract executable path if not in PATH
        if os.name == 'nt':  # Windows
            # Try to find Tesseract in common installation locations
            tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                # Add more common installation paths
                r'C:\Program Files\Tesseract-OCR\bin\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\bin\tesseract.exe',
                # Check Windows PATH for tesseract.exe
                *[os.path.join(path_dir, 'tesseract.exe') 
                  for path_dir in os.environ.get('PATH', '').split(os.pathsep) 
                  if os.path.exists(os.path.join(path_dir, 'tesseract.exe'))],
                # Add the path from user's environment
                os.environ.get('TESSERACT_PATH', '')
            ]
            
            for path in tesseract_paths:
                if path and os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"Found Tesseract at: {path}")
                    break
        
        # Tesseract configuration optimized for Indian government documents
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/-: @'
        
        # Alternative configs for different document types
        self.configs = {
            'default': r'--oem 3 --psm 6',
            'single_line': r'--oem 3 --psm 7',
            'single_word': r'--oem 3 --psm 8',
            'numbers_only': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',
            'alphanumeric': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        }
        
    def preprocess_image(self, image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: Input image as numpy array
            enhance_contrast: Whether to apply contrast enhancement
            
        Returns:
            Preprocessed image optimized for OCR
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply contrast enhancement if requested
        if enhance_contrast:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Apply adaptive thresholding for better text clarity
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up text
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel_denoise = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_denoise)
        
        return denoised
    
    def extract_text(self, image: np.ndarray, config_type: str = 'default') -> Tuple[str, List[Dict]]:
        """
        Extract text from image and return text with bounding box coordinates
        
        Args:
            image: Input image as numpy array
            config_type: Type of OCR configuration to use
            
        Returns:
            tuple: (extracted_text, list_of_text_boxes_with_coordinates)
        """
        try:
            # Preprocess image for better OCR
            processed_image = self.preprocess_image(image)
            
            # Get OCR configuration
            config = self.configs.get(config_type, self.tesseract_config)
            
            # Extract text with detailed data including bounding boxes
            data = pytesseract.image_to_data(
                processed_image, 
                config=config, 
                output_type=pytesseract.Output.DICT
            )
            
            # Extract full text
            extracted_text = pytesseract.image_to_string(processed_image, config=config)
            
            # Process bounding boxes and filter valid detections
            text_boxes = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                # Filter out low confidence and empty detections
                if confidence > 30 and text:
                    box = {
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': confidence,
                        'word_num': data['word_num'][i],
                        'line_num': data['line_num'][i],
                        'par_num': data['par_num'][i]
                    }
                    text_boxes.append(box)
            
            return extracted_text, text_boxes
            
        except Exception as e:
            print(f"OCR extraction error: {str(e)}")
            return "", []
    
    def extract_text_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Extract text from specific regions of the image
        
        Args:
            image: Input image
            regions: List of (x, y, width, height) tuples
            
        Returns:
            List of extracted text strings from each region
        """
        extracted_texts = []
        
        for region in regions:
            x, y, w, h = region
            
            # Extract region of interest
            roi = image[y:y+h, x:x+w]
            
            if roi.size > 0:
                # Preprocess the ROI
                processed_roi = self.preprocess_image(roi)
                
                # Extract text from ROI
                text = pytesseract.image_to_string(processed_roi, config=self.tesseract_config)
                extracted_texts.append(text.strip())
            else:
                extracted_texts.append("")
        
        return extracted_texts
    
    def detect_qr_codes(self, image: np.ndarray) -> List[Dict]:
        """
        Detect QR codes and barcodes in the image
        
        Args:
            image: Input image
            
        Returns:
            List of detected QR codes with bounding boxes
        """
        qr_detector = cv2.QRCodeDetector()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        qr_codes = []
        
        try:
            # Detect and decode QR codes
            data, bbox, straight_qrcode = qr_detector.detectAndDecode(gray)
            
            if bbox is not None:
                bbox = bbox.astype(int)
                
                # Handle multiple QR codes
                if data:  # Single QR code
                    qr_data = [data]
                    qr_boxes = [bbox]
                else:  # Multiple QR codes
                    qr_data, qr_boxes, _ = qr_detector.detectAndDecodeMulti(gray)
                
                for i, box in enumerate(qr_boxes):
                    # Calculate bounding rectangle
                    x_coords = box[:, 0]
                    y_coords = box[:, 1]
                    
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    
                    qr_codes.append({
                        'type': 'qr_code',
                        'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                        'data': qr_data[i] if i < len(qr_data) else "",
                        'confidence': 0.95,
                        'detection_method': 'opencv_qr_detector'
                    })
            
            # Also try to detect using contour analysis for damaged QR codes
            contour_qr_codes = self._detect_qr_by_contours(gray)
            qr_codes.extend(contour_qr_codes)
            
            return qr_codes
            
        except Exception as e:
            print(f"QR code detection error: {str(e)}")
            return []
    
    def _detect_qr_by_contours(self, gray_image: np.ndarray) -> List[Dict]:
        """
        Detect QR code-like patterns using contour analysis
        Useful for damaged or partially visible QR codes
        """
        qr_codes = []
        
        try:
            # Apply edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                
                if area > 1000:  # Minimum area for QR code
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # QR codes are typically square-ish
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.7 <= aspect_ratio <= 1.3 and area > 1000:  # Square-like shape
                        qr_codes.append({
                            'type': 'qr_code_pattern',
                            'bbox': [x, y, w, h],
                            'data': "",
                            'confidence': 0.6,
                            'detection_method': 'contour_analysis'
                        })
            
            return qr_codes
            
        except Exception as e:
            print(f"Contour QR detection error: {str(e)}")
            return []
    
    def extract_text_with_confidence(self, image: np.ndarray, min_confidence: int = 60) -> Tuple[str, List[Dict]]:
        """
        Extract text with confidence filtering for higher accuracy
        
        Args:
            image: Input image
            min_confidence: Minimum confidence threshold for text detection
            
        Returns:
            Tuple of (filtered_text, high_confidence_text_boxes)
        """
        try:
            processed_image = self.preprocess_image(image)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                processed_image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter high-confidence text
            high_confidence_text = []
            text_boxes = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence >= min_confidence and text:
                    high_confidence_text.append(text)
                    
                    box = {
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': confidence,
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i]
                    }
                    text_boxes.append(box)
            
            # Join high-confidence text
            filtered_text = ' '.join(high_confidence_text)
            
            return filtered_text, text_boxes
            
        except Exception as e:
            print(f"High-confidence extraction error: {str(e)}")
            return "", []
    
    def detect_document_orientation(self, image: np.ndarray) -> float:
        """
        Detect if document needs rotation for better OCR
        
        Returns:
            Rotation angle in degrees (0, 90, 180, 270)
        """
        try:
            # Use Tesseract's built-in orientation detection
            osd = pytesseract.image_to_osd(image, config='--psm 0')
            
            # Parse orientation information
            angle = 0
            for line in osd.split('\n'):
                if 'Rotate:' in line:
                    angle = int(line.split(':')[1].strip())
                    break
            
            return angle
            
        except Exception as e:
            print(f"Orientation detection error: {str(e)}")
            return 0
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        if angle == 0:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos_angle = np.abs(rotation_matrix[0, 0])
        sin_angle = np.abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Apply various image enhancement techniques for better OCR
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply sharpening kernel
        sharpening_kernel = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
        sharpened = cv2.filter2D(filtered, -1, sharpening_kernel)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_lines(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text line by line with coordinates
        
        Returns:
            List of dictionaries containing line text and bounding boxes
        """
        try:
            processed_image = self.preprocess_image(image)
            
            # Use line-level PSM
            data = pytesseract.image_to_data(
                processed_image,
                config='--oem 3 --psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            # Group words by line number
            lines = {}
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Filter low confidence
                    line_num = data['line_num'][i]
                    text = data['text'][i].strip()
                    
                    if text:
                        if line_num not in lines:
                            lines[line_num] = {
                                'text': [],
                                'boxes': [],
                                'x_coords': [],
                                'y_coords': []
                            }
                        
                        lines[line_num]['text'].append(text)
                        lines[line_num]['boxes'].append({
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        })
                        lines[line_num]['x_coords'].append(data['left'][i])
                        lines[line_num]['y_coords'].append(data['top'][i])
            
            # Create line-level results
            line_results = []
            for line_num, line_data in lines.items():
                if line_data['text']:
                    # Combine text in line
                    line_text = ' '.join(line_data['text'])
                    
                    # Calculate line bounding box
                    x_min = min(line_data['x_coords'])
                    y_min = min(line_data['y_coords'])
                    x_max = max([box['x'] + box['width'] for box in line_data['boxes']])
                    y_max = max([box['y'] + box['height'] for box in line_data['boxes']])
                    
                    line_results.append({
                        'line_number': line_num,
                        'text': line_text,
                        'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                        'word_count': len(line_data['text'])
                    })
            
            return line_results
            
        except Exception as e:
            print(f"Line extraction error: {str(e)}")
            return []
    
    def extract_numbers_only(self, image: np.ndarray) -> Tuple[str, List[Dict]]:
        """
        Extract only numeric content (useful for ID numbers)
        
        Returns:
            Tuple of (numeric_text, number_boxes)
        """
        try:
            processed_image = self.preprocess_image(image)
            
            # Use numbers-only configuration
            config = self.configs['numbers_only']
            
            # Extract numeric data
            data = pytesseract.image_to_data(
                processed_image,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            numeric_text = pytesseract.image_to_string(processed_image, config=config)
            
            # Filter numeric detections
            number_boxes = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                if confidence > 50 and text and text.isdigit():
                    box = {
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': confidence,
                        'type': 'numeric'
                    }
                    number_boxes.append(box)
            
            return numeric_text, number_boxes
            
        except Exception as e:
            print(f"Numeric extraction error: {str(e)}")
            return "", []
    
    def get_text_statistics(self, text: str) -> Dict:
        """
        Analyze extracted text and provide statistics
        
        Returns:
            Dictionary with text analysis statistics
        """
        if not text:
            return {}
        
        lines = text.split('\n')
        words = text.split()
        
        # Count different character types
        digit_count = sum(1 for char in text if char.isdigit())
        alpha_count = sum(1 for char in text if char.isalpha())
        special_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
        
        return {
            'total_characters': len(text),
            'total_words': len(words),
            'total_lines': len([line for line in lines if line.strip()]),
            'digit_count': digit_count,
            'alpha_count': alpha_count,
            'special_char_count': special_count,
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'contains_indian_numbers': bool(re.search(r'\b\d{4}\s*\d{4}\s*\d{4}\b', text)),  # Aadhaar pattern
            'contains_alphanumeric_id': bool(re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text))  # PAN pattern
        }