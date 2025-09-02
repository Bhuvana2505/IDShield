import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import tempfile
import os
import io
from typing import List, Optional, Dict

class FileHandler:
    def __init__(self):
        """Initialize file handler for various document formats"""
        self.supported_formats = ['jpg', 'jpeg', 'png', 'pdf']
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file format is supported"""
        if not os.path.exists(file_path):
            return False
        
        file_extension = file_path.split('.')[-1].lower()
        return file_extension in self.supported_formats
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        """
        Convert PDF pages to images using PyMuPDF
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of images as numpy arrays
        """
        images = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Create transformation matrix for desired DPI
                mat = fitz.Matrix(dpi/72, dpi/72)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Convert to OpenCV format (BGR)
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                images.append(opencv_image)
            
            doc.close()
            return images
            
        except Exception as e:
            print(f"PDF conversion error: {str(e)}")
            return []
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image file and return as OpenCV array"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL for better format support
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            print(f"Image loading error: {str(e)}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: str, quality: int = 95) -> bool:
        """Save OpenCV image to file"""
        try:
            # Determine file format
            file_extension = output_path.split('.')[-1].lower()
            
            if file_extension in ['jpg', 'jpeg']:
                # Save with JPEG compression
                cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif file_extension == 'png':
                # Save as PNG
                cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                # Default save
                cv2.imwrite(output_path, image)
            
            return True
            
        except Exception as e:
            print(f"Image saving error: {str(e)}")
            return False
    
    def get_image_info(self, image: np.ndarray) -> Dict:
        """Get basic information about the image"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'size_mb': (width * height * channels) / (1024 * 1024)
        }
    
    def resize_image(self, image: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_image
        
        return image
    
    def create_temp_file(self, data: bytes, suffix: str = '.tmp') -> str:
        """Create temporary file and return path"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(data)
            return tmp_file.name
    
    def cleanup_temp_file(self, file_path: str) -> bool:
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                return True
            return False
        except Exception as e:
            print(f"Cleanup error: {str(e)}")
            return False