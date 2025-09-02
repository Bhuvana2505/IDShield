import cv2
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

class DocumentRedactor:
    def __init__(self):
        """Initialize document redaction engine"""
        self.redaction_methods = {
            'blackout': self._apply_blackout,
            'pixelation': self._apply_pixelation,
            'blur': self._apply_blur
        }
    
    def redact_document(self, image: np.ndarray, entities: List[Dict], redaction_mode: str = 'blackout') -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply redaction to detected entities
        
        Args:
            image: Original document image
            entities: List of detected entities with bounding boxes
            redaction_mode: Type of redaction to apply
            
        Returns:
            tuple: (redacted_image, redaction_log)
        """
        redacted_image = image.copy()
        redaction_log = []
        
        redaction_func = self.redaction_methods.get(redaction_mode.lower(), self._apply_blackout)
        
        for entity in entities:
            try:
                bbox = entity['bbox']
                x, y, w, h = bbox
                
                # Ensure coordinates are within image bounds
                height, width = image.shape[:2]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w > 0 and h > 0:
                    # Apply redaction
                    redacted_image = redaction_func(redacted_image, x, y, w, h)
                    
                    # Log the redaction
                    log_entry = {
                        'entity_type': entity['type'],
                        'redaction_method': redaction_mode,
                        'coordinates': [x, y, w, h],
                        'confidence': entity['confidence'],
                        'timestamp': datetime.now().isoformat()
                    }
                    redaction_log.append(log_entry)
                    
            except Exception as e:
                print(f"Redaction error for entity {entity}: {str(e)}")
                continue
        
        return redacted_image, redaction_log
    
    def _apply_blackout(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply black rectangle over sensitive area"""
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
        return image
    
    def _apply_pixelation(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply pixelation effect over sensitive area"""
        try:
            # Extract region
            region = image[y:y+h, x:x+w]
            
            if region.size > 0:
                # Downsample and upsample for pixelation effect
                pixel_size = max(8, min(w, h) // 10)  # Adaptive pixel size
                
                small = cv2.resize(region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Replace region with pixelated version
                image[y:y+h, x:x+w] = pixelated
                
        except Exception as e:
            # Fallback to blackout if pixelation fails
            print(f"Pixelation failed, using blackout: {str(e)}")
            self._apply_blackout(image, x, y, w, h)
        
        return image
    
    def _apply_blur(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply Gaussian blur over sensitive area"""
        try:
            # Extract region
            region = image[y:y+h, x:x+w]
            
            if region.size > 0:
                # Apply strong Gaussian blur
                kernel_size = max(15, min(w, h) // 5)  # Adaptive kernel size
                if kernel_size % 2 == 0:  # Ensure odd kernel size
                    kernel_size += 1
                
                blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
                
                # Replace region with blurred version
                image[y:y+h, x:x+w] = blurred
                
        except Exception as e:
            # Fallback to blackout if blur fails
            print(f"Blur failed, using blackout: {str(e)}")
            self._apply_blackout(image, x, y, w, h)
        
        return image
    
    def redact_qr_codes(self, image: np.ndarray, qr_codes: List[Dict], redaction_mode: str = 'blackout') -> np.ndarray:
        """Specifically redact QR codes and barcodes"""
        redacted_image = image.copy()
        
        redaction_func = self.redaction_methods.get(redaction_mode.lower(), self._apply_blackout)
        
        for qr_code in qr_codes:
            bbox = qr_code['bbox']
            x, y, w, h = bbox
            
            # Add some padding around QR code
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w + 2*padding, image.shape[1] - x)
            h = min(h + 2*padding, image.shape[0] - y)
            
            redacted_image = redaction_func(redacted_image, x, y, w, h)
        
        return redacted_image
    
    def create_redaction_preview(self, image: np.ndarray, entities: List[Dict]) -> np.ndarray:
        """Create a preview image showing detected entities with colored boxes"""
        preview_image = image.copy()
        
        colors = {
            'aadhaar': (0, 0, 255),      # Red
            'pan': (255, 0, 0),          # Blue
            'passport': (0, 255, 0),     # Green
            'phone': (255, 255, 0),      # Cyan
            'email': (255, 0, 255),      # Magenta
            'name': (0, 255, 255),       # Yellow
            'address': (128, 0, 128),    # Purple
            'date': (255, 165, 0),       # Orange
            'face': (0, 128, 255),       # Orange-Red
            'signature': (128, 128, 128), # Gray
            'qr_code': (64, 224, 208)    # Turquoise
        }
        
        for entity in entities:
            bbox = entity['bbox']
            x, y, w, h = bbox
            entity_type = entity['type']
            
            color = colors.get(entity_type, (255, 255, 255))  # Default white
            
            # Draw rectangle around detected entity
            cv2.rectangle(preview_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{entity_type}: {entity['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(preview_image, (x, y - 20), (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(preview_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return preview_image