import re
import cv2
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

class EntityDetector:
    def __init__(self):
        """Initialize entity detection patterns for Indian documents"""
        self.patterns = {
            'aadhaar': r'\b\d{4}\s*\d{4}\s*\d{4}\b',
            'pan': r'\b[A-Z]{5}\d{4}[A-Z]{1}\b',
            'passport': r'\b[A-Z]\d{7}\b',
            'voter_id': r'\b[A-Z]{3}\d{7}\b',
            'phone': r'\b(?:\+91[-.\s]?)?[6-9]\d{9}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'pincode': r'\b\d{6}\b'
        }
        
        # Common Indian names patterns (partial list for demo)
        self.name_keywords = [
            'father', 'son', 'daughter', 'wife', 'husband', 'mother',
            'shri', 'smt', 'dr', 'mr', 'mrs', 'ms'
        ]
        
        # Address keywords
        self.address_keywords = [
            'address', 'house', 'street', 'road', 'lane', 'colony',
            'nagar', 'pune', 'mumbai', 'delhi', 'bangalore', 'chennai',
            'hyderabad', 'kolkata', 'ahmedabad', 'surat', 'jaipur'
        ]
    
    def detect_entities(self, text: str, text_boxes: List[Dict], confidence_threshold: float = 0.8) -> List[Dict]:
        """
        Detect sensitive entities in extracted text
        
        Args:
            text: Full extracted text
            text_boxes: List of text boxes with coordinates
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected entities with coordinates and types
        """
        detected_entities = []
        
        # Detect ID numbers using regex patterns
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Find corresponding text box
                match_text = match.group()
                bbox = self._find_text_bbox(match_text, text_boxes)
                
                if bbox:
                    entity = {
                        'type': entity_type,
                        'text': match_text,
                        'bbox': bbox,
                        'confidence': 0.9,  # High confidence for regex matches
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    }
                    detected_entities.append(entity)
        
        # Detect names (heuristic approach)
        name_entities = self._detect_names(text, text_boxes)
        detected_entities.extend(name_entities)
        
        # Detect addresses
        address_entities = self._detect_addresses(text, text_boxes)
        detected_entities.extend(address_entities)
        
        # Detect dates
        date_entities = self._detect_dates(text, text_boxes)
        detected_entities.extend(date_entities)
        
        # Filter by confidence threshold
        filtered_entities = [
            entity for entity in detected_entities 
            if entity['confidence'] >= confidence_threshold
        ]
        
        return filtered_entities
    
    def _find_text_bbox(self, target_text: str, text_boxes: List[Dict]) -> List[int]:
        """Find bounding box coordinates for specific text"""
        target_clean = re.sub(r'\s+', '', target_text.lower())
        
        for box in text_boxes:
            box_text_clean = re.sub(r'\s+', '', box['text'].lower())
            
            if target_clean in box_text_clean or box_text_clean in target_clean:
                return [box['x'], box['y'], box['width'], box['height']]
        
        return None
    
    def _detect_names(self, text: str, text_boxes: List[Dict]) -> List[Dict]:
        """Detect potential names using heuristic patterns"""
        names = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for name patterns
            if any(keyword in line.lower() for keyword in self.name_keywords):
                # Extract potential name after keywords
                for keyword in self.name_keywords:
                    if keyword in line.lower():
                        parts = line.lower().split(keyword)
                        if len(parts) > 1:
                            potential_name = parts[1].strip()
                            # Clean and validate name
                            name_clean = re.sub(r'[^a-zA-Z\s]', '', potential_name)
                            if len(name_clean.split()) >= 2:  # At least 2 words
                                bbox = self._find_text_bbox(name_clean, text_boxes)
                                if bbox:
                                    names.append({
                                        'type': 'name',
                                        'text': name_clean,
                                        'bbox': bbox,
                                        'confidence': 0.7
                                    })
        
        return names
    
    def _detect_addresses(self, text: str, text_boxes: List[Dict]) -> List[Dict]:
        """Detect address information"""
        addresses = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line contains address keywords
            if any(keyword in line.lower() for keyword in self.address_keywords):
                # Consider this line and next 2-3 lines as potential address
                address_lines = []
                for j in range(i, min(i + 3, len(lines))):
                    if lines[j].strip():
                        address_lines.append(lines[j].strip())
                
                if address_lines:
                    full_address = ' '.join(address_lines)
                    bbox = self._find_text_bbox(line, text_boxes)
                    
                    if bbox:
                        addresses.append({
                            'type': 'address',
                            'text': full_address,
                            'bbox': bbox,
                            'confidence': 0.75
                        })
        
        return addresses
    
    def _detect_dates(self, text: str, text_boxes: List[Dict]) -> List[Dict]:
        """Detect date patterns (DOB, issue dates, etc.)"""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'  # DD Month YYYY
        ]
        
        dates = []
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                date_text = match.group()
                bbox = self._find_text_bbox(date_text, text_boxes)
                
                if bbox:
                    dates.append({
                        'type': 'date',
                        'text': date_text,
                        'bbox': bbox,
                        'confidence': 0.85
                    })
        
        return dates
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in the document image"""
        try:
            # Load OpenCV face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_entities = []
            for (x, y, w, h) in faces:
                face_entities.append({
                    'type': 'face',
                    'text': 'FACE_DETECTED',
                    'bbox': [x, y, w, h],
                    'confidence': 0.8
                })
            
            return face_entities
            
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            return []
    
    def detect_signatures(self, image: np.ndarray) -> List[Dict]:
        """Detect potential signature regions using image processing"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            signatures = []
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Signature heuristics: horizontal rectangular regions with moderate area
                    if 1.5 < aspect_ratio < 5 and 500 < area < 10000:
                        signatures.append({
                            'type': 'signature',
                            'text': 'SIGNATURE_DETECTED',
                            'bbox': [x, y, w, h],
                            'confidence': 0.6
                        })
            
            return signatures
            
        except Exception as e:
            print(f"Signature detection error: {str(e)}")
            return []