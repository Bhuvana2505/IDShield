import json
import os
from datetime import datetime
from typing import Dict, List, Any
import hashlib

class RedactionLogger:
    def __init__(self):
        """Initialize redaction logger"""
        self.log_version = "1.0"
    
    def create_log(self, original_filename: str, detected_entities: List[Dict], 
                   redaction_log: List[Dict], redaction_mode: str) -> Dict[str, Any]:
        """
        Create comprehensive log of redaction process
        
        Args:
            original_filename: Name of original document
            detected_entities: List of all detected entities
            redaction_log: Log of applied redactions
            redaction_mode: Type of redaction applied
            
        Returns:
            Complete log dictionary
        """
        timestamp = datetime.now().isoformat()
        
        # Create document hash (filename-based for privacy)
        doc_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
        
        log_data = {
            "log_metadata": {
                "version": self.log_version,
                "timestamp": timestamp,
                "document_hash": doc_hash,
                "original_filename": original_filename,
                "redaction_mode": redaction_mode
            },
            "detection_summary": {
                "total_entities_detected": len(detected_entities),
                "entity_types_found": list(set([entity['type'] for entity in detected_entities])),
                "average_confidence": self._calculate_average_confidence(detected_entities),
                "entities_redacted": len(redaction_log)
            },
            "detected_entities": [
                {
                    "entity_id": f"entity_{i+1}",
                    "type": entity['type'],
                    "confidence": entity['confidence'],
                    "coordinates": {
                        "x": entity['bbox'][0],
                        "y": entity['bbox'][1],
                        "width": entity['bbox'][2],
                        "height": entity['bbox'][3]
                    },
                    "text_length": len(entity.get('text', '')) if entity.get('text') else 0
                }
                for i, entity in enumerate(detected_entities)
            ],
            "redaction_log": redaction_log,
            "privacy_compliance": {
                "data_retention": "No document data stored permanently",
                "processing_location": "Local processing",
                "audit_trail": "Complete redaction coordinates logged",
                "compliance_standards": ["DPDP Act", "GDPR Ready"]
            },
            "performance_metrics": {
                "processing_timestamp": timestamp,
                "entities_processed": len(detected_entities),
                "successful_redactions": len(redaction_log)
            }
        }
        
        return log_data
    
    def _calculate_average_confidence(self, entities: List[Dict]) -> float:
        """Calculate average confidence score of detected entities"""
        if not entities:
            return 0.0
        
        total_confidence = sum(entity['confidence'] for entity in entities)
        return round(total_confidence / len(entities), 3)
    
    def save_log_file(self, log_data: Dict[str, Any], output_path: str) -> bool:
        """Save log data to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            return True
            
        except Exception as e:
            print(f"Log saving error: {str(e)}")
            return False
    
    def load_log_file(self, log_path: str) -> Dict[str, Any]:
        """Load existing log file"""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Log loading error: {str(e)}")
            return {}
    
    def generate_audit_report(self, log_data: Dict[str, Any]) -> str:
        """Generate human-readable audit report"""
        try:
            metadata = log_data.get('log_metadata', {})
            summary = log_data.get('detection_summary', {})
            
            report = f"""
IDSHIELD REDACTION AUDIT REPORT
================================

Document Information:
- Document Hash: {metadata.get('document_hash', 'N/A')}
- Processing Date: {metadata.get('timestamp', 'N/A')}
- Redaction Mode: {metadata.get('redaction_mode', 'N/A')}

Detection Summary:
- Total Entities Detected: {summary.get('total_entities_detected', 0)}
- Entity Types Found: {', '.join(summary.get('entity_types_found', []))}
- Average Confidence: {summary.get('average_confidence', 0):.3f}
- Entities Successfully Redacted: {summary.get('entities_redacted', 0)}

Privacy Compliance:
- Data Retention: {log_data.get('privacy_compliance', {}).get('data_retention', 'N/A')}
- Processing Location: {log_data.get('privacy_compliance', {}).get('processing_location', 'N/A')}
- Audit Trail: {log_data.get('privacy_compliance', {}).get('audit_trail', 'N/A')}

Detailed Entity Log:
"""
            
            entities = log_data.get('detected_entities', [])
            for entity in entities:
                report += f"""
- {entity.get('entity_id', 'N/A')}: {entity.get('type', 'N/A')} 
  Confidence: {entity.get('confidence', 0):.3f}
  Location: ({entity.get('coordinates', {}).get('x', 0)}, {entity.get('coordinates', {}).get('y', 0)})
  Size: {entity.get('coordinates', {}).get('width', 0)}x{entity.get('coordinates', {}).get('height', 0)}
"""
            
            report += f"""
================================
Report Generated: {datetime.now().isoformat()}
IDShield v{self.log_version}
"""
            
            return report
            
        except Exception as e:
            return f"Error generating audit report: {str(e)}"