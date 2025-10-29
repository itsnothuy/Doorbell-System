#!/usr/bin/env python3
"""
Legacy Mapping - Configuration Mapping from Legacy to Pipeline

Defines mapping rules for converting legacy configuration to pipeline format.
"""

from typing import Dict, Any


class LegacyMapping:
    """Mapping rules for legacy to pipeline configuration conversion."""
    
    # Configuration field mappings
    FIELD_MAPPINGS = {
        # Legacy field -> Pipeline field
        'DEBOUNCE_TIME': 'frame_capture.debounce_time',
        'CAPTURES_DIR': 'storage.capture_path',
        'KNOWN_FACES_DIR': 'face_recognition.known_faces_path',
        'BLACKLIST_FACES_DIR': 'face_recognition.blacklist_faces_path',
        'LOGS_DIR': 'storage.log_path',
    }
    
    # Default values for pipeline configuration
    PIPELINE_DEFAULTS = {
        'frame_capture': {
            'enabled': True,
            'debounce_time': 5.0,
            'max_queue_size': 10
        },
        'face_detection': {
            'enabled': True,
            'detector_type': 'cpu',
            'confidence_threshold': 0.5
        },
        'face_recognition': {
            'enabled': True,
            'recognition_threshold': 0.6,
            'known_faces_path': 'data/known_faces',
            'blacklist_faces_path': 'data/blacklist_faces'
        },
        'storage': {
            'capture_path': 'data/captures',
            'log_path': 'data/logs',
            'database_path': 'data/events.db'
        }
    }
    
    @staticmethod
    def map_legacy_to_pipeline(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map legacy configuration to pipeline format.
        
        Args:
            legacy_config: Legacy configuration dictionary
            
        Returns:
            Pipeline configuration dictionary
        """
        pipeline_config = LegacyMapping.PIPELINE_DEFAULTS.copy()
        
        # Apply field mappings
        for legacy_field, pipeline_field in LegacyMapping.FIELD_MAPPINGS.items():
            if legacy_field in legacy_config:
                # Parse nested field path
                parts = pipeline_field.split('.')
                current = pipeline_config
                
                # Navigate to parent dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set value
                current[parts[-1]] = legacy_config[legacy_field]
        
        return pipeline_config
    
    @staticmethod
    def validate_pipeline_config(pipeline_config: Dict[str, Any]) -> bool:
        """
        Validate pipeline configuration.
        
        Args:
            pipeline_config: Pipeline configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_sections = ['frame_capture', 'face_detection', 'face_recognition', 'storage']
        
        for section in required_sections:
            if section not in pipeline_config:
                return False
        
        return True
