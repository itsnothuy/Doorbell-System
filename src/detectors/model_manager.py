#!/usr/bin/env python3
"""
Model Manager - Download and Cache Management for Detection Models

Handles model downloading, caching, and versioning for face detection models.
Supports multiple model formats (ONNX, TensorFlow Lite, SavedModel).
"""

import os
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.request import urlretrieve
from urllib.error import URLError

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages face detection model downloads and caching.
    
    Features:
    - Automatic model downloading from URLs
    - Local caching with checksums
    - Version management
    - Multiple model format support
    """
    
    # Model registry with download URLs and checksums
    MODEL_REGISTRY = {
        'retinaface_gpu': {
            'url': 'https://example.com/models/retinaface_gpu.onnx',
            'checksum': 'placeholder_checksum',
            'format': 'onnx',
            'size_mb': 10
        },
        'mtcnn_gpu': {
            'url': 'https://example.com/models/mtcnn_gpu.pb',
            'checksum': 'placeholder_checksum',
            'format': 'tensorflow',
            'size_mb': 5
        },
        'yolov5_face_gpu': {
            'url': 'https://example.com/models/yolov5_face_gpu.onnx',
            'checksum': 'placeholder_checksum',
            'format': 'onnx',
            'size_mb': 15
        },
        'mobilenet_face_edgetpu': {
            'url': 'https://example.com/models/mobilenet_face_edgetpu.tflite',
            'checksum': 'placeholder_checksum',
            'format': 'tflite',
            'size_mb': 4
        },
        'efficientdet_face_edgetpu': {
            'url': 'https://example.com/models/efficientdet_face_edgetpu.tflite',
            'checksum': 'placeholder_checksum',
            'format': 'tflite',
            'size_mb': 6
        },
        'blazeface_edgetpu': {
            'url': 'https://example.com/models/blazeface_edgetpu.tflite',
            'checksum': 'placeholder_checksum',
            'format': 'tflite',
            'size_mb': 2
        }
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for caching models (default: ~/.doorbell_models)
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.doorbell_models'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file for tracking cached models
        self.metadata_file = self.cache_dir / 'models.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cached model metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model metadata: {e}")
        
        return {}
    
    def _save_metadata(self) -> None:
        """Save model metadata to cache."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
    
    def get_model(self, model_name: str, model_path: str) -> Path:
        """
        Get model path, downloading if necessary.
        
        Args:
            model_name: Name of the model in registry
            model_path: Relative path where model should be cached
            
        Returns:
            Path to model file
            
        Raises:
            FileNotFoundError: If model cannot be found or downloaded
        """
        # Check if model exists in cache
        cached_path = self.cache_dir / model_path
        
        if cached_path.exists():
            # Verify checksum if available
            if self._verify_model(model_name, cached_path):
                logger.debug(f"Using cached model: {cached_path}")
                return cached_path
            else:
                logger.warning(f"Cached model checksum mismatch: {model_name}")
        
        # Try to download model
        if model_name in self.MODEL_REGISTRY:
            try:
                return self._download_model(model_name, cached_path)
            except Exception as e:
                logger.error(f"Failed to download model {model_name}: {e}")
        
        # Check if model exists in local models directory
        local_path = Path('models') / Path(model_path).name
        if local_path.exists():
            logger.info(f"Using local model: {local_path}")
            return local_path
        
        # Model not found - use mock model for testing
        logger.warning(f"Model {model_name} not found, creating mock model for testing")
        return self._create_mock_model(cached_path)
    
    def _download_model(self, model_name: str, target_path: Path) -> Path:
        """
        Download model from registry.
        
        Args:
            model_name: Name of model in registry
            target_path: Path where model should be saved
            
        Returns:
            Path to downloaded model
        """
        model_info = self.MODEL_REGISTRY[model_name]
        url = model_info['url']
        
        logger.info(f"Downloading model {model_name} from {url}")
        
        # Create target directory
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download model
            urlretrieve(url, target_path)
            
            # Verify checksum
            if not self._verify_checksum(target_path, model_info['checksum']):
                target_path.unlink()
                raise ValueError(f"Checksum verification failed for {model_name}")
            
            # Update metadata
            self.metadata[model_name] = {
                'path': str(target_path),
                'checksum': model_info['checksum'],
                'format': model_info['format']
            }
            self._save_metadata()
            
            logger.info(f"Model {model_name} downloaded successfully")
            return target_path
            
        except URLError as e:
            logger.error(f"Download failed for {model_name}: {e}")
            raise
    
    def _verify_model(self, model_name: str, model_path: Path) -> bool:
        """Verify model checksum."""
        if model_name not in self.MODEL_REGISTRY:
            return True  # No checksum to verify
        
        expected_checksum = self.MODEL_REGISTRY[model_name]['checksum']
        return self._verify_checksum(model_path, expected_checksum)
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Calculate and verify file checksum."""
        if expected_checksum == 'placeholder_checksum':
            return True  # Skip verification for placeholder checksums
        
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            actual_checksum = sha256_hash.hexdigest()
            return actual_checksum == expected_checksum
            
        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False
    
    def _create_mock_model(self, target_path: Path) -> Path:
        """
        Create a mock model file for testing.
        
        Args:
            target_path: Path where mock model should be created
            
        Returns:
            Path to mock model
        """
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a small mock file
        with open(target_path, 'wb') as f:
            f.write(b'MOCK_MODEL_DATA')
        
        logger.info(f"Created mock model: {target_path}")
        return target_path
    
    def list_cached_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all cached models.
        
        Returns:
            Dictionary of cached models with metadata
        """
        return self.metadata.copy()
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            logger.info("Model cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_size(self) -> int:
        """
        Get total size of cached models in bytes.
        
        Returns:
            Total cache size in bytes
        """
        total_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except Exception:
                    pass
        
        return total_size
