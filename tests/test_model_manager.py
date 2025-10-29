#!/usr/bin/env python3
"""
Unit Tests for Secure Model Manager

Tests cryptographic integrity verification, secure downloads, and security controls.
"""

import os
import hashlib
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from urllib.error import URLError, HTTPError

from src.detectors.model_manager import ModelManager, ModelInfo


class TestModelInfo:
    """Test ModelInfo dataclass validation."""
    
    def test_valid_model_info(self):
        """Test creating valid ModelInfo."""
        model = ModelInfo(
            name='test_model',
            url='https://example.com/model.dat',
            sha256_hash='a' * 64,  # Valid 64-character hex
            size=1024,
            format='onnx',
            version='1.0.0'
        )
        
        assert model.name == 'test_model'
        assert model.sha256_hash == 'a' * 64
        assert model.algorithm == 'SHA-256'
    
    def test_invalid_hash_length(self):
        """Test that invalid hash length raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SHA-256 hash length"):
            ModelInfo(
                name='test_model',
                url='https://example.com/model.dat',
                sha256_hash='abc123',  # Too short
                size=1024,
                format='onnx'
            )
    
    def test_invalid_hash_format(self):
        """Test that non-hexadecimal hash raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SHA-256 hash format"):
            ModelInfo(
                name='test_model',
                url='https://example.com/model.dat',
                sha256_hash='z' * 64,  # Invalid hex characters
                size=1024,
                format='onnx'
            )
    
    def test_invalid_size(self):
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            ModelInfo(
                name='test_model',
                url='https://example.com/model.dat',
                sha256_hash='a' * 64,
                size=0,  # Invalid size
                format='onnx'
            )


class TestModelManager:
    """Test ModelManager security functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Create ModelManager instance with temp cache."""
        return ModelManager(cache_dir=temp_cache_dir)
    
    @pytest.fixture
    def test_file(self, temp_cache_dir):
        """Create a test file with known content and hash."""
        test_path = temp_cache_dir / "test_file.dat"
        test_content = b"This is test content for hash verification"
        
        with open(test_path, 'wb') as f:
            f.write(test_content)
        
        # Calculate expected hash
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        return test_path, expected_hash, len(test_content)
    
    def test_initialization(self, manager, temp_cache_dir):
        """Test ModelManager initialization."""
        assert manager.cache_dir == temp_cache_dir
        assert manager.cache_dir.exists()
        assert len(manager.MODEL_REGISTRY) == 6  # 6 models defined
        assert manager.verification_key is None
    
    def test_initialization_with_hmac_key(self, temp_cache_dir):
        """Test initialization with HMAC verification key."""
        manager = ModelManager(
            cache_dir=temp_cache_dir,
            verification_key="test_secret_key"
        )
        
        assert manager.verification_key is not None
        assert isinstance(manager.verification_key, bytes)
    
    def test_calculate_file_hash_sha256(self, manager, test_file):
        """Test SHA-256 hash calculation accuracy."""
        test_path, expected_hash, _ = test_file
        
        actual_hash = manager.calculate_file_hash(test_path, "SHA-256")
        
        assert actual_hash == expected_hash
        assert len(actual_hash) == 64
    
    def test_calculate_file_hash_sha512(self, manager, test_file):
        """Test SHA-512 hash calculation."""
        test_path, _, _ = test_file
        
        actual_hash = manager.calculate_file_hash(test_path, "SHA-512")
        
        assert len(actual_hash) == 128  # SHA-512 produces 128 hex characters
    
    def test_calculate_file_hash_unsupported_algorithm(self, manager, test_file):
        """Test that unsupported algorithm raises ValueError."""
        test_path, _, _ = test_file
        
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            manager.calculate_file_hash(test_path, "INVALID_ALGO")
    
    def test_calculate_file_hash_nonexistent_file(self, manager, temp_cache_dir):
        """Test hash calculation on nonexistent file raises IOError."""
        nonexistent = temp_cache_dir / "nonexistent.dat"
        
        with pytest.raises(IOError, match="Failed to calculate hash"):
            manager.calculate_file_hash(nonexistent)
    
    def test_verify_model_integrity_unknown_model(self, manager, test_file):
        """Test verification of unknown model returns False."""
        test_path, _, _ = test_file
        
        is_valid, error = manager.verify_model_integrity("unknown_model", test_path)
        
        assert not is_valid
        assert "Unknown model" in error
    
    def test_verify_model_integrity_missing_file(self, manager, temp_cache_dir):
        """Test verification of missing file returns False."""
        missing_path = temp_cache_dir / "missing.dat"
        
        is_valid, error = manager.verify_model_integrity("retinaface_gpu", missing_path)
        
        assert not is_valid
        assert "not found" in error
    
    def test_verify_model_integrity_size_mismatch(self, manager, temp_cache_dir):
        """Test detection of file size mismatch."""
        # Create file with wrong size
        test_path = temp_cache_dir / "wrong_size.dat"
        with open(test_path, 'wb') as f:
            f.write(b"wrong size content")
        
        is_valid, error = manager.verify_model_integrity("retinaface_gpu", test_path)
        
        assert not is_valid
        assert "Size mismatch" in error
    
    def test_verify_model_integrity_hash_mismatch(self, manager, temp_cache_dir):
        """Test detection of hash mismatch (tampered file)."""
        # Get expected size for retinaface_gpu model
        model_info = manager.MODEL_REGISTRY["retinaface_gpu"]
        
        # Create file with correct size but wrong content
        test_path = temp_cache_dir / "tampered.dat"
        with open(test_path, 'wb') as f:
            f.write(b"X" * model_info.size)
        
        is_valid, error = manager.verify_model_integrity("retinaface_gpu", test_path)
        
        assert not is_valid
        assert "Hash mismatch" in error or "integrity check failed" in error
    
    def test_list_available_models(self, manager):
        """Test listing all available models."""
        models = manager.list_available_models()
        
        assert len(models) == 6
        assert "retinaface_gpu" in models
        assert "mtcnn_gpu" in models
        
        # Check structure
        for model_name, status in models.items():
            assert 'name' in status
            assert 'version' in status
            assert 'size' in status
            assert 'sha256_hash' in status
            assert 'exists' in status
            assert 'valid' in status
    
    def test_verify_all_models_empty_cache(self, manager):
        """Test verification when no models are cached."""
        results = manager.verify_all_models()
        
        assert len(results) == 6
        # All should be False (not downloaded)
        assert all(not valid for valid in results.values())
    
    def test_get_cache_size(self, manager, test_file):
        """Test cache size calculation."""
        test_path, _, size = test_file
        
        # Move test file to cache
        cached_file = manager.cache_dir / "cached_model.dat"
        test_path.rename(cached_file)
        
        cache_size = manager.get_cache_size()
        
        # Should include at least the test file size
        assert cache_size >= size
    
    def test_clear_cache(self, manager, test_file):
        """Test clearing model cache."""
        test_path, _, _ = test_file
        
        # Create a cached file
        cached_file = manager.cache_dir / "cached.dat"
        test_path.rename(cached_file)
        
        # Clear cache
        manager.clear_cache()
        
        # Cache directory should still exist but be empty
        assert manager.cache_dir.exists()
        assert not cached_file.exists()
        assert len(list(manager.cache_dir.iterdir())) <= 1  # Only metadata file might exist
    
    @patch('src.detectors.model_manager.urlopen')
    def test_download_model_securely_success(self, mock_urlopen, manager, temp_cache_dir):
        """Test successful secure model download."""
        model_name = "blazeface_edgetpu"
        model_info = manager.MODEL_REGISTRY[model_name]
        target_path = temp_cache_dir / f"{model_name}.dat"
        
        # Create mock response with correct size and content
        mock_content = b"X" * model_info.size
        mock_response = MagicMock()
        mock_response.headers.get.return_value = str(model_info.size)
        mock_response.read.side_effect = [mock_content, b""]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        
        mock_urlopen.return_value = mock_response
        
        # Mock verify_model_integrity to pass
        with patch.object(manager, 'verify_model_integrity', return_value=(True, "OK")):
            result_path = manager._download_model_securely(model_name, target_path)
        
        assert result_path == target_path
        assert target_path.exists()
    
    @patch('src.detectors.model_manager.urlopen')
    def test_download_model_securely_size_exceeded(self, mock_urlopen, manager, temp_cache_dir):
        """Test rejection of oversized downloads."""
        model_name = "blazeface_edgetpu"
        target_path = temp_cache_dir / f"{model_name}.dat"
        
        # Mock response with excessive size
        mock_response = MagicMock()
        mock_response.headers.get.return_value = str(manager.max_download_size + 1000)
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        
        mock_urlopen.return_value = mock_response
        
        # Should raise ValueError for oversized model
        with pytest.raises(ValueError, match="too large"):
            manager._download_model_securely(model_name, target_path)
    
    @patch('src.detectors.model_manager.urlopen')
    def test_download_model_securely_verification_failed(self, mock_urlopen, manager, temp_cache_dir):
        """Test handling of verification failure after download."""
        model_name = "blazeface_edgetpu"
        model_info = manager.MODEL_REGISTRY[model_name]
        target_path = temp_cache_dir / f"{model_name}.dat"
        
        # Create mock response
        mock_content = b"X" * model_info.size
        mock_response = MagicMock()
        mock_response.headers.get.return_value = str(model_info.size)
        mock_response.read.side_effect = [mock_content, b""]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        
        mock_urlopen.return_value = mock_response
        
        # Mock verify_model_integrity to fail
        with patch.object(manager, 'verify_model_integrity', return_value=(False, "Hash mismatch")):
            with pytest.raises(ValueError, match="failed verification"):
                manager._download_model_securely(model_name, target_path)
        
        # Temporary file should be cleaned up
        temp_file = target_path.with_suffix('.tmp')
        assert not temp_file.exists()
    
    @patch('src.detectors.model_manager.urlopen')
    def test_download_model_securely_network_error(self, mock_urlopen, manager, temp_cache_dir):
        """Test handling of network errors during download."""
        model_name = "blazeface_edgetpu"
        target_path = temp_cache_dir / f"{model_name}.dat"
        
        # Mock network error
        mock_urlopen.side_effect = URLError("Network error")
        
        with pytest.raises(URLError):
            manager._download_model_securely(model_name, target_path)
        
        # Temporary file should be cleaned up
        temp_file = target_path.with_suffix('.tmp')
        assert not temp_file.exists()
    
    def test_get_model_path_unknown_model(self, manager):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            manager.get_model_path("nonexistent_model", auto_download=False)
    
    def test_get_model_path_not_found_no_download(self, manager):
        """Test that missing model with auto_download=False raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            manager.get_model_path("blazeface_edgetpu", auto_download=False)
    
    def test_get_model_with_integrity_verification(self, manager, temp_cache_dir):
        """Test get_model method with integrity verification."""
        model_name = "blazeface_edgetpu"
        model_path = "models/blazeface.tflite"
        
        # Create a mock model file in cache with correct properties
        model_info = manager.MODEL_REGISTRY[model_name]
        cached_path = manager.cache_dir / model_path
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with correct size but wrong hash (will fail verification)
        with open(cached_path, 'wb') as f:
            f.write(b"X" * model_info.size)
        
        # get_model should detect invalid file and try to delete it
        # Since we can't actually download, it will fall back to creating a mock
        result = manager.get_model(model_name, model_path)
        
        # Should return some path (either downloaded or mock)
        assert result.exists()
    
    def test_hmac_verification_not_required_without_key(self, manager, test_file):
        """Test that HMAC verification is skipped when no key is provided."""
        test_path, _, _ = test_file
        
        model_info = ModelInfo(
            name='test',
            url='https://example.com/test.dat',
            sha256_hash='a' * 64,
            size=1024,
            format='onnx',
            signature='some_signature'  # Signature present but no key
        )
        
        # Should return True (skip HMAC verification)
        result = manager._verify_hmac(test_path, model_info)
        assert result is True


class TestSecurityControls:
    """Test security-specific functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_tampered_file_detection(self, temp_cache_dir):
        """Test detection of tampered model files."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        # Create a file with correct size but wrong content
        model_name = "blazeface_edgetpu"
        model_info = manager.MODEL_REGISTRY[model_name]
        test_path = temp_cache_dir / "tampered.dat"
        
        with open(test_path, 'wb') as f:
            # Write correct size but tampered content
            f.write(b"TAMPERED" * (model_info.size // 8))
        
        # Verification should fail
        is_valid, error = manager.verify_model_integrity(model_name, test_path)
        
        assert not is_valid
        assert "Hash mismatch" in error or "integrity" in error.lower()
    
    def test_size_validation_prevents_oversized_files(self, temp_cache_dir):
        """Test that oversized files are rejected."""
        manager = ModelManager(
            cache_dir=temp_cache_dir,
            max_download_size=1024  # Very small limit
        )
        
        # Create a file larger than expected
        model_name = "blazeface_edgetpu"
        model_info = manager.MODEL_REGISTRY[model_name]
        test_path = temp_cache_dir / "oversized.dat"
        
        # Write more data than expected size
        with open(test_path, 'wb') as f:
            f.write(b"X" * (model_info.size + 1000))
        
        # Verification should fail on size check
        is_valid, error = manager.verify_model_integrity(model_name, test_path)
        
        assert not is_valid
        assert "Size mismatch" in error
    
    def test_hash_calculation_consistency(self, temp_cache_dir):
        """Test that hash calculation is consistent across multiple calls."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        # Create test file
        test_path = temp_cache_dir / "consistency_test.dat"
        test_content = b"Consistent content for testing"
        
        with open(test_path, 'wb') as f:
            f.write(test_content)
        
        # Calculate hash multiple times
        hash1 = manager.calculate_file_hash(test_path)
        hash2 = manager.calculate_file_hash(test_path)
        hash3 = manager.calculate_file_hash(test_path)
        
        # All should be identical
        assert hash1 == hash2 == hash3
        
        # And match the expected hash
        expected = hashlib.sha256(test_content).hexdigest()
        assert hash1 == expected
    
    def test_placeholder_checksums_removed(self):
        """Test that no placeholder checksums remain in production registry."""
        manager = ModelManager()
        
        # Verify all models have proper SHA-256 hashes
        for model_name, model_info in manager.MODEL_REGISTRY.items():
            assert model_info.sha256_hash != 'placeholder_checksum'
            assert len(model_info.sha256_hash) == 64
            # Verify it's valid hex
            try:
                int(model_info.sha256_hash, 16)
            except ValueError:
                pytest.fail(f"Model {model_name} has invalid SHA-256 hash format")


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_get_model_method_still_works(self, temp_cache_dir):
        """Test that legacy get_model method still functions."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        # get_model should still work (returns mock for unavailable models)
        result = manager.get_model("blazeface_edgetpu", "models/blazeface.tflite")
        
        assert isinstance(result, Path)
        assert result.exists()
    
    def test_list_cached_models_still_works(self, temp_cache_dir):
        """Test that legacy list_cached_models method still works."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        cached_models = manager.list_cached_models()
        
        assert isinstance(cached_models, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
