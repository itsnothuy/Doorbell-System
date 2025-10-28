#!/usr/bin/env python3
"""
Tests for Video Streamer

Tests the live video streaming implementation.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock, patch

from src.streaming.video_streamer import VideoStreamer


class TestVideoStreamer:
    """Test suite for Video Streamer."""
    
    @pytest.fixture
    def mock_camera_handler(self):
        """Create mock camera handler."""
        camera = Mock()
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        camera.capture_image = Mock(return_value=dummy_image)
        return camera
    
    @pytest.fixture
    def video_streamer(self, mock_camera_handler):
        """Create video streamer instance."""
        return VideoStreamer(mock_camera_handler)
    
    def test_initialization(self, video_streamer, mock_camera_handler):
        """Test video streamer initializes correctly."""
        assert video_streamer.camera_handler == mock_camera_handler
        assert len(video_streamer.active_streams) == 0
        assert 'low' in video_streamer.quality_settings
        assert 'medium' in video_streamer.quality_settings
        assert 'high' in video_streamer.quality_settings
    
    def test_quality_settings(self, video_streamer):
        """Test quality settings configuration."""
        low_settings = video_streamer.quality_settings['low']
        assert low_settings['width'] == 320
        assert low_settings['height'] == 240
        assert low_settings['fps'] == 10
        
        medium_settings = video_streamer.quality_settings['medium']
        assert medium_settings['width'] == 640
        assert medium_settings['height'] == 480
        assert medium_settings['fps'] == 15
        
        high_settings = video_streamer.quality_settings['high']
        assert high_settings['width'] == 1280
        assert high_settings['height'] == 720
        assert high_settings['fps'] == 20
    
    def test_stop_stream(self, video_streamer):
        """Test stopping a video stream."""
        # Start a stream
        video_streamer.active_streams['client_1'] = True
        
        # Stop it
        video_streamer.stop_stream('client_1')
        
        assert video_streamer.active_streams['client_1'] is False
    
    def test_cleanup_stream(self, video_streamer):
        """Test stream cleanup."""
        # Setup stream
        video_streamer.active_streams['client_1'] = True
        
        # Cleanup
        video_streamer._cleanup_stream('client_1')
        
        assert 'client_1' not in video_streamer.active_streams
    
    def test_create_video_stream_initialization(self, video_streamer, mock_camera_handler):
        """Test video stream creation and initialization."""
        client_id = 'test_client'
        
        # Create generator
        stream_gen = video_streamer.create_video_stream(client_id, 'low')
        
        # Check that client is registered
        assert client_id in video_streamer.active_streams
        assert video_streamer.active_streams[client_id] is True
        
        # Stop the stream to prevent infinite loop
        video_streamer.stop_stream(client_id)
        
        # Consume generator to trigger cleanup
        try:
            for _ in stream_gen:
                break
        except StopIteration:
            pass
    
    def test_create_video_stream_frame_encoding(self, video_streamer, mock_camera_handler):
        """Test that frames are properly encoded."""
        client_id = 'test_client'
        
        # Create generator
        stream_gen = video_streamer.create_video_stream(client_id, 'low')
        
        # Get first frame
        try:
            frame = next(stream_gen)
            
            # Verify MJPEG format
            assert frame.startswith(b'--frame\r\n')
            assert b'Content-Type: image/jpeg' in frame
            
            # Stop stream
            video_streamer.stop_stream(client_id)
        except StopIteration:
            pass
        finally:
            # Ensure cleanup
            if client_id in video_streamer.active_streams:
                video_streamer.stop_stream(client_id)
