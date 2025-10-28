#!/usr/bin/env python3
"""
Live Video Streaming

Implements real-time video streaming endpoints for web interface
with quality adaptation and efficient frame delivery.
"""

import cv2
import time
import logging
import threading
from typing import Dict, Any, Iterator, Optional

logger = logging.getLogger(__name__)


class VideoStreamer:
    """Live video streaming for web interface."""
    
    def __init__(self, camera_handler):
        """
        Initialize video streamer.
        
        Args:
            camera_handler: Camera handler instance
        """
        self.camera_handler = camera_handler
        self.active_streams: Dict[str, bool] = {}
        self.stream_lock = threading.RLock()
        
        # Quality settings
        self.quality_settings = {
            'low': {'width': 320, 'height': 240, 'fps': 10, 'quality': 70},
            'medium': {'width': 640, 'height': 480, 'fps': 15, 'quality': 80},
            'high': {'width': 1280, 'height': 720, 'fps': 20, 'quality': 90}
        }
        
        logger.info("Video Streamer initialized")
    
    def create_video_stream(self, client_id: str, quality: str = 'medium') -> Iterator[bytes]:
        """
        Create video stream for client.
        
        Args:
            client_id: Unique client identifier
            quality: Video quality setting
            
        Yields:
            MJPEG video frames
        """
        try:
            # Register client stream
            with self.stream_lock:
                self.active_streams[client_id] = True
            
            # Get quality settings
            settings = self.quality_settings.get(quality, self.quality_settings['medium'])
            frame_interval = 1.0 / settings['fps']
            
            logger.info(f"Starting video stream for {client_id} (quality: {quality})")
            
            last_frame_time = 0
            
            while self.active_streams.get(client_id, False):
                current_time = time.time()
                
                # Respect frame rate limit
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.01)
                    continue
                
                # Capture frame
                try:
                    frame = self.camera_handler.capture_image()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                except Exception as e:
                    logger.warning(f"Failed to capture frame: {e}")
                    time.sleep(0.1)
                    continue
                
                # Resize frame based on quality
                try:
                    resized_frame = cv2.resize(
                        frame,
                        (settings['width'], settings['height'])
                    )
                    
                    # Encode as JPEG
                    ret, jpeg_frame = cv2.imencode(
                        '.jpg',
                        resized_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, settings['quality']]
                    )
                    
                    if ret:
                        # Yield frame in MJPEG format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' +
                               jpeg_frame.tobytes() + b'\r\n')
                        
                        last_frame_time = current_time
                    else:
                        logger.warning("Failed to encode video frame")
                except Exception as e:
                    logger.warning(f"Frame processing error: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Video streaming error for {client_id}: {e}")
        
        finally:
            self._cleanup_stream(client_id)
    
    def stop_stream(self, client_id: str) -> None:
        """Stop video stream for client."""
        with self.stream_lock:
            if client_id in self.active_streams:
                self.active_streams[client_id] = False
    
    def _cleanup_stream(self, client_id: str) -> None:
        """Clean up stream resources."""
        with self.stream_lock:
            if client_id in self.active_streams:
                del self.active_streams[client_id]
        
        logger.info(f"Cleaned up video stream for {client_id}")
