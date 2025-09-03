"""
Web interface for testing and monitoring the doorbell system
"""

import os
import logging
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json

logger = logging.getLogger(__name__)


class WebInterface:
    """Web interface for testing and monitoring"""
    
    def __init__(self, doorbell_system=None):
        self.app = Flask(__name__, 
                         template_folder=str(Path(__file__).parent.parent / 'templates'),
                         static_folder=str(Path(__file__).parent.parent / 'static'))
        CORS(self.app)
        
        self.doorbell_system = doorbell_system
        self.setup_routes()
        
        logger.info("Web interface initialized")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status"""
            try:
                if not self.doorbell_system:
                    return jsonify({
                        'status': 'error',
                        'message': 'System not initialized'
                    })
                
                # Get component statuses
                camera_info = self.doorbell_system.camera.get_camera_info()
                gpio_status = self.doorbell_system.gpio.get_gpio_status()
                face_stats = self.doorbell_system.face_manager.get_stats()
                
                return jsonify({
                    'status': 'online',
                    'timestamp': datetime.now().isoformat(),
                    'camera': camera_info,
                    'gpio': gpio_status,
                    'faces': face_stats,
                    'telegram': {
                        'initialized': self.doorbell_system.notifier.initialized
                    }
                })
                
            except Exception as e:
                logger.error(f"Status API error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/trigger-doorbell', methods=['POST'])
        def trigger_doorbell():
            """Trigger doorbell simulation"""
            try:
                if not self.doorbell_system:
                    return jsonify({
                        'status': 'error',
                        'message': 'System not initialized'
                    })
                
                # Simulate doorbell press
                self.doorbell_system.gpio.simulate_doorbell_press()
                
                return jsonify({
                    'status': 'success',
                    'message': 'Doorbell triggered',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Doorbell trigger error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/capture-photo')
        def capture_photo():
            """Capture a photo from camera"""
            try:
                if not self.doorbell_system:
                    return jsonify({
                        'status': 'error',
                        'message': 'System not initialized'
                    })
                
                # Capture image
                image = self.doorbell_system.camera.capture_image()
                if image is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to capture image'
                    })
                
                # Save image
                timestamp = datetime.now()
                filename = f"test_capture_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                capture_path = self.doorbell_system.settings.CAPTURES_DIR / filename
                
                self.doorbell_system.camera.save_image(image, capture_path)
                
                # Convert to base64 for display
                with open(capture_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                return jsonify({
                    'status': 'success',
                    'image_data': f"data:image/jpeg;base64,{image_data}",
                    'filename': filename,
                    'timestamp': timestamp.isoformat()
                })
                
            except Exception as e:
                logger.error(f"Photo capture error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/test-face-recognition', methods=['POST'])
        def test_face_recognition():
            """Test face recognition on current camera view"""
            try:
                if not self.doorbell_system:
                    return jsonify({
                        'status': 'error',
                        'message': 'System not initialized'
                    })
                
                # Capture image
                image = self.doorbell_system.camera.capture_image()
                if image is None:
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to capture image'
                    })
                
                # Detect faces
                faces = self.doorbell_system.face_manager.detect_faces(image)
                results = []
                
                for i, face_encoding in enumerate(faces):
                    result = self.doorbell_system.face_manager.identify_face(face_encoding)
                    results.append({
                        'face_id': i + 1,
                        'status': result['status'],
                        'name': result.get('name'),
                        'confidence': round(result.get('confidence', 0), 3),
                        'distance': round(result.get('distance', float('inf')), 3)
                    })
                
                return jsonify({
                    'status': 'success',
                    'faces_detected': len(faces),
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Face recognition test error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/set-led', methods=['POST'])
        def set_led():
            """Set LED status"""
            try:
                data = request.get_json()
                status = data.get('status', 'idle')
                duration = data.get('duration')
                
                if not self.doorbell_system:
                    return jsonify({
                        'status': 'error',
                        'message': 'System not initialized'
                    })
                
                self.doorbell_system.gpio.set_status_led(status, duration)
                
                return jsonify({
                    'status': 'success',
                    'led_status': status,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"LED control error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/test-notification', methods=['POST'])
        def test_notification():
            """Send test notification"""
            try:
                data = request.get_json()
                message = data.get('message', 'Test notification from web interface')
                priority = data.get('priority', 'normal')
                
                if not self.doorbell_system:
                    return jsonify({
                        'status': 'error',
                        'message': 'System not initialized'
                    })
                
                success = self.doorbell_system.notifier.send_alert(
                    message=message,
                    priority=priority
                )
                
                return jsonify({
                    'status': 'success' if success else 'error',
                    'message': 'Notification sent' if success else 'Failed to send notification',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Notification test error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/recent-events')
        def get_recent_events():
            """Get recent visitor events"""
            try:
                events_file = self.doorbell_system.settings.LOGS_DIR / 'events.log'
                events = []
                
                if events_file.exists():
                    with open(events_file, 'r') as f:
                        lines = f.readlines()
                        # Get last 20 events
                        for line in lines[-20:]:
                            try:
                                event = eval(line.strip())  # Note: In production, use json.loads
                                events.append(event)
                            except:
                                continue
                
                return jsonify({
                    'status': 'success',
                    'events': events
                })
                
            except Exception as e:
                logger.error(f"Recent events error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/logs')
        def get_logs():
            """Get recent system logs"""
            try:
                log_file = self.doorbell_system.settings.LOGS_DIR / 'doorbell.log'
                logs = []
                
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Get last 50 log lines
                        logs = [line.strip() for line in lines[-50:]]
                
                return jsonify({
                    'status': 'success',
                    'logs': logs
                })
                
            except Exception as e:
                logger.error(f"Logs retrieval error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web interface"""
        try:
            logger.info(f"Starting web interface on http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Failed to start web interface: {e}")
            raise


def create_web_app(doorbell_system=None):
    """Factory function to create web app"""
    web_interface = WebInterface(doorbell_system)
    return web_interface.app
