"""
Web interface for testing and monitoring the doorbell system
"""

import os
import logging
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
from werkzeug.utils import secure_filename
from PIL import Image

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

        # ------------------------------
        # Known Faces Management Routes
        # ------------------------------

        def _sanitize_person_name(raw: str) -> str:
            """Allow only alnum, space, dash, underscore; collapse spaces; trim length"""
            if not raw:
                return ''
            name = raw.strip()
            name = re.sub(r'[^a-zA-Z0-9 _-]', '', name)
            name = re.sub(r'\s+', ' ', name)
            return name[:64]

        def _slugify_name(name: str) -> str:
            slug = name.strip().lower()
            slug = re.sub(r'\s+', '_', slug)
            slug = re.sub(r'[^a-z0-9_-]', '', slug)
            return slug

        @self.app.route('/api/faces/known', methods=['GET'])
        def list_known_faces():
            try:
                if not self.doorbell_system:
                    return jsonify({ 'status': 'error', 'message': 'System not initialized' }), 400

                stats = self.doorbell_system.face_manager.get_stats()
                return jsonify({
                    'status': 'success',
                    'count': len(stats.get('known_names', [])),
                    'names': stats.get('known_names', [])
                })
            except Exception as e:
                logger.error(f"List known faces error: {e}")
                return jsonify({ 'status': 'error', 'message': str(e) }), 500

        @self.app.route('/api/faces/known/upload', methods=['POST'])
        def upload_known_face():
            try:
                if not self.doorbell_system:
                    return jsonify({ 'status': 'error', 'message': 'System not initialized' }), 400

                person_name = _sanitize_person_name(request.form.get('name', ''))
                file = request.files.get('file')
                if not person_name:
                    return jsonify({ 'status': 'error', 'message': 'Name is required' }), 400
                if not file:
                    return jsonify({ 'status': 'error', 'message': 'Image file is required' }), 400

                settings = self.doorbell_system.settings
                settings.KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                slug = _slugify_name(person_name)
                filename = f"{slug}_{timestamp}.jpg"
                save_path = settings.KNOWN_FACES_DIR / filename

                # Convert to JPEG to ensure compatibility with loader
                try:
                    image = Image.open(file.stream).convert('RGB')
                    image.save(save_path, 'JPEG', quality=85)
                except Exception as img_err:
                    return jsonify({ 'status': 'error', 'message': f'Failed to process image: {img_err}' }), 400

                # Add to face database
                success = self.doorbell_system.face_manager.add_known_face(save_path, person_name)
                if not success:
                    # Cleanup file if no face found
                    try:
                        if save_path.exists():
                            save_path.unlink()
                    except:
                        pass
                    return jsonify({ 'status': 'error', 'message': 'No face found in image' }), 400

                return jsonify({ 'status': 'success', 'name': person_name, 'filename': filename })

            except Exception as e:
                logger.error(f"Upload known face error: {e}")
                return jsonify({ 'status': 'error', 'message': str(e) }), 500

        @self.app.route('/api/faces/known/add-from-camera', methods=['POST'])
        def add_known_from_camera():
            try:
                if not self.doorbell_system:
                    return jsonify({ 'status': 'error', 'message': 'System not initialized' }), 400

                data = request.get_json(force=True, silent=True) or {}
                person_name = _sanitize_person_name(data.get('name', ''))
                if not person_name:
                    return jsonify({ 'status': 'error', 'message': 'Name is required' }), 400

                # Capture image
                image = self.doorbell_system.camera.capture_image()
                if image is None:
                    return jsonify({ 'status': 'error', 'message': 'Failed to capture image' }), 500

                settings = self.doorbell_system.settings
                settings.KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                slug = _slugify_name(person_name)
                filename = f"{slug}_{timestamp}.jpg"
                save_path = settings.KNOWN_FACES_DIR / filename

                # Save using camera handler to ensure rotation/settings applied
                self.doorbell_system.camera.save_image(image, save_path)

                # Add to face database
                success = self.doorbell_system.face_manager.add_known_face(save_path, person_name)
                if not success:
                    try:
                        if save_path.exists():
                            save_path.unlink()
                    except:
                        pass
                    return jsonify({ 'status': 'error', 'message': 'No face found in captured image' }), 400

                return jsonify({ 'status': 'success', 'name': person_name, 'filename': filename })

            except Exception as e:
                logger.error(f"Add known from camera error: {e}")
                return jsonify({ 'status': 'error', 'message': str(e) }), 500

        @self.app.route('/api/faces/known/<name>', methods=['DELETE'])
        def delete_known_face(name):
            try:
                if not self.doorbell_system:
                    return jsonify({ 'status': 'error', 'message': 'System not initialized' }), 400

                person_name = _sanitize_person_name(name)
                if not person_name:
                    return jsonify({ 'status': 'error', 'message': 'Invalid name' }), 400

                success = self.doorbell_system.face_manager.remove_known_face(person_name)
                if not success:
                    return jsonify({ 'status': 'error', 'message': 'Name not found' }), 404

                return jsonify({ 'status': 'success', 'name': person_name })

            except Exception as e:
                logger.error(f"Delete known face error: {e}")
                return jsonify({ 'status': 'error', 'message': str(e) }), 500
        
        @self.app.route('/api/faces/candidates')
        def list_face_candidates():
            """List cropped unknown faces for labeling UI"""
            try:
                if not self.doorbell_system:
                    return jsonify({ 'status': 'error', 'message': 'System not initialized' }), 400

                settings = self.doorbell_system.settings
                unknown_dir = settings.CROPPED_FACES_DIR / 'unknown'
                items = []
                if unknown_dir.exists():
                    for p in sorted(unknown_dir.glob('*.jpg'), reverse=True)[:100]:
                        items.append({
                            'filename': p.name,
                            'path': f"/api/files/cropped/unknown/{p.name}"
                        })

                return jsonify({ 'status': 'success', 'items': items })
            except Exception as e:
                logger.error(f"List face candidates error: {e}")
                return jsonify({ 'status': 'error', 'message': str(e) }), 500

        @self.app.route('/api/files/cropped/<category>/<filename>')
        def get_cropped_face(category, filename):
            try:
                if category not in ['unknown', 'known']:
                    return jsonify({ 'status': 'error', 'message': 'Invalid category' }), 400
                base = self.doorbell_system.settings.CROPPED_FACES_DIR / category
                file_path = base / filename
                if not file_path.exists():
                    return jsonify({ 'status': 'error', 'message': 'File not found' }), 404
                return send_file(str(file_path), mimetype='image/jpeg')
            except Exception as e:
                logger.error(f"Get cropped face error: {e}")
                return jsonify({ 'status': 'error', 'message': str(e) }), 500

        @self.app.route('/api/faces/label', methods=['POST'])
        def label_face_candidate():
            """Label an unknown cropped face as a known person"""
            try:
                if not self.doorbell_system:
                    return jsonify({ 'status': 'error', 'message': 'System not initialized' }), 400

                data = request.get_json(force=True)
                filename = data.get('filename', '')
                person_name = _sanitize_person_name(data.get('name', ''))
                if not filename or not person_name:
                    return jsonify({ 'status': 'error', 'message': 'filename and name are required' }), 400

                settings = self.doorbell_system.settings
                src_path = settings.CROPPED_FACES_DIR / 'unknown' / filename
                if not src_path.exists():
                    return jsonify({ 'status': 'error', 'message': 'Source file not found' }), 404

                # Move to known_faces with slug
                slug = _slugify_name(person_name)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest_filename = f"{slug}_{ts}.jpg"
                dest_path = settings.KNOWN_FACES_DIR / dest_filename

                # Ensure RGB and save
                img = Image.open(src_path).convert('RGB')
                img.save(dest_path, 'JPEG', quality=85)

                # Update database
                success = self.doorbell_system.face_manager.add_known_face(dest_path, person_name)
                if not success:
                    return jsonify({ 'status': 'error', 'message': 'No face found while adding' }), 400

                # Remove the candidate after successful labeling
                try:
                    src_path.unlink()
                except Exception:
                    pass

                return jsonify({ 'status': 'success', 'name': person_name, 'filename': dest_filename })

            except Exception as e:
                logger.error(f"Label face candidate error: {e}")
                return jsonify({ 'status': 'error', 'message': str(e) }), 500
    
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
