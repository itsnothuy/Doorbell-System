#!/bin/bash
echo "🧪 Testing Doorbell Security System on macOS..."
source venv/bin/activate
export DEVELOPMENT_MODE=true

python3 -c "
import sys
sys.path.append('.')
from src.platform_detector import platform_detector
from src.camera_handler import CameraHandler
from src.telegram_notifier import TelegramNotifier
from src.gpio_handler import GPIOHandler
from src.face_manager import FaceManager

print('🍎 Testing macOS components...')
print(f'Platform: {platform_detector.system} {platform_detector.machine}')
print(f'Development mode: {platform_detector.is_development}')
print()

# Test camera
try:
    camera = CameraHandler()
    camera.initialize()
    print(f'✅ Camera: OK ({camera.camera_type} mode for testing)')
    camera.cleanup()
except Exception as e:
    print(f'❌ Camera: {e}')

# Test Telegram
try:
    notifier = TelegramNotifier()
    if notifier.initialized:
        print('✅ Telegram: OK')
    else:
        print('⚠️  Telegram: Not configured (edit credentials_telegram.py)')
except Exception as e:
    print(f'❌ Telegram: {e}')

# Test GPIO (mock)
try:
    gpio = GPIOHandler()
    print('✅ GPIO: OK (Mock mode for testing)')
    gpio.cleanup()
except Exception as e:
    print(f'❌ GPIO: {e}')

# Test Face Recognition
try:
    face_manager = FaceManager()
    face_manager.load_known_faces()
    face_manager.load_blacklist_faces()
    stats = face_manager.get_stats()
    print(f'✅ Face Recognition: {stats}')
except Exception as e:
    print(f'❌ Face Recognition: {e}')

print()
print('🎉 macOS test completed!')
print('💡 To start the web interface: ./scripts/start-web.sh')
print('🌐 Then open: http://localhost:5000')
"
