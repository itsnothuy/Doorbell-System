#!/usr/bin/env python3
"""
Test suite for the Doorbell Security System
"""

import sys
import os
import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.face_manager import FaceManager
from src.camera_handler import MockCameraHandler
from src.telegram_notifier import MockTelegramNotifier
from src.gpio_handler import MockGPIOHandler


class TestFaceManager(unittest.TestCase):
    """Test face recognition functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.face_manager = FaceManager()
    
    def test_face_detection_empty_image(self):
        """Test face detection with empty image"""
        # Create a blank image
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Should return empty list for blank image
        faces = self.face_manager.detect_faces(blank_image)
        self.assertEqual(len(faces), 0)
    
    def test_identify_face_empty_database(self):
        """Test face identification with empty database"""
        # Create a dummy face encoding
        dummy_encoding = np.random.rand(128)
        
        # Should return unknown status
        result = self.face_manager.identify_face(dummy_encoding)
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['name'])
    
    def test_face_database_stats(self):
        """Test face database statistics"""
        stats = self.face_manager.get_stats()
        
        self.assertIn('known_faces', stats)
        self.assertIn('blacklist_faces', stats)
        self.assertIn('known_names', stats)
        self.assertIn('blacklist_names', stats)
        
        # Should be integers
        self.assertIsInstance(stats['known_faces'], int)
        self.assertIsInstance(stats['blacklist_faces'], int)


class TestCameraHandler(unittest.TestCase):
    """Test camera functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.camera = MockCameraHandler()
    
    def test_camera_initialization(self):
        """Test camera initialization"""
        self.camera.initialize()
        self.assertTrue(self.camera.is_initialized)
        self.assertEqual(self.camera.camera_type, 'mock')
    
    def test_image_capture(self):
        """Test image capture"""
        self.camera.initialize()
        
        image = self.camera.capture_image()
        self.assertIsNotNone(image)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(len(image.shape), 3)  # Should be RGB
    
    def test_camera_info(self):
        """Test camera information"""
        self.camera.initialize()
        
        info = self.camera.get_camera_info()
        self.assertIn('initialized', info)
        self.assertIn('camera_type', info)
        self.assertIn('resolution', info)
        self.assertTrue(info['initialized'])
    
    def test_image_save(self):
        """Test image saving"""
        self.camera.initialize()
        
        image = self.camera.capture_image()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            self.camera.save_image(image, temp_path)
            self.assertTrue(temp_path.exists())
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestTelegramNotifier(unittest.TestCase):
    """Test Telegram notification functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.notifier = MockTelegramNotifier()
    
    def test_notifier_initialization(self):
        """Test notifier initialization"""
        self.assertTrue(self.notifier.initialized)
    
    def test_send_alert(self):
        """Test sending alerts"""
        result = self.notifier.send_alert("Test message", priority='normal')
        self.assertTrue(result)
        
        messages = self.notifier.get_sent_messages()
        self.assertEqual(len(messages), 1)
        self.assertIn("Test message", messages[0]['message'])
    
    def test_send_alert_with_image(self):
        """Test sending alerts with image"""
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
            temp_path = Path(temp_file.name)
            
            result = self.notifier.send_alert(
                "Test with image", 
                image_path=temp_path, 
                priority='urgent'
            )
            self.assertTrue(result)
    
    def test_priority_levels(self):
        """Test different priority levels"""
        priorities = ['low', 'normal', 'urgent']
        
        for priority in priorities:
            self.notifier.send_alert(f"Test {priority}", priority=priority)
        
        messages = self.notifier.get_sent_messages()
        self.assertEqual(len(messages), 3)


class TestGPIOHandler(unittest.TestCase):
    """Test GPIO functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.gpio = MockGPIOHandler()
    
    def test_gpio_initialization(self):
        """Test GPIO initialization"""
        self.assertTrue(self.gpio.initialized)
    
    def test_doorbell_setup(self):
        """Test doorbell button setup"""
        callback = Mock()
        self.gpio.setup_doorbell_button(callback)
        
        # Simulate button press
        self.gpio.simulate_doorbell_press()
        
        # Should increment counter
        self.assertEqual(self.gpio.button_pressed_count, 1)
    
    def test_led_control(self):
        """Test LED control"""
        # Test different states
        states = ['idle', 'processing', 'known', 'unknown', 'alert', 'error']
        
        for state in states:
            self.gpio.set_status_led(state)
            # Just ensure no exceptions are raised
    
    def test_led_test(self):
        """Test LED testing function"""
        # Should not raise any exceptions
        self.gpio.test_leds()
    
    def test_gpio_status(self):
        """Test GPIO status reporting"""
        status = self.gpio.get_gpio_status()
        
        self.assertIn('initialized', status)
        self.assertIn('doorbell_pin', status)
        self.assertIn('led_pins', status)
        self.assertIn('led_states', status)


class TestSystemIntegration(unittest.TestCase):
    """Test system integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.face_manager = FaceManager()
        self.camera = MockCameraHandler()
        self.notifier = MockTelegramNotifier()
        self.gpio = MockGPIOHandler()
    
    def test_component_initialization(self):
        """Test all components can be initialized"""
        self.camera.initialize()
        
        # All components should initialize without error
        self.assertIsNotNone(self.face_manager)
        self.assertTrue(self.camera.is_initialized)
        self.assertTrue(self.notifier.initialized)
        self.assertTrue(self.gpio.initialized)
    
    def test_visitor_processing_workflow(self):
        """Test the complete visitor processing workflow"""
        # Initialize components
        self.camera.initialize()
        
        # Setup GPIO callback
        callback_called = False
        
        def mock_callback(channel):
            nonlocal callback_called
            callback_called = True
        
        self.gpio.setup_doorbell_button(mock_callback)
        
        # Simulate doorbell press
        self.gpio.simulate_doorbell_press()
        
        # Verify callback was triggered
        self.assertTrue(callback_called)
        
        # Capture image
        image = self.camera.capture_image()
        self.assertIsNotNone(image)
        
        # Process face detection
        faces = self.face_manager.detect_faces(image)
        # Note: Mock image likely won't have faces, so this is just testing the pipeline
        
        # Send notification
        result = self.notifier.send_alert("Test visitor detected")
        self.assertTrue(result)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with uninitialized camera
        uninit_camera = MockCameraHandler()
        image = uninit_camera.capture_image()
        self.assertIsNone(image)
        
        # Test GPIO cleanup
        self.gpio.cleanup()
        self.assertFalse(self.gpio.initialized)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestFaceManager,
        TestCameraHandler,
        TestTelegramNotifier,
        TestGPIOHandler,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Doorbell Security System Tests")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
