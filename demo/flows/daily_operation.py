#!/usr/bin/env python3
"""
Daily Operation Flow Demo

Demonstrates typical daily operation scenarios including known person detection,
unknown person alerts, and real-time dashboard interactions.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    """Represents a face detection event."""
    
    timestamp: datetime
    event_type: str  # known_person, unknown_person, motion_only
    trigger: str
    processing_pipeline: List[Dict[str, Any]] = field(default_factory=list)
    total_processing_time: float = 0.0
    notifications_sent: List[str] = field(default_factory=list)
    event_stored: bool = True
    user_experience: str = ""
    person_name: Optional[str] = None
    confidence: float = 0.0


class DailyOperationFlow:
    """
    Demonstrates typical daily operation scenarios.
    
    This includes:
    - Known person detection
    - Unknown person alerts
    - Real-time dashboard interactions
    - Mobile experience
    """
    
    def __init__(self):
        self.events: List[DetectionEvent] = []
        self.dashboard_status: Dict[str, Any] = {}
        self._initialize_dashboard()
    
    def _initialize_dashboard(self) -> None:
        """Initialize the dashboard status."""
        self.dashboard_status = {
            'camera_status': 'Online (15 FPS)',
            'detection_engine': 'Active (CPU: 23%)',
            'storage_usage': '2.3GB / 32GB (7%)',
            'uptime': '7 days, 14 hours',
            'events_today': 0,
            'known_faces': 4,
            'recent_activity': 'System idle'
        }
    
    def demo_known_person_detection(self) -> DetectionEvent:
        """
        Demonstrate a known person detection event.
        
        Returns:
            DetectionEvent object with complete processing pipeline
        """
        logger.info("Simulating known person detection")
        
        event = DetectionEvent(
            timestamp=datetime.now(),
            event_type='known_person',
            trigger='Motion detected at front door',
            person_name='John Smith',
            confidence=0.91
        )
        
        # Simulate processing pipeline
        event.processing_pipeline = [
            {
                'stage': 'Motion Detection',
                'duration': 0.05,
                'result': 'Motion detected in zone 1 (confidence: 0.87)'
            },
            {
                'stage': 'Frame Capture',
                'duration': 0.02,
                'result': 'High-quality frame captured (640x480)'
            },
            {
                'stage': 'Face Detection',
                'duration': 0.15,
                'result': '1 face detected (confidence: 0.94)'
            },
            {
                'stage': 'Face Recognition',
                'duration': 0.08,
                'result': f'Recognized: {event.person_name} (confidence: {event.confidence})'
            },
            {
                'stage': 'Event Processing',
                'duration': 0.02,
                'result': 'Welcome notification generated'
            }
        ]
        
        event.total_processing_time = sum(s['duration'] for s in event.processing_pipeline)
        
        event.notifications_sent = [
            f'Web dashboard: "Welcome home, {event.person_name}!"',
            'Mobile push: Silent notification (known person)',
        ]
        
        event.user_experience = 'Seamless - no action required'
        
        self.events.append(event)
        self.dashboard_status['events_today'] += 1
        self.dashboard_status['recent_activity'] = f'{event.person_name} detected {int((datetime.now() - event.timestamp).total_seconds())} seconds ago'
        
        return event
    
    def demo_unknown_person_detection(self) -> DetectionEvent:
        """
        Demonstrate an unknown person detection event.
        
        Returns:
            DetectionEvent object with alert details
        """
        logger.info("Simulating unknown person detection")
        
        event = DetectionEvent(
            timestamp=datetime.now(),
            event_type='unknown_person',
            trigger='Motion detected at front door',
            person_name='Unknown',
            confidence=0.0
        )
        
        # Simulate processing pipeline
        event.processing_pipeline = [
            {
                'stage': 'Motion Detection',
                'duration': 0.04,
                'result': 'Motion detected in zone 1 (confidence: 0.91)'
            },
            {
                'stage': 'Frame Capture',
                'duration': 0.02,
                'result': 'High-quality frame captured (640x480)'
            },
            {
                'stage': 'Face Detection',
                'duration': 0.12,
                'result': '1 face detected (confidence: 0.88)'
            },
            {
                'stage': 'Face Recognition',
                'duration': 0.25,
                'result': 'Unknown person (closest match: 0.42 distance)'
            },
            {
                'stage': 'Alert Generation',
                'duration': 0.03,
                'result': 'Security alert generated'
            }
        ]
        
        event.total_processing_time = sum(s['duration'] for s in event.processing_pipeline)
        
        event.notifications_sent = [
            'Web dashboard: "Unknown person detected"',
            'Email alert: "Security Alert - Unknown Person"',
            'Mobile push: "Unknown visitor at front door"'
        ]
        
        event.user_experience = 'Alert sent - user action available'
        
        self.events.append(event)
        self.dashboard_status['events_today'] += 1
        self.dashboard_status['recent_activity'] = f'Unknown person detected {int((datetime.now() - event.timestamp).total_seconds())} seconds ago'
        
        return event
    
    def display_event(self, event: DetectionEvent) -> None:
        """
        Display a detection event with full details.
        
        Args:
            event: The detection event to display
        """
        print(f"\n{'='*80}")
        print(f"Event Type: {event.event_type.upper().replace('_', ' ')}")
        print(f"Timestamp: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        print(f"\n🔔 Trigger: {event.trigger}")
        
        if event.person_name and event.person_name != 'Unknown':
            print(f"👤 Person: {event.person_name} (confidence: {event.confidence:.2f})")
        else:
            print(f"❓ Person: Unknown visitor")
        
        print(f"\n⚙️  Processing Pipeline:")
        for stage in event.processing_pipeline:
            print(f"   {stage['stage']}:")
            print(f"     Duration: {stage['duration']:.3f}s")
            print(f"     Result: {stage['result']}")
        
        print(f"\n⏱️  Total Processing Time: {event.total_processing_time:.3f}s")
        
        print(f"\n📱 Notifications Sent:")
        for notification in event.notifications_sent:
            print(f"   ✓ {notification}")
        
        print(f"\n👁️  User Experience: {event.user_experience}")
        print(f"💾 Event Stored: {'Yes' if event.event_stored else 'No'}")
    
    def demo_real_time_dashboard(self) -> Dict[str, Any]:
        """
        Demonstrate the real-time dashboard interface.
        
        Returns:
            Dictionary with dashboard data
        """
        logger.info("Displaying real-time dashboard")
        
        dashboard_demo = {
            # Real-time status display
            'system_status': self.dashboard_status,
            
            # Live camera feed with overlays
            'camera_feed': {
                'stream_quality': '640x480 @ 15 FPS',
                'latency': '< 200ms',
                'detection_overlays': [
                    {
                        'type': 'face_box',
                        'coordinates': [120, 80, 200, 160],
                        'label': 'John Smith',
                        'confidence': 0.91,
                        'color': 'green'
                    },
                    {
                        'type': 'motion_zone',
                        'coordinates': [50, 50, 550, 400],
                        'active': True,
                        'sensitivity': 'medium'
                    }
                ],
                'controls': [
                    'Take snapshot',
                    'Start recording',
                    'Adjust detection zones',
                    'Camera settings'
                ]
            },
            
            # Event timeline
            'recent_events': []
        }
        
        # Add recent events to timeline
        for event in self.events[-5:]:  # Last 5 events
            time_ago = (datetime.now() - event.timestamp).total_seconds()
            if time_ago < 60:
                time_str = f"{int(time_ago)} seconds ago"
            elif time_ago < 3600:
                time_str = f"{int(time_ago/60)} minutes ago"
            else:
                time_str = f"{int(time_ago/3600)} hours ago"
            
            dashboard_demo['recent_events'].append({
                'time': time_str,
                'type': event.event_type,
                'person': event.person_name or 'Unknown',
                'confidence': event.confidence,
                'action': 'viewed' if event.event_type == 'known_person' else 'alerted'
            })
        
        return dashboard_demo
    
    def display_dashboard(self, dashboard: Dict[str, Any]) -> None:
        """
        Display the dashboard in a formatted way.
        
        Args:
            dashboard: Dashboard data to display
        """
        print(f"\n{'='*80}")
        print("REAL-TIME DASHBOARD")
        print(f"{'='*80}")
        
        print("\n📊 System Status:")
        for key, value in dashboard['system_status'].items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print("\n📹 Live Camera Feed:")
        feed = dashboard['camera_feed']
        print(f"   Quality: {feed['stream_quality']}")
        print(f"   Latency: {feed['latency']}")
        print(f"   Active Overlays: {len(feed['detection_overlays'])}")
        print(f"   Available Controls: {', '.join(feed['controls'])}")
        
        print("\n🕐 Recent Events:")
        if dashboard['recent_events']:
            for event in dashboard['recent_events']:
                icon = "🟢" if event['type'] == 'known_person' else "🔴"
                print(f"   {icon} {event['time']} - {event['person']} ({event['action']})")
        else:
            print("   No recent events")
    
    def demo_mobile_experience(self) -> Dict[str, Any]:
        """
        Demonstrate the mobile application experience.
        
        Returns:
            Dictionary with mobile experience details
        """
        logger.info("Demonstrating mobile experience")
        
        mobile_demo = {
            'push_notification_flow': {
                'notification_received': {
                    'title': 'Unknown Person Detected',
                    'body': 'Someone is at your front door',
                    'category': 'SECURITY_ALERT',
                    'data': {
                        'event_id': f'evt_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        'image_url': 'https://doorbell.local/api/events/latest/image',
                        'confidence': 0.88
                    }
                },
                'user_interaction': {
                    'tap_notification': 'Opens app to event details',
                    'swipe_actions': [
                        'View Live Feed',
                        'Add to Known Faces',
                        'Dismiss'
                    ]
                },
                'app_experience': {
                    'load_time': '< 2 seconds',
                    'image_quality': 'Full resolution available',
                    'actions_available': [
                        'View 30-second clip',
                        'Access live camera',
                        'Add person to contacts',
                        'Review similar events'
                    ]
                }
            },
            'offline_features': {
                'cached_events': 'Last 50 events available offline',
                'sync_on_reconnect': 'Automatic sync when connection restored',
                'local_notifications': 'System alerts stored locally',
                'configuration_cache': 'Settings persist offline'
            }
        }
        
        return mobile_demo
    
    def display_mobile_experience(self, mobile: Dict[str, Any]) -> None:
        """
        Display mobile experience details.
        
        Args:
            mobile: Mobile experience data
        """
        print(f"\n{'='*80}")
        print("MOBILE APPLICATION EXPERIENCE")
        print(f"{'='*80}")
        
        notification = mobile['push_notification_flow']['notification_received']
        print(f"\n📱 Push Notification:")
        print(f"   Title: {notification['title']}")
        print(f"   Body: {notification['body']}")
        print(f"   Category: {notification['category']}")
        
        interaction = mobile['push_notification_flow']['user_interaction']
        print(f"\n👆 User Interactions:")
        print(f"   Tap: {interaction['tap_notification']}")
        print(f"   Swipe Actions:")
        for action in interaction['swipe_actions']:
            print(f"     • {action}")
        
        app = mobile['push_notification_flow']['app_experience']
        print(f"\n📲 App Experience:")
        print(f"   Load Time: {app['load_time']}")
        print(f"   Image Quality: {app['image_quality']}")
        print(f"   Available Actions:")
        for action in app['actions_available']:
            print(f"     • {action}")
        
        offline = mobile['offline_features']
        print(f"\n🔌 Offline Capabilities:")
        for feature, description in offline.items():
            print(f"   • {feature.replace('_', ' ').title()}: {description}")
    
    def run_demo(self, num_events: int = 5) -> Dict[str, Any]:
        """
        Run a complete daily operation demonstration.
        
        Args:
            num_events: Number of detection events to simulate
            
        Returns:
            Dictionary with demo results
        """
        logger.info(f"Starting Daily Operation Flow Demo ({num_events} events)")
        
        print(f"\n{'='*80}")
        print("DAILY OPERATION FLOW DEMONSTRATION")
        print(f"{'='*80}")
        
        # Simulate various events
        for i in range(num_events):
            time.sleep(0.5)  # Small delay between events
            
            # Randomly choose event type (70% known, 30% unknown)
            if random.random() < 0.7:
                event = self.demo_known_person_detection()
            else:
                event = self.demo_unknown_person_detection()
            
            self.display_event(event)
            time.sleep(0.3)
        
        # Display dashboard
        dashboard = self.demo_real_time_dashboard()
        self.display_dashboard(dashboard)
        
        # Display mobile experience
        mobile = self.demo_mobile_experience()
        self.display_mobile_experience(mobile)
        
        # Generate summary
        results = {
            'total_events': len(self.events),
            'known_person_events': sum(1 for e in self.events if e.event_type == 'known_person'),
            'unknown_person_events': sum(1 for e in self.events if e.event_type == 'unknown_person'),
            'avg_processing_time': sum(e.total_processing_time for e in self.events) / len(self.events),
            'dashboard_status': self.dashboard_status
        }
        
        print(f"\n{'='*80}")
        print("DAILY OPERATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Events: {results['total_events']}")
        print(f"Known Person Events: {results['known_person_events']}")
        print(f"Unknown Person Events: {results['unknown_person_events']}")
        print(f"Avg Processing Time: {results['avg_processing_time']:.3f}s")
        
        return results


if __name__ == "__main__":
    # Run demonstration
    demo = DailyOperationFlow()
    results = demo.run_demo(num_events=5)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
