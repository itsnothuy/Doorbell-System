#!/usr/bin/env python3
"""
Demo Data Generator

Generates realistic demo data for testing and demonstration purposes.
"""

import random
from typing import Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class PersonData:
    """Represents a person for demo purposes."""
    name: str
    role: str
    photos: List[str]
    confidence: float


class DemoDataGenerator:
    """
    Generates realistic demo data for system demonstrations.
    
    This includes:
    - Sample person data
    - Detection events
    - Performance metrics
    - System logs
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        random.seed(seed)
        self.known_persons = self._generate_known_persons()
    
    def _generate_known_persons(self) -> List[PersonData]:
        """Generate sample known persons."""
        return [
            PersonData(
                name="John Smith",
                role="Homeowner",
                photos=["john_front.jpg", "john_left.jpg", "john_right.jpg"],
                confidence=0.91
            ),
            PersonData(
                name="Sarah Smith",
                role="Spouse",
                photos=["sarah_front.jpg", "sarah_profile.jpg"],
                confidence=0.89
            ),
            PersonData(
                name="Emma Smith",
                role="Daughter",
                photos=["emma_1.jpg", "emma_2.jpg", "emma_3.jpg"],
                confidence=0.95
            ),
            PersonData(
                name="Bob Wilson",
                role="Friend",
                photos=["bob_1.jpg", "bob_2.jpg"],
                confidence=0.87
            )
        ]
    
    def generate_detection_event(self, event_type: str = "random") -> Dict[str, Any]:
        """
        Generate a detection event.
        
        Args:
            event_type: Type of event ('known_person', 'unknown_person', or 'random')
            
        Returns:
            Dictionary with event data
        """
        if event_type == "random":
            event_type = random.choice(["known_person", "unknown_person"])
        
        timestamp = datetime.now()
        
        if event_type == "known_person":
            person = random.choice(self.known_persons)
            return {
                'timestamp': timestamp,
                'event_type': 'known_person',
                'person_name': person.name,
                'person_role': person.role,
                'confidence': person.confidence + random.uniform(-0.05, 0.05),
                'processing_time': random.uniform(0.25, 0.40),
                'notification_type': 'silent'
            }
        else:
            return {
                'timestamp': timestamp,
                'event_type': 'unknown_person',
                'person_name': 'Unknown',
                'confidence': 0.0,
                'processing_time': random.uniform(0.35, 0.50),
                'notification_type': 'alert'
            }
    
    def generate_performance_metrics(self) -> Dict[str, Any]:
        """
        Generate realistic performance metrics.
        
        Returns:
            Dictionary with performance data
        """
        return {
            'cpu_usage': random.uniform(15, 30),
            'memory_usage': random.uniform(35, 55),
            'temperature': random.uniform(38, 48),
            'avg_processing_time': random.uniform(0.25, 0.40),
            'detection_fps': random.uniform(10, 15),
            'accuracy_rate': random.uniform(95, 98),
            'false_positive_rate': random.uniform(1.5, 3.0)
        }
    
    def generate_event_history(self, num_events: int = 50) -> List[Dict[str, Any]]:
        """
        Generate a history of events.
        
        Args:
            num_events: Number of events to generate
            
        Returns:
            List of event dictionaries
        """
        events = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(num_events):
            # Generate event at random time within the past week
            event_time = base_time + timedelta(
                seconds=random.uniform(0, 7 * 24 * 3600)
            )
            
            event = self.generate_detection_event()
            event['timestamp'] = event_time
            events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        return events
    
    def generate_system_logs(self, num_entries: int = 100) -> List[Dict[str, Any]]:
        """
        Generate system log entries.
        
        Args:
            num_entries: Number of log entries to generate
            
        Returns:
            List of log entry dictionaries
        """
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
        log_messages = [
            'System started successfully',
            'Camera initialized',
            'Face detection model loaded',
            'Event processed successfully',
            'Database updated',
            'Notification sent',
            'Performance metrics collected',
            'Backup completed',
            'System health check passed',
            'Configuration updated'
        ]
        
        logs = []
        base_time = datetime.now() - timedelta(days=1)
        
        for i in range(num_entries):
            log_time = base_time + timedelta(
                seconds=random.uniform(0, 24 * 3600)
            )
            
            logs.append({
                'timestamp': log_time,
                'level': random.choice(log_levels),
                'message': random.choice(log_messages),
                'module': random.choice(['camera', 'detection', 'recognition', 'notification', 'system'])
            })
        
        logs.sort(key=lambda x: x['timestamp'])
        
        return logs


if __name__ == "__main__":
    # Test data generation
    generator = DemoDataGenerator()
    
    print("Sample Person Data:")
    for person in generator.known_persons:
        print(f"  {person.name} ({person.role}) - {len(person.photos)} photos")
    
    print("\nSample Detection Event:")
    event = generator.generate_detection_event()
    print(f"  Type: {event['event_type']}")
    print(f"  Person: {event['person_name']}")
    print(f"  Confidence: {event['confidence']:.2f}")
    
    print("\nSample Performance Metrics:")
    metrics = generator.generate_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
