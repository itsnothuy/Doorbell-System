#!/usr/bin/env python3
"""
Locust Load Testing Configuration

Load testing scenarios for the Doorbell Security System web interface and API.

Usage:
    # Run with web UI
    locust -f tests/load/locustfile.py --host=http://localhost:5000
    
    # Run headless
    locust -f tests/load/locustfile.py --host=http://localhost:5000 \\
           --headless --users 50 --spawn-rate 5 --run-time 60s
    
    # With HTML report
    locust -f tests/load/locustfile.py --host=http://localhost:5000 \\
           --headless --users 100 --spawn-rate 10 --run-time 120s \\
           --html load-test-report.html
"""

from locust import HttpUser, task, between, SequentialTaskSet
import random
import json
import time


class DashboardBehavior(SequentialTaskSet):
    """Sequential user behavior for dashboard interactions."""

    @task
    def view_dashboard(self):
        """User views the main dashboard."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dashboard returned {response.status_code}")

    @task
    def check_system_status(self):
        """User checks system status via API."""
        with self.client.get("/api/system/status", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "camera_status" in data and "detection_status" in data:
                    response.success()
                else:
                    response.failure("Missing required status fields")
            else:
                response.failure(f"Status API returned {response.status_code}")

    @task
    def view_recent_events(self):
        """User views recent events."""
        with self.client.get("/api/events/recent?limit=20", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Events API returned {response.status_code}")


class CameraStreamBehavior(SequentialTaskSet):
    """User behavior for camera streaming."""

    @task
    def view_camera_page(self):
        """User navigates to camera page."""
        self.client.get("/camera")

    @task
    def get_camera_stream(self):
        """User connects to camera stream."""
        # Simulate stream connection (actual WebSocket would be different)
        self.client.get("/api/camera/stream")

    @task
    def take_snapshot(self):
        """User takes a snapshot."""
        self.client.post("/api/camera/snapshot")


class FaceManagementBehavior(SequentialTaskSet):
    """User behavior for face management."""

    @task
    def view_known_faces(self):
        """User views known faces page."""
        self.client.get("/known-faces")

    @task
    def get_known_faces_api(self):
        """User fetches known faces via API."""
        self.client.get("/api/faces")

    @task
    def view_face_details(self):
        """User views details of a specific face."""
        # Simulate viewing a random face
        face_id = random.randint(1, 10)
        self.client.get(f"/api/faces/{face_id}")


class WebInterfaceUser(HttpUser):
    """Simulates a typical web interface user."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    weight = 3  # More common user type

    tasks = {
        DashboardBehavior: 5,
        CameraStreamBehavior: 2,
        FaceManagementBehavior: 3,
    }

    def on_start(self):
        """Called when a user starts."""
        # Could authenticate here if needed
        pass


class ApiOnlyUser(HttpUser):
    """Simulates an API-only client (e.g., mobile app, automation)."""

    wait_time = between(0.5, 2)
    weight = 2

    @task(5)
    def get_system_metrics(self):
        """High-frequency metrics polling."""
        self.client.get("/api/system/metrics")

    @task(3)
    def get_recent_events(self):
        """Event polling."""
        limit = random.choice([10, 20, 50])
        self.client.get(f"/api/events/recent?limit={limit}")

    @task(2)
    def get_camera_status(self):
        """Camera status check."""
        self.client.get("/api/camera/status")

    @task(1)
    def get_detection_config(self):
        """Fetch detection configuration."""
        self.client.get("/api/config/detection")


class AdminUser(HttpUser):
    """Simulates an admin user performing configuration changes."""

    wait_time = between(5, 15)  # Admins act less frequently
    weight = 1

    @task(3)
    def view_settings(self):
        """Admin views settings page."""
        self.client.get("/settings")

    @task(2)
    def update_detection_config(self):
        """Admin updates detection configuration."""
        config = {
            "detection_threshold": random.uniform(0.5, 0.9),
            "recognition_threshold": random.uniform(0.6, 0.8),
            "frame_skip": random.randint(1, 5),
        }
        self.client.post("/api/config/detection", json=config)

    @task(1)
    def view_system_logs(self):
        """Admin views system logs."""
        self.client.get("/api/logs?limit=100")

    @task(1)
    def trigger_system_health_check(self):
        """Admin triggers a system health check."""
        self.client.post("/api/system/health-check")


class SpikeLoadUser(HttpUser):
    """Simulates spike load scenarios (sudden traffic increase)."""

    wait_time = between(0.1, 0.5)  # Very frequent requests
    weight = 0  # Disabled by default, enable for spike testing

    @task
    def rapid_status_check(self):
        """Rapid-fire status checks."""
        self.client.get("/api/system/status")

    @task
    def rapid_event_polling(self):
        """Rapid-fire event polling."""
        self.client.get("/api/events/recent")


# Custom load shape for gradual ramp-up
from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    A load test shape that gradually increases users in steps.
    
    Step 1: 10 users for 60 seconds
    Step 2: 25 users for 60 seconds
    Step 3: 50 users for 60 seconds
    Step 4: 100 users for 60 seconds
    Step 5: Back to 25 users for 60 seconds (cool down)
    """

    step_time = 60  # Duration of each step in seconds
    step_load = 10  # Initial number of users
    spawn_rate = 2  # Users spawned per second
    time_limit = 300  # Total test duration (5 minutes)

    def tick(self):
        """Define load shape over time."""
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        if run_time < 60:
            user_count = 10
        elif run_time < 120:
            user_count = 25
        elif run_time < 180:
            user_count = 50
        elif run_time < 240:
            user_count = 100
        else:
            user_count = 25  # Cool down

        return (user_count, self.spawn_rate)


class ConstantLoadShape(LoadTestShape):
    """
    A simple constant load test.
    
    Maintains 50 users for 300 seconds (5 minutes).
    """

    def tick(self):
        run_time = self.get_run_time()

        if run_time < 300:
            return (50, 5)

        return None
