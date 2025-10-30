# Issue #23: End-to-End Demo Validation System

## Overview
Implement a comprehensive end-to-end demo validation system that provides automated testing of complete user scenarios, real-world simulation, and production-readiness verification for the Doorbell Security System.

## Problem Statement
While the system has comprehensive unit, integration, and performance tests, there is no automated end-to-end validation system that:
- Simulates complete real-world doorbell scenarios
- Validates user experience flows from doorbell trigger to notification delivery
- Tests production deployment configurations
- Provides confidence in system reliability for end users
- Demonstrates system capabilities in realistic conditions

## Success Criteria
- [ ] Complete doorbell scenario simulation (trigger → face detection → recognition → notification)
- [ ] Production environment deployment validation
- [ ] User experience flow testing with timing and performance validation
- [ ] Hardware simulation and mock integration testing
- [ ] Demo mode for showcasing system capabilities
- [ ] Automated scenario validation with pass/fail criteria
- [ ] Real-world condition simulation (lighting, movement, multiple faces)
- [ ] Performance validation under realistic load conditions

## Technical Requirements

### 1. Demo Orchestrator (`tests/e2e/demo_orchestrator.py`)

```python
#!/usr/bin/env python3
"""
End-to-End Demo Validation System

Comprehensive demo orchestration and validation for real-world doorbell scenarios.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class DemoScenario(Enum):
    """Demo scenario types."""
    KNOWN_PERSON_RECOGNITION = "known_person_recognition"
    UNKNOWN_PERSON_DETECTION = "unknown_person_detection"
    BLACKLIST_PERSON_ALERT = "blacklist_person_alert"
    MULTIPLE_FACES_DETECTION = "multiple_faces_detection"
    NO_FACE_SCENARIO = "no_face_scenario"
    LOW_LIGHT_CONDITIONS = "low_light_conditions"
    MOTION_WITHOUT_FACE = "motion_without_face"
    SYSTEM_OVERLOAD_TEST = "system_overload_test"
    NETWORK_INTERRUPTION = "network_interruption"
    HARDWARE_FAILURE_SIMULATION = "hardware_failure_simulation"


class DemoEnvironment(Enum):
    """Demo environment configurations."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION_LIKE = "production_like"
    EDGE_DEVICE = "edge_device"


@dataclass
class DemoConfiguration:
    """Demo execution configuration."""
    scenarios: List[DemoScenario] = field(default_factory=lambda: [DemoScenario.KNOWN_PERSON_RECOGNITION])
    environment: DemoEnvironment = DemoEnvironment.DEVELOPMENT
    test_data_path: Path = Path("tests/fixtures/demo_data")
    output_path: Path = Path("demo_results")
    timeout_seconds: int = 300
    real_time_mode: bool = False
    generate_video: bool = True
    validate_performance: bool = True
    
    # Performance expectations
    max_detection_time: float = 2.0
    max_recognition_time: float = 1.0
    max_notification_time: float = 3.0
    min_fps: float = 15.0
    
    # System configuration
    api_base_url: str = "http://localhost:5000"
    notification_endpoints: List[str] = field(default_factory=list)


@dataclass
class ScenarioStep:
    """Individual step in a demo scenario."""
    step_name: str
    action: str  # "trigger_motion", "send_image", "wait", "validate"
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_result: Optional[Dict[str, Any]] = None
    timeout: float = 30.0


@dataclass
class ScenarioResult:
    """Result of a demo scenario execution."""
    scenario: DemoScenario
    status: str  # "passed", "failed", "error"
    start_time: float
    end_time: float
    duration: float
    steps_executed: int
    steps_passed: int
    steps_failed: int
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)


@dataclass
class DemoExecutionResult:
    """Complete demo execution result."""
    configuration: DemoConfiguration
    start_time: float
    end_time: float
    total_duration: float
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    overall_status: str = "unknown"
    demo_video_path: Optional[Path] = None
    performance_report_path: Optional[Path] = None


class DemoOrchestrator:
    """End-to-end demo orchestration and validation."""
    
    def __init__(self, config: DemoConfiguration):
        self.config = config
        self.system_client = SystemClient(config.api_base_url)
        self.scenario_runner = ScenarioRunner(config)
        self.validator = DemoValidator(config)
        
        # Ensure output directory exists
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Demo orchestrator initialized with {len(config.scenarios)} scenarios")
    
    async def execute_demo(self) -> DemoExecutionResult:
        """Execute complete demo validation."""
        start_time = time.time()
        
        try:
            # Initialize system
            await self._initialize_system()
            
            # Execute scenarios
            scenario_results = []
            for scenario in self.config.scenarios:
                logger.info(f"Executing scenario: {scenario.value}")
                result = await self._execute_scenario(scenario)
                scenario_results.append(result)
                
                # Brief pause between scenarios
                if not self.config.real_time_mode:
                    await asyncio.sleep(2.0)
            
            end_time = time.time()
            
            # Generate comprehensive result
            result = DemoExecutionResult(
                configuration=self.config,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
                scenario_results=scenario_results
            )
            
            # Determine overall status
            result.overall_status = self._determine_overall_status(scenario_results)
            
            # Generate artifacts
            if self.config.generate_video:
                result.demo_video_path = await self._generate_demo_video(scenario_results)
            
            if self.config.validate_performance:
                result.performance_report_path = await self._generate_performance_report(scenario_results)
            
            return result
            
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            raise
        finally:
            await self._cleanup_system()
    
    async def _initialize_system(self) -> None:
        """Initialize system for demo execution."""
        # Wait for system to be ready
        await self.system_client.wait_for_ready(timeout=60.0)
        
        # Load demo test data
        await self._load_demo_data()
        
        # Configure system for demo mode
        await self.system_client.configure_demo_mode()
    
    async def _execute_scenario(self, scenario: DemoScenario) -> ScenarioResult:
        """Execute a specific demo scenario."""
        start_time = time.time()
        
        try:
            # Get scenario definition
            scenario_def = self._get_scenario_definition(scenario)
            
            # Execute scenario steps
            result = await self.scenario_runner.execute_scenario(scenario_def)
            
            # Validate results
            validation_result = await self.validator.validate_scenario(scenario, result)
            
            end_time = time.time()
            
            return ScenarioResult(
                scenario=scenario,
                status="passed" if validation_result["valid"] else "failed",
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                steps_executed=result["steps_executed"],
                steps_passed=result["steps_passed"],
                steps_failed=result["steps_failed"],
                performance_metrics=result["performance_metrics"]
            )
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Scenario {scenario.value} failed: {e}")
            
            return ScenarioResult(
                scenario=scenario,
                status="error",
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                steps_executed=0,
                steps_passed=0,
                steps_failed=1,
                error_message=str(e)
            )
    
    def _get_scenario_definition(self, scenario: DemoScenario) -> List[ScenarioStep]:
        """Get scenario step definitions."""
        scenarios = {
            DemoScenario.KNOWN_PERSON_RECOGNITION: [
                ScenarioStep(
                    step_name="trigger_motion",
                    action="trigger_motion",
                    parameters={"duration": 2.0},
                    expected_result={"motion_detected": True}
                ),
                ScenarioStep(
                    step_name="send_known_face",
                    action="send_image",
                    parameters={"image_path": "known_faces/john_doe.jpg"},
                    expected_result={"status": "recognized", "name": "John Doe"}
                ),
                ScenarioStep(
                    step_name="validate_notification",
                    action="validate",
                    parameters={"type": "notification", "person": "John Doe"},
                    expected_result={"notification_sent": True}
                )
            ],
            
            DemoScenario.UNKNOWN_PERSON_DETECTION: [
                ScenarioStep(
                    step_name="trigger_motion",
                    action="trigger_motion",
                    parameters={"duration": 2.0}
                ),
                ScenarioStep(
                    step_name="send_unknown_face",
                    action="send_image", 
                    parameters={"image_path": "unknown_faces/stranger.jpg"},
                    expected_result={"status": "unknown", "name": None}
                ),
                ScenarioStep(
                    step_name="validate_alert",
                    action="validate",
                    parameters={"type": "unknown_person_alert"},
                    expected_result={"alert_sent": True}
                )
            ],
            
            DemoScenario.BLACKLIST_PERSON_ALERT: [
                ScenarioStep(
                    step_name="trigger_motion",
                    action="trigger_motion",
                    parameters={"duration": 2.0}
                ),
                ScenarioStep(
                    step_name="send_blacklisted_face",
                    action="send_image",
                    parameters={"image_path": "blacklist_faces/unwanted.jpg"},
                    expected_result={"status": "blacklisted", "alert": True}
                ),
                ScenarioStep(
                    step_name="validate_security_alert",
                    action="validate",
                    parameters={"type": "security_alert", "priority": "high"},
                    expected_result={"security_alert_sent": True}
                )
            ],
            
            DemoScenario.MULTIPLE_FACES_DETECTION: [
                ScenarioStep(
                    step_name="trigger_motion",
                    action="trigger_motion",
                    parameters={"duration": 2.0}
                ),
                ScenarioStep(
                    step_name="send_multiple_faces",
                    action="send_image",
                    parameters={"image_path": "multiple_faces/group.jpg"},
                    expected_result={"faces_detected": {"min": 2, "max": 5}}
                ),
                ScenarioStep(
                    step_name="validate_group_detection",
                    action="validate",
                    parameters={"type": "multiple_faces"},
                    expected_result={"group_notification_sent": True}
                )
            ],
            
            DemoScenario.SYSTEM_OVERLOAD_TEST: [
                ScenarioStep(
                    step_name="start_load_generation",
                    action="start_load",
                    parameters={"concurrent_requests": 10, "duration": 30.0}
                ),
                ScenarioStep(
                    step_name="send_faces_under_load",
                    action="send_multiple_images",
                    parameters={"image_count": 20, "interval": 1.0}
                ),
                ScenarioStep(
                    step_name="validate_system_stability",
                    action="validate",
                    parameters={"type": "system_performance"},
                    expected_result={"response_time": {"max": 5.0}, "error_rate": {"max": 0.05}}
                )
            ]
        }
        
        return scenarios.get(scenario, [])
    
    def _determine_overall_status(self, scenario_results: List[ScenarioResult]) -> str:
        """Determine overall demo status."""
        if not scenario_results:
            return "no_scenarios"
        
        failed_count = sum(1 for r in scenario_results if r.status in ["failed", "error"])
        return "passed" if failed_count == 0 else "failed"


class SystemClient:
    """Client for interacting with the doorbell system."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30.0
    
    async def wait_for_ready(self, timeout: float = 60.0) -> None:
        """Wait for system to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/api/health")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        logger.info("System is ready")
                        return
            except Exception:
                pass
            
            await asyncio.sleep(2.0)
        
        raise TimeoutError("System did not become ready within timeout")
    
    async def configure_demo_mode(self) -> None:
        """Configure system for demo mode."""
        config = {
            "demo_mode": True,
            "notification_delay": 0.5,
            "face_recognition_timeout": 10.0,
            "motion_detection_sensitivity": 0.8
        }
        
        response = self.session.post(
            f"{self.base_url}/api/config/update",
            json=config
        )
        response.raise_for_status()
    
    async def trigger_motion(self, duration: float = 2.0) -> Dict[str, Any]:
        """Trigger motion detection."""
        response = self.session.post(
            f"{self.base_url}/api/hardware/motion/trigger",
            json={"duration": duration}
        )
        response.raise_for_status()
        return response.json()
    
    async def send_image(self, image_path: Path) -> Dict[str, Any]:
        """Send image for face recognition."""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = self.session.post(
                f"{self.base_url}/api/recognition/analyze",
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    async def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent system events."""
        response = self.session.get(
            f"{self.base_url}/api/events/recent",
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()["events"]


class ScenarioRunner:
    """Execute demo scenarios step by step."""
    
    def __init__(self, config: DemoConfiguration):
        self.config = config
        self.system_client = SystemClient(config.api_base_url)
    
    async def execute_scenario(self, steps: List[ScenarioStep]) -> Dict[str, Any]:
        """Execute scenario steps."""
        results = {
            "steps_executed": 0,
            "steps_passed": 0,
            "steps_failed": 0,
            "performance_metrics": {},
            "step_results": []
        }
        
        for step in steps:
            try:
                step_start = time.time()
                step_result = await self._execute_step(step)
                step_duration = time.time() - step_start
                
                # Record performance metrics
                results["performance_metrics"][f"{step.step_name}_duration"] = step_duration
                
                # Validate step result
                if step.expected_result:
                    validation = self._validate_step_result(step_result, step.expected_result)
                    if validation["valid"]:
                        results["steps_passed"] += 1
                    else:
                        results["steps_failed"] += 1
                        logger.warning(f"Step {step.step_name} validation failed: {validation['reason']}")
                else:
                    results["steps_passed"] += 1
                
                results["steps_executed"] += 1
                results["step_results"].append({
                    "step_name": step.step_name,
                    "duration": step_duration,
                    "result": step_result,
                    "validation": validation if step.expected_result else None
                })
                
            except Exception as e:
                logger.error(f"Step {step.step_name} failed: {e}")
                results["steps_failed"] += 1
                results["steps_executed"] += 1
                
                results["step_results"].append({
                    "step_name": step.step_name,
                    "error": str(e)
                })
        
        return results
    
    async def _execute_step(self, step: ScenarioStep) -> Dict[str, Any]:
        """Execute individual scenario step."""
        if step.action == "trigger_motion":
            return await self.system_client.trigger_motion(
                duration=step.parameters.get("duration", 2.0)
            )
        
        elif step.action == "send_image":
            image_path = self.config.test_data_path / step.parameters["image_path"]
            return await self.system_client.send_image(image_path)
        
        elif step.action == "wait":
            await asyncio.sleep(step.parameters.get("duration", 1.0))
            return {"action": "wait", "completed": True}
        
        elif step.action == "validate":
            # Validation steps are handled by the validator
            return {"action": "validate", "type": step.parameters.get("type")}
        
        else:
            raise ValueError(f"Unknown step action: {step.action}")
    
    def _validate_step_result(self, result: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate step result against expected outcome."""
        validation = {"valid": True, "reasons": []}
        
        for key, expected_value in expected.items():
            if key not in result:
                validation["valid"] = False
                validation["reasons"].append(f"Missing key: {key}")
                continue
            
            actual_value = result[key]
            
            # Handle different validation types
            if isinstance(expected_value, dict):
                # Range validation (min/max)
                if "min" in expected_value or "max" in expected_value:
                    if "min" in expected_value and actual_value < expected_value["min"]:
                        validation["valid"] = False
                        validation["reasons"].append(f"{key} below minimum: {actual_value} < {expected_value['min']}")
                    if "max" in expected_value and actual_value > expected_value["max"]:
                        validation["valid"] = False
                        validation["reasons"].append(f"{key} above maximum: {actual_value} > {expected_value['max']}")
            else:
                # Direct value comparison
                if actual_value != expected_value:
                    validation["valid"] = False
                    validation["reasons"].append(f"{key} mismatch: {actual_value} != {expected_value}")
        
        if not validation["valid"]:
            validation["reason"] = "; ".join(validation["reasons"])
        
        return validation


class DemoValidator:
    """Validate demo execution results."""
    
    def __init__(self, config: DemoConfiguration):
        self.config = config
        self.system_client = SystemClient(config.api_base_url)
    
    async def validate_scenario(self, scenario: DemoScenario, 
                              execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete scenario execution."""
        validation = {"valid": True, "reasons": []}
        
        # Check basic execution success
        if execution_result["steps_failed"] > 0:
            validation["valid"] = False
            validation["reasons"].append(f"Steps failed: {execution_result['steps_failed']}")
        
        # Validate performance requirements
        performance_validation = await self._validate_performance(execution_result["performance_metrics"])
        if not performance_validation["valid"]:
            validation["valid"] = False
            validation["reasons"].extend(performance_validation["reasons"])
        
        # Scenario-specific validations
        scenario_validation = await self._validate_scenario_specific(scenario, execution_result)
        if not scenario_validation["valid"]:
            validation["valid"] = False
            validation["reasons"].extend(scenario_validation["reasons"])
        
        if not validation["valid"]:
            validation["reason"] = "; ".join(validation["reasons"])
        
        return validation
    
    async def _validate_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate performance metrics."""
        validation = {"valid": True, "reasons": []}
        
        # Check detection time
        detection_times = [v for k, v in metrics.items() if "detection" in k or "recognition" in k]
        if detection_times:
            max_detection_time = max(detection_times)
            if max_detection_time > self.config.max_detection_time:
                validation["valid"] = False
                validation["reasons"].append(
                    f"Detection time too slow: {max_detection_time:.2f}s > {self.config.max_detection_time}s"
                )
        
        return validation
    
    async def _validate_scenario_specific(self, scenario: DemoScenario, 
                                        execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform scenario-specific validation."""
        validation = {"valid": True, "reasons": []}
        
        # Get recent events to validate system behavior
        try:
            events = await self.system_client.get_recent_events(limit=20)
            
            if scenario == DemoScenario.KNOWN_PERSON_RECOGNITION:
                # Should have face recognition event
                recognition_events = [e for e in events if e.get("event_type") == "face_recognized"]
                if not recognition_events:
                    validation["valid"] = False
                    validation["reasons"].append("No face recognition event found")
            
            elif scenario == DemoScenario.BLACKLIST_PERSON_ALERT:
                # Should have security alert
                alert_events = [e for e in events if e.get("event_type") == "security_alert"]
                if not alert_events:
                    validation["valid"] = False
                    validation["reasons"].append("No security alert event found")
            
        except Exception as e:
            logger.warning(f"Could not validate events: {e}")
        
        return validation
```

### 2. Demo Data Manager (`tests/e2e/demo_data.py`)

```python
#!/usr/bin/env python3
"""
Demo Data Management

Management of test images, scenarios, and demo assets for end-to-end validation.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DemoDataManager:
    """Manage demo test data and assets."""
    
    def __init__(self, data_root: Path = Path("tests/fixtures/demo_data")):
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Setup data structure
        self.known_faces_dir = self.data_root / "known_faces"
        self.unknown_faces_dir = self.data_root / "unknown_faces"
        self.blacklist_faces_dir = self.data_root / "blacklist_faces"
        self.multiple_faces_dir = self.data_root / "multiple_faces"
        self.scenarios_dir = self.data_root / "scenarios"
        
        for dir_path in [self.known_faces_dir, self.unknown_faces_dir, 
                        self.blacklist_faces_dir, self.multiple_faces_dir, self.scenarios_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def generate_test_images(self) -> None:
        """Generate synthetic test images for demo scenarios."""
        # Generate known person images
        self._generate_person_images(
            self.known_faces_dir,
            [
                {"name": "john_doe", "count": 5},
                {"name": "jane_smith", "count": 5},
                {"name": "bob_wilson", "count": 3}
            ]
        )
        
        # Generate unknown person images
        self._generate_person_images(
            self.unknown_faces_dir,
            [
                {"name": "stranger_1", "count": 3},
                {"name": "stranger_2", "count": 3},
                {"name": "delivery_person", "count": 2}
            ]
        )
        
        # Generate blacklist person images
        self._generate_person_images(
            self.blacklist_faces_dir,
            [
                {"name": "unwanted_person", "count": 4},
                {"name": "suspicious_individual", "count": 3}
            ]
        )
        
        # Generate multiple faces scenarios
        self._generate_group_images()
        
        # Generate scenario variations
        self._generate_scenario_images()
    
    def _generate_person_images(self, output_dir: Path, persons: List[Dict[str, Any]]) -> None:
        """Generate person images with variations."""
        for person in persons:
            name = person["name"]
            count = person["count"]
            
            for i in range(count):
                # Create base image (simulated face)
                img = self._create_face_image(name, variation=i)
                
                # Save image
                image_path = output_dir / f"{name}_{i+1:02d}.jpg"
                cv2.imwrite(str(image_path), img)
    
    def _create_face_image(self, person_name: str, variation: int = 0) -> np.ndarray:
        """Create synthetic face image for testing."""
        # Create 640x480 image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background gradient
        for y in range(480):
            intensity = int(50 + (y / 480) * 100)
            img[y, :] = [intensity, intensity, intensity]
        
        # Add simple face-like features (rectangle for demonstration)
        face_color = (200, 180, 160)  # Skin-like color
        
        # Face outline
        cv2.rectangle(img, (220, 120), (420, 360), face_color, -1)
        
        # Eyes
        cv2.circle(img, (270, 200), 15, (50, 50, 50), -1)
        cv2.circle(img, (370, 200), 15, (50, 50, 50), -1)
        
        # Nose
        cv2.line(img, (320, 220), (320, 280), (100, 100, 100), 3)
        
        # Mouth
        cv2.ellipse(img, (320, 320), (40, 20), 0, 0, 180, (100, 50, 50), 3)
        
        # Add person name text
        cv2.putText(img, person_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add variation (slight position changes)
        if variation > 0:
            # Shift face slightly
            shift_x = (variation - 2) * 10
            shift_y = (variation - 2) * 5
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (640, 480))
        
        return img
    
    def _generate_group_images(self) -> None:
        """Generate images with multiple faces."""
        scenarios = [
            {"name": "family_group", "faces": ["john_doe", "jane_smith", "child"]},
            {"name": "delivery_team", "faces": ["delivery_1", "delivery_2"]},
            {"name": "visitors_group", "faces": ["visitor_1", "visitor_2", "visitor_3"]}
        ]
        
        for scenario in scenarios:
            img = self._create_group_image(scenario["faces"])
            image_path = self.multiple_faces_dir / f"{scenario['name']}.jpg"
            cv2.imwrite(str(image_path), img)
    
    def _create_group_image(self, face_names: List[str]) -> np.ndarray:
        """Create image with multiple faces."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background
        img[:] = [80, 80, 120]  # Darker background
        
        # Position faces side by side
        face_width = 120
        face_height = 160
        start_x = (640 - len(face_names) * face_width) // 2
        
        for i, name in enumerate(face_names):
            x = start_x + i * face_width
            y = 160
            
            # Draw simple face
            face_color = (200, 180, 160)
            cv2.rectangle(img, (x, y), (x + face_width - 20, y + face_height), face_color, -1)
            
            # Eyes
            cv2.circle(img, (x + 30, y + 40), 8, (50, 50, 50), -1)
            cv2.circle(img, (x + 70, y + 40), 8, (50, 50, 50), -1)
            
            # Add name label
            cv2.putText(img, name[:8], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def _generate_scenario_images(self) -> None:
        """Generate scenario-specific test images."""
        scenarios = {
            "low_light": self._create_low_light_image(),
            "motion_only": self._create_motion_only_image(),
            "no_face": self._create_no_face_image(),
            "blurry_face": self._create_blurry_face_image()
        }
        
        for scenario_name, img in scenarios.items():
            image_path = self.scenarios_dir / f"{scenario_name}.jpg"
            cv2.imwrite(str(image_path), img)
    
    def _create_low_light_image(self) -> np.ndarray:
        """Create low-light test image."""
        img = self._create_face_image("low_light_person")
        # Darken the image
        img = (img * 0.3).astype(np.uint8)
        return img
    
    def _create_motion_only_image(self) -> np.ndarray:
        """Create image with motion but no clear face."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = [40, 60, 80]  # Dark background
        
        # Add motion blur effect
        kernel = np.ones((5, 15), np.float32) / 75
        img = cv2.filter2D(img, -1, kernel)
        
        # Add some moving objects
        cv2.rectangle(img, (100, 200), (200, 300), (100, 100, 100), -1)
        cv2.rectangle(img, (400, 150), (500, 250), (120, 120, 120), -1)
        
        return img
    
    def _create_no_face_image(self) -> np.ndarray:
        """Create image with no faces."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = [60, 120, 60]  # Green background
        
        # Add some objects but no faces
        cv2.rectangle(img, (200, 200), (440, 280), (139, 69, 19), -1)  # Package
        cv2.putText(img, "DELIVERY", (250, 245), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img
    
    def _create_blurry_face_image(self) -> np.ndarray:
        """Create blurry face image for quality testing."""
        img = self._create_face_image("blurry_person")
        # Apply blur
        img = cv2.GaussianBlur(img, (15, 15), 0)
        return img
    
    def create_demo_metadata(self) -> None:
        """Create metadata files for demo scenarios."""
        metadata = {
            "known_persons": [
                {
                    "name": "John Doe",
                    "images": ["john_doe_01.jpg", "john_doe_02.jpg", "john_doe_03.jpg", "john_doe_04.jpg", "john_doe_05.jpg"],
                    "description": "Authorized resident"
                },
                {
                    "name": "Jane Smith", 
                    "images": ["jane_smith_01.jpg", "jane_smith_02.jpg", "jane_smith_03.jpg", "jane_smith_04.jpg", "jane_smith_05.jpg"],
                    "description": "Authorized resident"
                },
                {
                    "name": "Bob Wilson",
                    "images": ["bob_wilson_01.jpg", "bob_wilson_02.jpg", "bob_wilson_03.jpg"],
                    "description": "Authorized visitor"
                }
            ],
            "blacklist_persons": [
                {
                    "name": "Unwanted Person",
                    "images": ["unwanted_person_01.jpg", "unwanted_person_02.jpg", "unwanted_person_03.jpg", "unwanted_person_04.jpg"],
                    "description": "Security risk - alert immediately"
                },
                {
                    "name": "Suspicious Individual",
                    "images": ["suspicious_individual_01.jpg", "suspicious_individual_02.jpg", "suspicious_individual_03.jpg"],
                    "description": "Previous incident - high alert"
                }
            ],
            "demo_scenarios": [
                {
                    "scenario": "known_person_recognition",
                    "description": "Authorized person arrives at door",
                    "expected_outcome": "Recognition and friendly notification",
                    "test_images": ["john_doe_03.jpg", "jane_smith_02.jpg"]
                },
                {
                    "scenario": "unknown_person_detection",
                    "description": "Stranger appears at door",
                    "expected_outcome": "Unknown person alert sent",
                    "test_images": ["stranger_1_01.jpg", "delivery_person_01.jpg"]
                },
                {
                    "scenario": "blacklist_person_alert",
                    "description": "Blacklisted person detected",
                    "expected_outcome": "Immediate security alert",
                    "test_images": ["unwanted_person_02.jpg", "suspicious_individual_01.jpg"]
                }
            ]
        }
        
        metadata_path = self.data_root / "demo_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
    
    def setup_demo_environment(self, target_system_dir: Path) -> None:
        """Setup demo data in target system."""
        system_data_dir = target_system_dir / "data"
        
        # Copy known faces
        known_faces_target = system_data_dir / "known_faces"
        known_faces_target.mkdir(parents=True, exist_ok=True)
        
        # Copy a subset of known faces to system
        for image_file in self.known_faces_dir.glob("*.jpg"):
            if any(name in image_file.name for name in ["john_doe_01", "jane_smith_01", "bob_wilson_01"]):
                shutil.copy2(image_file, known_faces_target)
        
        # Copy blacklist faces
        blacklist_faces_target = system_data_dir / "blacklist_faces"
        blacklist_faces_target.mkdir(parents=True, exist_ok=True)
        
        for image_file in self.blacklist_faces_dir.glob("*.jpg"):
            if any(name in image_file.name for name in ["unwanted_person_01", "suspicious_individual_01"]):
                shutil.copy2(image_file, blacklist_faces_target)
```

### 3. Demo Video Generator (`tests/e2e/demo_video.py`)

```python
#!/usr/bin/env python3
"""
Demo Video Generation

Generate demonstration videos showing system capabilities and test results.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import time


class DemoVideoGenerator:
    """Generate demo videos from test scenarios."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Video settings
        self.fps = 30
        self.frame_size = (1280, 720)
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    def generate_scenario_video(self, scenario_results: List[Dict[str, Any]]) -> Path:
        """Generate video showing scenario execution."""
        video_path = self.output_path / "demo_scenarios.mp4"
        
        writer = cv2.VideoWriter(
            str(video_path),
            self.codec,
            self.fps,
            self.frame_size
        )
        
        try:
            # Title frame
            self._add_title_frame(writer, "Doorbell Security System Demo")
            
            # Add scenario frames
            for i, scenario_result in enumerate(scenario_results):
                self._add_scenario_frame(writer, scenario_result, i + 1)
                self._add_scenario_execution(writer, scenario_result)
            
            # Summary frame
            self._add_summary_frame(writer, scenario_results)
            
        finally:
            writer.release()
        
        return video_path
    
    def _add_title_frame(self, writer: cv2.VideoWriter, title: str) -> None:
        """Add title frame to video."""
        frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        frame[:] = [30, 30, 30]  # Dark background
        
        # Add title
        cv2.putText(frame, title, (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(frame, "AI-Powered Face Recognition", (350, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Generated: {timestamp}", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
        
        # Display for 3 seconds
        for _ in range(self.fps * 3):
            writer.write(frame)
    
    def _add_scenario_frame(self, writer: cv2.VideoWriter, scenario_result: Dict[str, Any], scenario_num: int) -> None:
        """Add scenario introduction frame."""
        frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        frame[:] = [40, 60, 40]  # Dark green background
        
        # Scenario title
        scenario_name = scenario_result["scenario"].replace("_", " ").title()
        cv2.putText(frame, f"Scenario {scenario_num}: {scenario_name}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Status
        status = scenario_result["status"]
        status_color = (0, 255, 0) if status == "passed" else (0, 0, 255)
        cv2.putText(frame, f"Status: {status.upper()}", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Duration
        duration = scenario_result["duration"]
        cv2.putText(frame, f"Duration: {duration:.2f}s", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Steps
        steps_text = f"Steps: {scenario_result['steps_passed']}/{scenario_result['steps_executed']} passed"
        cv2.putText(frame, steps_text, (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Display for 2 seconds
        for _ in range(self.fps * 2):
            writer.write(frame)
    
    def _add_scenario_execution(self, writer: cv2.VideoWriter, scenario_result: Dict[str, Any]) -> None:
        """Add scenario execution visualization."""
        # This would show actual test execution if available
        # For now, create a simple progress visualization
        
        for step_i in range(scenario_result.get('steps_executed', 3)):
            frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
            frame[:] = [20, 20, 40]  # Dark blue background
            
            # Progress bar
            progress = (step_i + 1) / scenario_result.get('steps_executed', 3)
            bar_width = int(800 * progress)
            cv2.rectangle(frame, (240, 300), (240 + bar_width, 350), (0, 255, 0), -1)
            cv2.rectangle(frame, (240, 300), (1040, 350), (100, 100, 100), 2)
            
            # Step text
            cv2.putText(frame, f"Executing Step {step_i + 1}...", (240, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Progress: {progress:.0%}", (240, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            
            # Display for 1 second per step
            for _ in range(self.fps):
                writer.write(frame)
    
    def _add_summary_frame(self, writer: cv2.VideoWriter, scenario_results: List[Dict[str, Any]]) -> None:
        """Add summary frame to video."""
        frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        frame[:] = [20, 40, 20]  # Dark green background
        
        # Title
        cv2.putText(frame, "Demo Summary", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Statistics
        total_scenarios = len(scenario_results)
        passed_scenarios = sum(1 for r in scenario_results if r["status"] == "passed")
        total_duration = sum(r["duration"] for r in scenario_results)
        
        y_pos = 200
        cv2.putText(frame, f"Total Scenarios: {total_scenarios}", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_pos += 60
        cv2.putText(frame, f"Passed: {passed_scenarios}", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_pos += 60
        cv2.putText(frame, f"Failed: {total_scenarios - passed_scenarios}", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_pos += 60
        cv2.putText(frame, f"Total Duration: {total_duration:.2f}s", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Overall status
        overall_status = "PASSED" if passed_scenarios == total_scenarios else "FAILED"
        status_color = (0, 255, 0) if overall_status == "PASSED" else (0, 0, 255)
        cv2.putText(frame, f"Overall: {overall_status}", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        
        # Display for 5 seconds
        for _ in range(self.fps * 5):
            writer.write(frame)
```

## Acceptance Criteria

### Core Demo Scenarios
- [ ] **Known Person Recognition**: Complete flow from motion trigger to notification delivery
- [ ] **Unknown Person Detection**: Proper unknown person handling and alert generation
- [ ] **Blacklist Person Alert**: Security alert generation for blacklisted individuals
- [ ] **Multiple Faces Detection**: Group detection and notification scenarios
- [ ] **Edge Cases**: No face, low light, motion-only scenarios

### Performance Validation
- [ ] **Response Times**: Face detection < 2s, recognition < 1s, notification < 3s
- [ ] **System Stability**: <5% error rate under normal load conditions
- [ ] **Resource Usage**: Memory and CPU usage within acceptable limits
- [ ] **Concurrent Handling**: Multiple simultaneous requests processing

### Production Readiness
- [ ] **Environment Testing**: Development, staging, and production-like environments
- [ ] **Hardware Simulation**: Mock hardware integration testing
- [ ] **Error Handling**: Graceful degradation under failure conditions
- [ ] **Configuration Validation**: System configuration verification

### Documentation & Reporting
- [ ] **Demo Video Generation**: Automated video creation showing system capabilities
- [ ] **Performance Reports**: Detailed performance metrics and analysis
- [ ] **Scenario Documentation**: Clear scenario descriptions and expected outcomes
- [ ] **Troubleshooting Guide**: Common issues and resolution steps

## Implementation Plan

### Phase 1: Core Demo Framework (Week 1)
1. Implement `DemoOrchestrator` with basic scenario execution
2. Create `SystemClient` for API interaction
3. Add `ScenarioRunner` for step-by-step execution
4. Implement basic validation framework

### Phase 2: Demo Data & Scenarios (Week 2)  
1. Implement `DemoDataManager` for test data generation
2. Create comprehensive demo scenarios and test images
3. Add scenario-specific validation logic
4. Implement demo environment setup

### Phase 3: Performance & Validation (Week 3)
1. Add performance validation and regression testing
2. Implement comprehensive scenario validation
3. Add system stability and load testing
4. Create production environment validation

### Phase 4: Reporting & Integration (Week 4)
1. Implement `DemoVideoGenerator` for visual demonstrations
2. Add comprehensive reporting and metrics
3. CI/CD integration for automated demo execution
4. Documentation and troubleshooting guides

## Testing Strategy

### Unit Tests
- Demo orchestrator configuration and execution
- Scenario runner step execution and validation
- System client API interaction
- Data manager test image generation

### Integration Tests
- Complete scenario execution with real system
- Performance validation accuracy
- Cross-environment compatibility
- Error handling and recovery

### End-to-End Tests
- Full demo suite execution
- Production environment validation
- Performance regression detection
- Video generation and reporting

## Dependencies

### Required Packages
```txt
opencv-python>=4.8.0
Pillow>=10.0.0
requests>=2.31.0
aiohttp>=3.8.0
pytest-asyncio>=0.21.0
```

### System Requirements
- OpenCV for video processing
- Sufficient storage for demo videos and images
- Network access to system API endpoints
- Camera simulation capabilities for hardware testing

## Security Considerations

- **Test Data Security**: Synthetic test images to avoid privacy concerns
- **API Security**: Secure communication with system endpoints
- **Environment Isolation**: Proper test environment isolation
- **Data Cleanup**: Automatic cleanup of test data and artifacts

## Performance Considerations

- **Video Generation**: Efficient video encoding and processing
- **Concurrent Testing**: Parallel scenario execution optimization
- **Resource Monitoring**: System resource usage during demo execution
- **Caching**: Test data and environment caching for faster execution

---

**Estimated Effort**: 4 weeks (160 hours)
**Priority**: High (Required for production confidence and system demonstration)
**Dependencies**: Issue #22 (Comprehensive Testing Framework)