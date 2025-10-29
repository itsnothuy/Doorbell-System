#!/usr/bin/env python3
"""
Pipeline Implementation Issues Generator

Generates comprehensive GitHub issues for each phase of the Frigate-inspired
architecture implementation. Each issue includes detailed specifications,
testing requirements, and acceptance criteria.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any


class IssueTemplate:
    """Base template for GitHub issues."""
    
    def __init__(self, title: str, phase: int, pr_number: int):
        self.title = title
        self.phase = phase
        self.pr_number = pr_number
        self.labels = []
        self.assignees = []
        self.milestone = f"Phase {phase}"
        
    def generate_markdown(self) -> str:
        """Generate GitHub issue markdown."""
        return f"""
# {self.title}

{self._generate_overview()}

{self._generate_requirements()}

{self._generate_implementation()}

{self._generate_testing()}

{self._generate_acceptance_criteria()}

{self._generate_labels()}
"""
    
    def _generate_overview(self) -> str:
        """Generate overview section."""
        return """
## 📋 Overview

### Phase Information
- **Phase**: {phase}
- **PR Number**: #{pr_number}
- **Complexity**: {complexity}
- **Estimated Duration**: {duration}
- **Dependencies**: {dependencies}

### Goals
{goals}
""".format(
            phase=self.phase,
            pr_number=self.pr_number,
            complexity=self._get_complexity(),
            duration=self._get_duration(),
            dependencies=self._get_dependencies(),
            goals=self._get_goals()
        )
    
    def _generate_requirements(self) -> str:
        """Generate requirements section."""
        return """
## 🎯 Requirements

### Functional Requirements
{functional_requirements}

### Non-Functional Requirements
{non_functional_requirements}

### Performance Targets
{performance_targets}
""".format(
            functional_requirements=self._get_functional_requirements(),
            non_functional_requirements=self._get_non_functional_requirements(),
            performance_targets=self._get_performance_targets()
        )
    
    def _generate_implementation(self) -> str:
        """Generate implementation section."""
        return """
## 🔧 Implementation

### Files to Create/Modify
{files}

### Architecture Patterns
{patterns}

### Code Template
```python
{code_template}
```

### Configuration
{configuration}
""".format(
            files=self._get_files(),
            patterns=self._get_patterns(),
            code_template=self._get_code_template(),
            configuration=self._get_configuration()
        )
    
    def _generate_testing(self) -> str:
        """Generate testing section."""
        return """
## 🧪 Testing Requirements

### Unit Tests
{unit_tests}

### Integration Tests
{integration_tests}

### Performance Tests
{performance_tests}

### Test Coverage
- **Minimum Coverage**: 90%
- **Critical Path Coverage**: 100%
- **Error Handling Coverage**: 100%
""".format(
            unit_tests=self._get_unit_tests(),
            integration_tests=self._get_integration_tests(),
            performance_tests=self._get_performance_tests()
        )
    
    def _generate_acceptance_criteria(self) -> str:
        """Generate acceptance criteria section."""
        return """
## ✅ Acceptance Criteria

### Definition of Done
{definition_of_done}

### Quality Gates
{quality_gates}

### Performance Benchmarks
{performance_benchmarks}
""".format(
            definition_of_done=self._get_definition_of_done(),
            quality_gates=self._get_quality_gates(),
            performance_benchmarks=self._get_performance_benchmarks()
        )
    
    def _generate_labels(self) -> str:
        """Generate labels section."""
        return f"""
## 🏷️ Labels

{', '.join(self.labels)}
"""
    
    # Abstract methods to be implemented by subclasses
    def _get_complexity(self) -> str:
        return "Medium"
    
    def _get_duration(self) -> str:
        return "3-4 days"
    
    def _get_dependencies(self) -> str:
        return "None"
    
    def _get_goals(self) -> str:
        return "Implementation goals"
    
    def _get_functional_requirements(self) -> str:
        return "Functional requirements"
    
    def _get_non_functional_requirements(self) -> str:
        return "Non-functional requirements"
    
    def _get_performance_targets(self) -> str:
        return "Performance targets"
    
    def _get_files(self) -> str:
        return "Files to create/modify"
    
    def _get_patterns(self) -> str:
        return "Architecture patterns"
    
    def _get_code_template(self) -> str:
        return "# Code template"
    
    def _get_configuration(self) -> str:
        return "Configuration requirements"
    
    def _get_unit_tests(self) -> str:
        return "Unit test requirements"
    
    def _get_integration_tests(self) -> str:
        return "Integration test requirements"
    
    def _get_performance_tests(self) -> str:
        return "Performance test requirements"
    
    def _get_definition_of_done(self) -> str:
        return "Definition of done criteria"
    
    def _get_quality_gates(self) -> str:
        return "Quality gate requirements"
    
    def _get_performance_benchmarks(self) -> str:
        return "Performance benchmark requirements"


class FrameCaptureWorkerIssue(IssueTemplate):
    """Issue template for Frame Capture Worker implementation."""
    
    def __init__(self):
        super().__init__("Implement Frame Capture Worker with Ring Buffer", 2, 4)
        self.labels = ["enhancement", "pipeline", "hardware", "phase-2"]
    
    def _get_complexity(self) -> str:
        return "Medium"
    
    def _get_duration(self) -> str:
        return "3-5 days"
    
    def _get_dependencies(self) -> str:
        return "Camera handler refactor, Message bus system"
    
    def _get_goals(self) -> str:
        return """
- Implement high-performance frame capture with ring buffer
- Integrate GPIO event-triggered capture bursts
- Add multi-threaded capture handling
- Support platform-specific camera implementations
- Implement automatic resource management and cleanup
"""
    
    def _get_functional_requirements(self) -> str:
        return """
- Ring buffer for continuous frame capture (30-60 frames)
- GPIO event integration for doorbell press detection
- Multi-threaded capture with proper synchronization
- Platform detection (Raspberry Pi vs macOS vs Docker)
- Frame preprocessing and optimization
- Automatic camera resource management
"""
    
    def _get_non_functional_requirements(self) -> str:
        return """
- Thread safety for all operations
- Memory efficient ring buffer implementation
- Graceful error handling and recovery
- Resource cleanup on shutdown
- Comprehensive logging and monitoring
"""
    
    def _get_performance_targets(self) -> str:
        return """
- **Capture Rate**: 30 FPS on Raspberry Pi 4
- **Buffer Latency**: <100ms from trigger to first frame
- **Memory Usage**: <50MB for ring buffer
- **CPU Usage**: <20% during idle periods
- **Startup Time**: <2 seconds for camera initialization
"""
    
    def _get_files(self) -> str:
        return """
- `src/pipeline/frame_capture.py` (new)
- `src/hardware/camera_handler.py` (refactor from existing)
- `tests/test_frame_capture.py` (new)
- `tests/mocks/mock_camera.py` (new)
"""
    
    def _get_patterns(self) -> str:
        return """
- **Pipeline Worker Pattern**: Inherit from PipelineWorker base class
- **Ring Buffer Pattern**: Circular buffer for continuous capture
- **Producer-Consumer Pattern**: Camera capture and frame publishing
- **Strategy Pattern**: Platform-specific camera implementations
"""
    
    def _get_code_template(self) -> str:
        return """class FrameCaptureWorker(PipelineWorker):
    def __init__(self, camera_handler: CameraHandler, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__(message_bus, config)
        self.camera_handler = camera_handler
        self.ring_buffer = deque(maxlen=config.get('buffer_size', 30))
        self.capture_thread = None
        self.capture_lock = threading.Lock()
        
    def _setup_subscriptions(self):
        self.message_bus.subscribe('doorbell_pressed', self.handle_doorbell_event, self.worker_id)
        
    def handle_doorbell_event(self, message: Message):
        \"\"\"Handle doorbell press event and capture frame burst.\"\"\"
        try:
            with self.capture_lock:
                frames = self._capture_burst(count=5, interval=0.2)
                
                for i, frame in enumerate(frames):
                    frame_event = FrameEvent(
                        frame_data=frame,
                        sequence_number=i,
                        capture_timestamp=time.time()
                    )
                    self.message_bus.publish('frame_captured', frame_event)
                    
        except Exception as e:
            logger.error(f\"Frame capture failed: {e}\")
            self._handle_capture_error(e)"""
    
    def _get_configuration(self) -> str:
        return """
```yaml
frame_capture:
  buffer_size: 30
  capture_fps: 30
  burst_count: 5
  burst_interval: 0.2
  platform_specific:
    raspberry_pi:
      camera_module: "picamera2"
      resolution: [640, 480]
    macos:
      camera_module: "opencv"
      device_index: 0
```
"""
    
    def _get_unit_tests(self) -> str:
        return """
- Test worker initialization and configuration
- Test ring buffer operations (add, retrieve, overflow)
- Test doorbell event handling
- Test frame capture burst functionality
- Test thread safety with concurrent access
- Test error handling and recovery
- Test resource cleanup on shutdown
"""
    
    def _get_integration_tests(self) -> str:
        return """
- Test integration with camera handler
- Test message bus event publishing
- Test GPIO event integration
- Test platform-specific camera implementations
- Test end-to-end capture workflow
"""
    
    def _get_performance_tests(self) -> str:
        return """
- Benchmark capture rate on target hardware
- Test memory usage under sustained operation
- Test CPU usage during capture bursts
- Test latency from trigger to frame availability
- Load test with continuous operation
"""
    
    def _get_definition_of_done(self) -> str:
        return """
- [ ] All functional requirements implemented
- [ ] Unit tests pass with >90% coverage
- [ ] Integration tests pass on all platforms
- [ ] Performance targets met on Raspberry Pi 4
- [ ] Error handling covers all edge cases
- [ ] Documentation updated
- [ ] Code review completed
- [ ] CI/CD pipeline passes
"""
    
    def _get_quality_gates(self) -> str:
        return """
- [ ] PEP 8 compliance with type hints
- [ ] No memory leaks during 24h operation
- [ ] Thread safety verified
- [ ] Resource cleanup verified
- [ ] Platform compatibility tested
"""
    
    def _get_performance_benchmarks(self) -> str:
        return """
- [ ] 30 FPS capture rate on Raspberry Pi 4
- [ ] <100ms trigger-to-frame latency
- [ ] <50MB memory usage for ring buffer
- [ ] <20% CPU usage during idle
- [ ] <2 second camera initialization time
"""


def generate_all_issues() -> List[IssueTemplate]:
    """Generate all implementation issues."""
    issues = [
        FrameCaptureWorkerIssue(),
        # Add more issue templates here
    ]
    
    return issues


def main():
    """Generate GitHub issues for pipeline implementation."""
    print("Generating Pipeline Implementation Issues...")
    
    issues = generate_all_issues()
    
    # Create output directory
    output_dir = Path("generated_issues")
    output_dir.mkdir(exist_ok=True)
    
    for issue in issues:
        # Generate markdown
        markdown = issue.generate_markdown()
        
        # Write to file
        filename = f"PR_{issue.pr_number:02d}_{issue.title.replace(' ', '_').lower()}.md"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(markdown)
        
        print(f"Generated: {filepath}")
    
    print(f"\nGenerated {len(issues)} issues in {output_dir}/")
    print("\nNext steps:")
    print("1. Review generated issues")
    print("2. Create GitHub issues using the generated markdown")
    print("3. Assign issues to team members")
    print("4. Begin implementation phase")


if __name__ == "__main__":
    main()