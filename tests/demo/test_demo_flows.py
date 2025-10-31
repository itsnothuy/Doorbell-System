#!/usr/bin/env python3
"""
Demo Flow Tests

Tests for all demo flows to ensure they run successfully and produce
expected outputs.
"""

import pytest
from datetime import datetime
from demo.flows.initial_setup import InitialSetupFlow, demo_face_registration
from demo.flows.daily_operation import DailyOperationFlow
from demo.flows.advanced_features import AdvancedFeaturesFlow
from demo.flows.administration import AdministrationFlow
from demo.flows.troubleshooting import TroubleshootingFlow
from demo.orchestrator import DemoOrchestrator
from demo.utils.data_generator import DemoDataGenerator


class TestInitialSetupFlow:
    """Tests for initial setup flow."""
    
    def test_setup_flow_initialization(self):
        """Test that setup flow initializes correctly."""
        flow = InitialSetupFlow()
        assert len(flow.steps) == 8
        assert flow.start_time is None
        assert flow.end_time is None
    
    def test_setup_flow_run_demo(self):
        """Test running the setup flow demo."""
        flow = InitialSetupFlow()
        results = flow.run_demo(interactive=False)
        
        assert 'flow_name' in results
        assert results['total_steps'] == 8
        assert results['completed_steps'] > 0
        assert results['success_rate'] > 0
    
    def test_setup_flow_progress(self):
        """Test setup flow progress calculation."""
        flow = InitialSetupFlow()
        
        # Before running
        assert flow.get_progress() == 0.0
        
        # Mark first step as completed
        flow.steps[0].status = 'completed'
        assert flow.get_progress() == 12.5  # 1/8 = 12.5%
    
    def test_face_registration_demo(self):
        """Test face registration demonstration."""
        results = demo_face_registration()
        
        assert 'add_primary_user' in results
        assert 'add_family_members' in results
        assert 'test_recognition' in results
        assert results['total_registration_time'] > 0


class TestDailyOperationFlow:
    """Tests for daily operation flow."""
    
    def test_operation_flow_initialization(self):
        """Test that operation flow initializes correctly."""
        flow = DailyOperationFlow()
        assert len(flow.events) == 0
        assert flow.dashboard_status is not None
    
    def test_known_person_detection(self):
        """Test known person detection simulation."""
        flow = DailyOperationFlow()
        event = flow.demo_known_person_detection()
        
        assert event.event_type == 'known_person'
        assert event.person_name is not None
        assert event.confidence > 0
        assert event.total_processing_time > 0
        assert len(event.processing_pipeline) > 0
    
    def test_unknown_person_detection(self):
        """Test unknown person detection simulation."""
        flow = DailyOperationFlow()
        event = flow.demo_unknown_person_detection()
        
        assert event.event_type == 'unknown_person'
        assert event.person_name == 'Unknown'
        assert event.total_processing_time > 0
        assert len(event.notifications_sent) > 0
    
    def test_run_demo(self):
        """Test running the daily operation demo."""
        flow = DailyOperationFlow()
        results = flow.run_demo(num_events=3)
        
        assert 'total_events' in results
        assert results['total_events'] == 3
        assert 'avg_processing_time' in results


class TestAdvancedFeaturesFlow:
    """Tests for advanced features flow."""
    
    def test_advanced_flow_initialization(self):
        """Test that advanced flow initializes correctly."""
        flow = AdvancedFeaturesFlow()
        assert len(flow.patterns) > 0
        assert len(flow.anomalies) > 0
    
    def test_intelligent_analysis(self):
        """Test intelligent analysis demonstration."""
        flow = AdvancedFeaturesFlow()
        analysis = flow.demo_intelligent_analysis()
        
        assert 'weekly_analysis' in analysis
        assert 'anomaly_detection' in analysis
        assert len(analysis['weekly_analysis']['patterns_detected']) > 0
    
    def test_multi_camera_setup(self):
        """Test multi-camera setup demonstration."""
        flow = AdvancedFeaturesFlow()
        setup = flow.demo_multi_camera_setup()
        
        assert 'camera_network' in setup
        assert 'multi_camera_event' in setup
        assert setup['camera_network']['primary_camera']['status'] == 'Online'
    
    def test_run_demo(self):
        """Test running the advanced features demo."""
        flow = AdvancedFeaturesFlow()
        results = flow.run_demo()
        
        assert 'patterns_detected' in results
        assert 'cameras_configured' in results
        assert results['coordination_enabled'] is True


class TestAdministrationFlow:
    """Tests for administration flow."""
    
    def test_admin_flow_initialization(self):
        """Test that admin flow initializes correctly."""
        flow = AdministrationFlow()
        assert flow.performance_data is not None
    
    def test_performance_monitoring(self):
        """Test performance monitoring demonstration."""
        flow = AdministrationFlow()
        monitoring = flow.demo_performance_monitoring()
        
        assert 'performance_dashboard' in monitoring
        assert 'performance_trends' in monitoring
    
    def test_backup_recovery(self):
        """Test backup and recovery demonstration."""
        flow = AdministrationFlow()
        backup = flow.demo_backup_recovery()
        
        assert 'daily_backup' in backup
        assert 'backup_components' in backup
        assert 'retention_policy' in backup
    
    def test_run_demo(self):
        """Test running the administration demo."""
        flow = AdministrationFlow()
        results = flow.run_demo()
        
        assert 'system_health' in results
        assert 'uptime' in results
        assert results['monitoring_enabled'] is True


class TestTroubleshootingFlow:
    """Tests for troubleshooting flow."""
    
    def test_troubleshooting_flow_initialization(self):
        """Test that troubleshooting flow initializes correctly."""
        flow = TroubleshootingFlow()
        assert flow.diagnostic_results is not None
    
    def test_automated_diagnostics(self):
        """Test automated diagnostics demonstration."""
        flow = TroubleshootingFlow()
        diagnostics = flow.demo_automated_diagnostics()
        
        assert 'system_check' in diagnostics
        assert 'performance_analysis' in diagnostics
        assert len(diagnostics['system_check']['checks_performed']) > 0
    
    def test_common_issues(self):
        """Test common issues demonstration."""
        flow = TroubleshootingFlow()
        issues = flow.demo_common_issues()
        
        assert 'camera_issues' in issues
        assert 'recognition_issues' in issues
        assert 'performance_issues' in issues
    
    def test_remote_support(self):
        """Test remote support demonstration."""
        flow = TroubleshootingFlow()
        support = flow.demo_remote_support()
        
        assert 'support_package' in support
        assert 'remote_maintenance' in support
    
    def test_run_demo(self):
        """Test running the troubleshooting demo."""
        flow = TroubleshootingFlow()
        results = flow.run_demo()
        
        assert results['diagnostics_run'] is True
        assert results['remote_support_available'] is True


class TestDemoOrchestrator:
    """Tests for demo orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly."""
        orchestrator = DemoOrchestrator(interactive=False)
        
        assert orchestrator.setup_flow is not None
        assert orchestrator.operation_flow is not None
        assert orchestrator.advanced_flow is not None
        assert orchestrator.admin_flow is not None
        assert orchestrator.troubleshooting_flow is not None
    
    def test_quick_demo(self):
        """Test running the quick demo."""
        orchestrator = DemoOrchestrator(interactive=False)
        results = orchestrator.run_quick_demo()
        
        assert 'duration' in results
        assert 'features_shown' in results
        assert results['features_shown'] > 0
    
    @pytest.mark.slow
    def test_complete_demo(self):
        """Test running the complete demo (marked as slow)."""
        orchestrator = DemoOrchestrator(interactive=False)
        results = orchestrator.run_complete_demo()
        
        assert 'demo_metadata' in results
        assert 'sections' in results
        assert 'overall_status' in results
        assert results['overall_status'] == 'Success'


class TestDemoDataGenerator:
    """Tests for demo data generator."""
    
    def test_data_generator_initialization(self):
        """Test that data generator initializes correctly."""
        generator = DemoDataGenerator()
        assert len(generator.known_persons) > 0
    
    def test_generate_detection_event(self):
        """Test event generation."""
        generator = DemoDataGenerator()
        
        # Test random event
        event = generator.generate_detection_event('random')
        assert 'timestamp' in event
        assert 'event_type' in event
        assert event['event_type'] in ['known_person', 'unknown_person']
        
        # Test known person event
        known_event = generator.generate_detection_event('known_person')
        assert known_event['event_type'] == 'known_person'
        assert known_event['person_name'] != 'Unknown'
        
        # Test unknown person event
        unknown_event = generator.generate_detection_event('unknown_person')
        assert unknown_event['event_type'] == 'unknown_person'
        assert unknown_event['person_name'] == 'Unknown'
    
    def test_generate_performance_metrics(self):
        """Test performance metrics generation."""
        generator = DemoDataGenerator()
        metrics = generator.generate_performance_metrics()
        
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'avg_processing_time' in metrics
        assert metrics['cpu_usage'] > 0
    
    def test_generate_event_history(self):
        """Test event history generation."""
        generator = DemoDataGenerator()
        history = generator.generate_event_history(num_events=10)
        
        assert len(history) == 10
        # Check that events are sorted by timestamp
        for i in range(len(history) - 1):
            assert history[i]['timestamp'] <= history[i + 1]['timestamp']
    
    def test_generate_system_logs(self):
        """Test system log generation."""
        generator = DemoDataGenerator()
        logs = generator.generate_system_logs(num_entries=20)
        
        assert len(logs) == 20
        # Check that logs are sorted by timestamp
        for i in range(len(logs) - 1):
            assert logs[i]['timestamp'] <= logs[i + 1]['timestamp']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
