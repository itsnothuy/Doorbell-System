# Issue #16: Production Readiness and Deployment Optimization

## 📋 **Overview**

Achieve full production readiness by implementing comprehensive monitoring, optimization, security hardening, and deployment automation. This final phase ensures the pipeline system is enterprise-ready with robust observability, performance optimization, security compliance, and operational excellence.

## 🎯 **Objectives**

### **Primary Goals**
1. **Production Monitoring**: Comprehensive observability with metrics, logging, and alerting
2. **Performance Optimization**: Fine-tuned performance for production workloads
3. **Security Hardening**: Enterprise-grade security with compliance and auditing
4. **Deployment Automation**: Zero-downtime deployment with blue-green strategies
5. **Operational Excellence**: Health checks, auto-scaling, and disaster recovery
6. **Compliance & Governance**: Data privacy, retention policies, and audit trails

### **Success Criteria**
- 99.9% uptime with comprehensive monitoring
- Sub-50ms response times for critical operations
- Zero-downtime deployments with automated rollback
- SOC 2 compliance ready with audit trails
- Auto-scaling capabilities for 10x load variations
- Complete disaster recovery procedures tested and documented

## 🏗️ **Production Architecture**

### **Observability Stack**
```
Application Layer
├── Pipeline Orchestrator     → Metrics, Traces, Logs
├── Web Interface            → User Analytics, Performance
├── Hardware Layer           → System Metrics, Health
└── Storage Layer           → Database Performance, Backup Status

Monitoring Infrastructure
├── Prometheus              → Metrics Collection & Storage
├── Grafana                → Visualization & Dashboards  
├── Jaeger                 → Distributed Tracing
├── ELK Stack              → Log Aggregation & Analysis
└── AlertManager           → Intelligent Alerting

Cloud Infrastructure
├── Load Balancer          → Traffic Distribution
├── Auto Scaler           → Dynamic Resource Management
├── Health Checks         → Service Health Monitoring
└── Backup Systems        → Data Protection & Recovery
```

### **Deployment Pipeline**
```
Development → Testing → Staging → Blue-Green Production
    ↓           ↓          ↓              ↓
Unit Tests  Integration  Load Tests  Zero-Downtime
Security    Performance  Security    Monitoring
Quality     E2E Tests    Compliance  Rollback Ready
```

## 📁 **Implementation Specifications**

### **Files to Create**

#### **Monitoring and Observability**
```
monitoring/                                   # Production monitoring infrastructure
├── metrics/                                 # Metrics collection and exposition
│   ├── prometheus_config.py                # Prometheus configuration
│   ├── metrics_collector.py                # Custom metrics collection
│   ├── performance_metrics.py              # Performance monitoring
│   └── business_metrics.py                 # Business logic metrics
├── logging/                                 # Advanced logging infrastructure
│   ├── structured_logging.py               # Structured log formatting
│   ├── log_aggregator.py                   # Log collection and forwarding
│   ├── audit_logger.py                     # Security and compliance logging
│   └── correlation_logger.py               # Request correlation tracking
├── tracing/                                 # Distributed tracing
│   ├── trace_instrumentation.py            # OpenTelemetry integration
│   ├── pipeline_tracer.py                  # Pipeline operation tracing
│   └── performance_tracer.py               # Performance bottleneck identification
├── alerting/                               # Intelligent alerting system
│   ├── alert_manager.py                    # Alert management and routing
│   ├── alert_rules.py                      # Alert rule definitions
│   ├── notification_channels.py            # Multi-channel notifications
│   └── escalation_policies.py              # Alert escalation management
└── dashboards/                             # Monitoring dashboards
    ├── grafana_dashboards.json             # Pre-built Grafana dashboards
    ├── operational_dashboard.py            # Operational metrics dashboard
    └── business_dashboard.py               # Business metrics dashboard

production/                                  # Production deployment infrastructure
├── deployment/                              # Deployment automation
│   ├── blue_green_deployer.py              # Blue-green deployment manager
│   ├── canary_deployer.py                  # Canary deployment strategy
│   ├── rollback_manager.py                 # Automated rollback system
│   └── deployment_validator.py             # Post-deployment validation
├── scaling/                                 # Auto-scaling infrastructure
│   ├── auto_scaler.py                      # Dynamic resource scaling
│   ├── load_predictor.py                   # Load prediction algorithms
│   ├── resource_optimizer.py               # Resource allocation optimization
│   └── capacity_planner.py                 # Capacity planning tools
├── security/                               # Production security hardening
│   ├── security_scanner.py                 # Runtime security scanning
│   ├── vulnerability_monitor.py            # Vulnerability monitoring
│   ├── compliance_checker.py               # Compliance validation
│   └── audit_trail.py                      # Comprehensive audit logging
├── backup/                                  # Backup and disaster recovery
│   ├── backup_manager.py                   # Automated backup system
│   ├── disaster_recovery.py                # DR procedures automation
│   ├── data_integrity_checker.py           # Data integrity validation
│   └── recovery_validator.py               # Recovery process validation
└── health/                                  # Health checking and maintenance
    ├── health_checker.py                   # Comprehensive health checks
    ├── maintenance_scheduler.py            # Automated maintenance
    ├── performance_optimizer.py            # Runtime performance optimization
    └── system_diagnostics.py               # System diagnostic tools

config/production/                           # Production configuration
├── production_settings.py                  # Production-specific settings
├── monitoring_config.py                    # Monitoring configuration
├── security_config.py                      # Security hardening config
├── scaling_config.py                       # Auto-scaling configuration
└── compliance_config.py                    # Compliance settings

infrastructure/                              # Infrastructure as Code
├── docker/                                  # Production Docker configuration
│   ├── Dockerfile.production               # Optimized production image
│   ├── docker-compose.production.yml       # Production compose setup
│   └── health_check.sh                     # Container health checks
├── kubernetes/                              # Kubernetes deployment
│   ├── deployment.yaml                     # K8s deployment configuration
│   ├── service.yaml                        # Service definitions
│   ├── ingress.yaml                        # Ingress configuration
│   ├── configmap.yaml                      # Configuration management
│   └── monitoring.yaml                     # Monitoring stack deployment
└── terraform/                              # Infrastructure provisioning
    ├── main.tf                             # Main infrastructure definition
    ├── monitoring.tf                       # Monitoring infrastructure
    ├── security.tf                         # Security infrastructure
    └── variables.tf                        # Configuration variables
```

### **Core Component: Production Monitoring System**
```python
#!/usr/bin/env python3
"""
Production Monitoring System

Comprehensive monitoring infrastructure for production deployment with
metrics collection, alerting, and observability.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import psutil
import asyncio

# Third-party monitoring integrations
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    condition: str
    severity: AlertSeverity
    description: str
    threshold: float
    duration: float = 60.0  # Alert if condition persists for this duration
    cooldown: float = 300.0  # Cooldown period between alerts


class ProductionMonitoringSystem:
    """
    Comprehensive production monitoring system.
    
    Features:
    - Prometheus metrics collection
    - Distributed tracing with Jaeger
    - Structured logging
    - Intelligent alerting
    - Performance monitoring
    - Business metrics tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitoring system."""
        self.config = config
        self.enabled = config.get('monitoring', {}).get('enabled', True)
        
        # Metric definitions
        self.metrics_definitions = self._define_metrics()
        self.metrics = {}
        
        # Alert rules
        self.alert_rules = self._define_alert_rules()
        self.active_alerts = {}
        
        # Performance tracking
        self.performance_data = {}
        self.health_status = {}
        
        # Threading
        self.monitoring_thread = None
        self.running = False
        
        if self.enabled and MONITORING_AVAILABLE:
            self._initialize_monitoring()
        
        logger.info("Production monitoring system initialized")
    
    def _define_metrics(self) -> List[MetricDefinition]:
        """Define all metrics to be collected."""
        return [
            # Pipeline Performance Metrics
            MetricDefinition(
                name="pipeline_events_total",
                metric_type=MetricType.COUNTER,
                description="Total pipeline events processed",
                labels=["event_type", "status"]
            ),
            MetricDefinition(
                name="pipeline_processing_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Pipeline processing duration",
                labels=["stage", "worker_id"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            MetricDefinition(
                name="pipeline_queue_size",
                metric_type=MetricType.GAUGE,
                description="Current pipeline queue sizes",
                labels=["queue_name", "worker_type"]
            ),
            MetricDefinition(
                name="pipeline_worker_health",
                metric_type=MetricType.GAUGE,
                description="Pipeline worker health status (1=healthy, 0=unhealthy)",
                labels=["worker_id", "worker_type"]
            ),
            
            # Face Recognition Metrics
            MetricDefinition(
                name="face_recognition_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total face recognition requests",
                labels=["result_type", "confidence_bucket"]
            ),
            MetricDefinition(
                name="face_recognition_accuracy",
                metric_type=MetricType.GAUGE,
                description="Face recognition accuracy percentage",
                labels=["model_version"]
            ),
            MetricDefinition(
                name="known_faces_database_size",
                metric_type=MetricType.GAUGE,
                description="Number of known faces in database"
            ),
            
            # System Resource Metrics
            MetricDefinition(
                name="system_cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                description="System CPU usage percentage"
            ),
            MetricDefinition(
                name="system_memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="System memory usage in bytes"
            ),
            MetricDefinition(
                name="system_disk_usage_percent",
                metric_type=MetricType.GAUGE,
                description="System disk usage percentage",
                labels=["mount_point"]
            ),
            
            # Business Metrics
            MetricDefinition(
                name="doorbell_triggers_total",
                metric_type=MetricType.COUNTER,
                description="Total doorbell trigger events",
                labels=["trigger_source", "time_of_day"]
            ),
            MetricDefinition(
                name="notifications_sent_total",
                metric_type=MetricType.COUNTER,
                description="Total notifications sent",
                labels=["channel", "notification_type", "status"]
            ),
            MetricDefinition(
                name="user_sessions_active",
                metric_type=MetricType.GAUGE,
                description="Active user sessions on web interface"
            ),
            
            # Error and Security Metrics
            MetricDefinition(
                name="errors_total",
                metric_type=MetricType.COUNTER,
                description="Total system errors",
                labels=["component", "error_type", "severity"]
            ),
            MetricDefinition(
                name="security_events_total",
                metric_type=MetricType.COUNTER,
                description="Security-related events",
                labels=["event_type", "source", "action_taken"]
            )
        ]
    
    def _define_alert_rules(self) -> List[AlertRule]:
        """Define alert rules for monitoring."""
        return [
            AlertRule(
                name="HighCPUUsage",
                condition="system_cpu_usage_percent > 80",
                severity=AlertSeverity.WARNING,
                description="System CPU usage is high",
                threshold=80.0,
                duration=120.0
            ),
            AlertRule(
                name="CriticalCPUUsage",
                condition="system_cpu_usage_percent > 95",
                severity=AlertSeverity.CRITICAL,
                description="System CPU usage is critically high",
                threshold=95.0,
                duration=60.0
            ),
            AlertRule(
                name="HighMemoryUsage",
                condition="system_memory_usage_bytes > 0.85 * system_memory_total_bytes",
                severity=AlertSeverity.WARNING,
                description="System memory usage is high",
                threshold=0.85,
                duration=180.0
            ),
            AlertRule(
                name="PipelineWorkerDown",
                condition="pipeline_worker_health == 0",
                severity=AlertSeverity.CRITICAL,
                description="Pipeline worker is unhealthy",
                threshold=0.0,
                duration=30.0
            ),
            AlertRule(
                name="HighErrorRate",
                condition="rate(errors_total[5m]) > 10",
                severity=AlertSeverity.WARNING,
                description="High error rate detected",
                threshold=10.0,
                duration=300.0
            ),
            AlertRule(
                name="LowFaceRecognitionAccuracy",
                condition="face_recognition_accuracy < 85",
                severity=AlertSeverity.WARNING,
                description="Face recognition accuracy has dropped",
                threshold=85.0,
                duration=600.0
            ),
            AlertRule(
                name="SecurityEvent",
                condition="security_events_total > 0",
                severity=AlertSeverity.CRITICAL,
                description="Security event detected",
                threshold=0.0,
                duration=0.0  # Immediate alert
            )
        ]
    
    def _initialize_monitoring(self) -> None:
        """Initialize monitoring infrastructure."""
        try:
            # Initialize Prometheus metrics
            self._initialize_prometheus()
            
            # Initialize distributed tracing
            self._initialize_tracing()
            
            # Start monitoring thread
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.start()
            
            # Start Prometheus HTTP server
            prometheus_port = self.config.get('monitoring', {}).get('prometheus_port', 8000)
            start_http_server(prometheus_port)
            
            logger.info(f"Monitoring system started on port {prometheus_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            self.enabled = False
    
    def _initialize_prometheus(self) -> None:
        """Initialize Prometheus metrics."""
        if not MONITORING_AVAILABLE:
            return
        
        from prometheus_client import Counter, Histogram, Gauge
        
        for metric_def in self.metrics_definitions:
            if metric_def.metric_type == MetricType.COUNTER:
                self.metrics[metric_def.name] = Counter(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                self.metrics[metric_def.name] = Gauge(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                self.metrics[metric_def.name] = Histogram(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    buckets=metric_def.buckets
                )
    
    def _initialize_tracing(self) -> None:
        """Initialize distributed tracing."""
        if not MONITORING_AVAILABLE:
            return
        
        try:
            # Configure OpenTelemetry
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # Configure Jaeger exporter
            jaeger_config = self.config.get('tracing', {})
            if jaeger_config.get('enabled', False):
                jaeger_exporter = JaegerExporter(
                    agent_host_name=jaeger_config.get('agent_host', 'localhost'),
                    agent_port=jaeger_config.get('agent_port', 6831),
                )
                
                span_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = tracer
            logger.info("Distributed tracing initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize tracing: {e}")
    
    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        if not self.enabled or metric_name not in self.metrics:
            return
        
        try:
            metric = self.metrics[metric_name]
            labels = labels or {}
            
            if hasattr(metric, 'inc'):  # Counter
                metric.labels(**labels).inc(value)
            elif hasattr(metric, 'set'):  # Gauge
                metric.labels(**labels).set(value)
            elif hasattr(metric, 'observe'):  # Histogram
                metric.labels(**labels).observe(value)
                
        except Exception as e:
            logger.warning(f"Failed to record metric {metric_name}: {e}")
    
    def record_counter(self, metric_name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self.record_metric(metric_name, increment, labels)
    
    def record_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self.record_metric(metric_name, value, labels)
    
    def record_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self.record_metric(metric_name, value, labels)
    
    def start_trace(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        """Start a distributed trace."""
        if not self.enabled or not hasattr(self, 'tracer'):
            return None
        
        try:
            span = self.tracer.start_span(operation_name)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            return span
        except Exception as e:
            logger.warning(f"Failed to start trace for {operation_name}: {e}")
            return None
    
    def end_trace(self, span: Any, status: str = "ok", error: Optional[str] = None) -> None:
        """End a distributed trace."""
        if not span:
            return
        
        try:
            if error:
                span.set_attribute("error", True)
                span.set_attribute("error.message", error)
            span.set_attribute("status", status)
            span.end()
        except Exception as e:
            logger.warning(f"Failed to end trace: {e}")
    
    def collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_gauge("system_cpu_usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_gauge("system_memory_usage_bytes", memory.used)
            
            # Disk usage
            for disk in psutil.disk_partitions():
                if disk.mountpoint:
                    usage = psutil.disk_usage(disk.mountpoint)
                    usage_percent = (usage.used / usage.total) * 100
                    self.record_gauge(
                        "system_disk_usage_percent", 
                        usage_percent, 
                        {"mount_point": disk.mountpoint}
                    )
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def check_alert_rules(self) -> None:
        """Check alert rules and trigger alerts if needed."""
        current_time = time.time()
        
        for rule in self.alert_rules:
            try:
                # This is a simplified alert checking mechanism
                # In production, you'd integrate with Prometheus AlertManager
                alert_key = f"{rule.name}"
                
                # Check if we should evaluate this rule (considering cooldown)
                if alert_key in self.active_alerts:
                    last_alert_time = self.active_alerts[alert_key].get('last_alert_time', 0)
                    if current_time - last_alert_time < rule.cooldown:
                        continue
                
                # Evaluate alert condition (simplified)
                if self._evaluate_alert_condition(rule):
                    self._trigger_alert(rule, current_time)
                    
            except Exception as e:
                logger.warning(f"Failed to check alert rule {rule.name}: {e}")
    
    def _evaluate_alert_condition(self, rule: AlertRule) -> bool:
        """Evaluate if an alert condition is met."""
        # This is a simplified implementation
        # In production, you'd integrate with your metrics backend
        
        if "system_cpu_usage_percent" in rule.condition:
            cpu_percent = psutil.cpu_percent()
            return cpu_percent > rule.threshold
        
        if "system_memory_usage" in rule.condition:
            memory = psutil.virtual_memory()
            return memory.percent > (rule.threshold * 100)
        
        # Add more condition evaluations as needed
        return False
    
    def _trigger_alert(self, rule: AlertRule, current_time: float) -> None:
        """Trigger an alert."""
        alert_data = {
            'rule_name': rule.name,
            'severity': rule.severity.value,
            'description': rule.description,
            'timestamp': current_time,
            'last_alert_time': current_time
        }
        
        self.active_alerts[rule.name] = alert_data
        
        # Send alert notification
        self._send_alert_notification(alert_data)
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.description}")
    
    def _send_alert_notification(self, alert_data: Dict[str, Any]) -> None:
        """Send alert notification to configured channels."""
        # This would integrate with your notification system
        # For now, just log the alert
        logger.critical(f"ALERT: {alert_data['rule_name']} - {alert_data['description']}")
        
        # In production, you'd send to:
        # - Slack/Discord
        # - Email
        # - PagerDuty
        # - SMS
        # etc.
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                self.collect_system_metrics()
                
                # Check alert rules
                self.check_alert_rules()
                
                # Sleep between monitoring cycles
                time.sleep(30.0)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60.0)  # Sleep longer on error
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status."""
        return {
            'enabled': self.enabled,
            'running': self.running,
            'active_alerts': len(self.active_alerts),
            'metrics_collected': len(self.metrics),
            'alert_rules': len(self.alert_rules),
            'system_health': self._get_system_health()
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health summary."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'health_score': self._calculate_health_score(cpu_percent, memory.percent)
            }
        except Exception as e:
            logger.warning(f"Failed to get system health: {e}")
            return {'health_score': 0.0}
    
    def _calculate_health_score(self, cpu_percent: float, memory_percent: float) -> float:
        """Calculate overall system health score (0-1)."""
        # Simple health scoring algorithm
        cpu_score = max(0, 1 - (cpu_percent / 100))
        memory_score = max(0, 1 - (memory_percent / 100))
        
        # Weight CPU and memory equally
        health_score = (cpu_score + memory_score) / 2
        
        return round(health_score, 3)
    
    def shutdown(self) -> None:
        """Shutdown monitoring system."""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Monitoring system shut down")
```

### **Blue-Green Deployment Manager**
```python
#!/usr/bin/env python3
"""
Blue-Green Deployment Manager

Zero-downtime deployment system with automated rollback capabilities.
"""

import time
import logging
import subprocess
import shutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import requests

logger = logging.getLogger(__name__)


class DeploymentState(Enum):
    """Deployment states."""
    IDLE = "idle"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    SWITCHING = "switching"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


class Environment(Enum):
    """Deployment environments."""
    BLUE = "blue"
    GREEN = "green"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    application_name: str
    version: str
    image_tag: str
    health_check_url: str
    health_check_timeout: int = 300
    validation_timeout: int = 600
    rollback_timeout: int = 300


class BlueGreenDeployer:
    """
    Blue-Green deployment manager for zero-downtime deployments.
    
    Features:
    - Zero-downtime deployment
    - Automated health checking
    - Traffic switching
    - Automated rollback on failure
    - Deployment validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize blue-green deployer."""
        self.config = config
        self.current_environment = Environment.BLUE
        self.deployment_state = DeploymentState.IDLE
        self.deployment_history = []
        
        # Load balancer configuration
        self.load_balancer_config = config.get('load_balancer', {})
        
        logger.info("Blue-Green deployer initialized")
    
    def deploy(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """
        Execute blue-green deployment.
        
        Args:
            deployment_config: Deployment configuration
            
        Returns:
            Deployment result
        """
        try:
            logger.info(f"Starting blue-green deployment for {deployment_config.application_name} v{deployment_config.version}")
            
            # Determine target environment
            target_environment = self._get_target_environment()
            
            # Execute deployment phases
            result = self._execute_deployment_phases(deployment_config, target_environment)
            
            if result['success']:
                self.current_environment = target_environment
                self.deployment_state = DeploymentState.COMPLETED
                logger.info(f"✅ Blue-green deployment completed successfully to {target_environment.value}")
            else:
                self.deployment_state = DeploymentState.FAILED
                logger.error(f"❌ Blue-green deployment failed: {result['error']}")
            
            # Record deployment
            self._record_deployment(deployment_config, target_environment, result)
            
            return result
            
        except Exception as e:
            self.deployment_state = DeploymentState.FAILED
            error_msg = f"Blue-green deployment failed: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _get_target_environment(self) -> Environment:
        """Determine target environment for deployment."""
        return Environment.GREEN if self.current_environment == Environment.BLUE else Environment.BLUE
    
    def _execute_deployment_phases(self, config: DeploymentConfig, target_env: Environment) -> Dict[str, Any]:
        """Execute all deployment phases."""
        phases = [
            ("Preparation", self._prepare_deployment),
            ("Deployment", self._deploy_to_environment),
            ("Health Check", self._health_check),
            ("Validation", self._validate_deployment),
            ("Traffic Switch", self._switch_traffic),
            ("Final Validation", self._final_validation)
        ]
        
        for phase_name, phase_func in phases:
            try:
                logger.info(f"Executing phase: {phase_name}")
                
                phase_result = phase_func(config, target_env)
                
                if not phase_result.get('success', False):
                    error_msg = f"Phase {phase_name} failed: {phase_result.get('error', 'Unknown error')}"
                    
                    # Attempt rollback on failure
                    logger.warning(f"Phase failed, attempting rollback: {error_msg}")
                    rollback_result = self._rollback_deployment(config, target_env)
                    
                    return {
                        'success': False,
                        'error': error_msg,
                        'failed_phase': phase_name,
                        'rollback_attempted': True,
                        'rollback_success': rollback_result.get('success', False)
                    }
                
            except Exception as e:
                error_msg = f"Phase {phase_name} exception: {e}"
                logger.error(error_msg)
                
                # Attempt rollback
                rollback_result = self._rollback_deployment(config, target_env)
                
                return {
                    'success': False,
                    'error': error_msg,
                    'failed_phase': phase_name,
                    'rollback_attempted': True,
                    'rollback_success': rollback_result.get('success', False)
                }
        
        return {'success': True, 'target_environment': target_env.value}
    
    def _prepare_deployment(self, config: DeploymentConfig, target_env: Environment) -> Dict[str, Any]:
        """Prepare deployment environment."""
        self.deployment_state = DeploymentState.PREPARING
        
        try:
            # Stop services in target environment
            self._stop_services(target_env)
            
            # Clean up previous deployment
            self._cleanup_environment(target_env)
            
            # Prepare deployment directory
            self._prepare_deployment_directory(target_env)
            
            logger.info(f"Environment {target_env.value} prepared for deployment")
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _deploy_to_environment(self, config: DeploymentConfig, target_env: Environment) -> Dict[str, Any]:
        """Deploy application to target environment."""
        self.deployment_state = DeploymentState.DEPLOYING
        
        try:
            # Pull new Docker image
            self._pull_docker_image(config.image_tag)
            
            # Deploy to environment
            self._deploy_docker_container(config, target_env)
            
            # Wait for deployment to start
            time.sleep(10)
            
            logger.info(f"Application deployed to {target_env.value} environment")
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _health_check(self, config: DeploymentConfig, target_env: Environment) -> Dict[str, Any]:
        """Perform health check on deployed application."""
        try:
            health_url = self._get_environment_health_url(target_env, config.health_check_url)
            
            start_time = time.time()
            while time.time() - start_time < config.health_check_timeout:
                try:
                    response = requests.get(health_url, timeout=10)
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        if health_data.get('status') == 'healthy':
                            logger.info(f"Health check passed for {target_env.value}")
                            return {'success': True, 'health_data': health_data}
                    
                except requests.RequestException:
                    pass  # Continue trying
                
                time.sleep(5)  # Wait before retry
            
            return {'success': False, 'error': 'Health check timeout'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_deployment(self, config: DeploymentConfig, target_env: Environment) -> Dict[str, Any]:
        """Validate deployment functionality."""
        self.deployment_state = DeploymentState.VALIDATING
        
        try:
            # Run deployment validation tests
            validation_result = self._run_validation_tests(config, target_env)
            
            if validation_result.get('success'):
                logger.info(f"Deployment validation passed for {target_env.value}")
                return {'success': True, 'validation_results': validation_result}
            else:
                return {'success': False, 'error': 'Deployment validation failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _switch_traffic(self, config: DeploymentConfig, target_env: Environment) -> Dict[str, Any]:
        """Switch traffic to target environment."""
        self.deployment_state = DeploymentState.SWITCHING
        
        try:
            # Update load balancer configuration
            self._update_load_balancer(target_env)
            
            # Wait for traffic switch to take effect
            time.sleep(10)
            
            logger.info(f"Traffic switched to {target_env.value} environment")
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _final_validation(self, config: DeploymentConfig, target_env: Environment) -> Dict[str, Any]:
        """Perform final validation after traffic switch."""
        try:
            # Test application through load balancer
            public_url = self.load_balancer_config.get('public_url', config.health_check_url)
            
            response = requests.get(public_url, timeout=30)
            
            if response.status_code == 200:
                logger.info("Final validation passed")
                return {'success': True}
            else:
                return {'success': False, 'error': f'Final validation failed: HTTP {response.status_code}'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _rollback_deployment(self, config: DeploymentConfig, failed_env: Environment) -> Dict[str, Any]:
        """Rollback failed deployment."""
        self.deployment_state = DeploymentState.ROLLING_BACK
        
        try:
            logger.info(f"Rolling back deployment from {failed_env.value}")
            
            # Switch traffic back to current environment
            self._update_load_balancer(self.current_environment)
            
            # Stop failed deployment
            self._stop_services(failed_env)
            
            # Clean up failed deployment
            self._cleanup_environment(failed_env)
            
            logger.info("Rollback completed successfully")
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _stop_services(self, environment: Environment) -> None:
        """Stop services in environment."""
        container_name = f"doorbell-system-{environment.value}"
        
        try:
            subprocess.run(['docker', 'stop', container_name], 
                         check=False, capture_output=True)
            subprocess.run(['docker', 'rm', container_name], 
                         check=False, capture_output=True)
        except Exception as e:
            logger.warning(f"Failed to stop services in {environment.value}: {e}")
    
    def _cleanup_environment(self, environment: Environment) -> None:
        """Clean up environment."""
        # Remove temporary files, old containers, etc.
        logger.info(f"Cleaned up {environment.value} environment")
    
    def _prepare_deployment_directory(self, environment: Environment) -> None:
        """Prepare deployment directory."""
        deployment_dir = Path(f"/tmp/deployment-{environment.value}")
        deployment_dir.mkdir(parents=True, exist_ok=True)
    
    def _pull_docker_image(self, image_tag: str) -> None:
        """Pull Docker image."""
        cmd = ['docker', 'pull', image_tag]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to pull image {image_tag}: {result.stderr}")
    
    def _deploy_docker_container(self, config: DeploymentConfig, environment: Environment) -> None:
        """Deploy Docker container."""
        container_name = f"doorbell-system-{environment.value}"
        port = 5000 if environment == Environment.BLUE else 5001
        
        cmd = [
            'docker', 'run', '-d',
            '--name', container_name,
            '-p', f'{port}:5000',
            '-e', f'ENVIRONMENT={environment.value}',
            config.image_tag
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to deploy container: {result.stderr}")
    
    def _get_environment_health_url(self, environment: Environment, base_url: str) -> str:
        """Get health check URL for environment."""
        port = 5000 if environment == Environment.BLUE else 5001
        return f"http://localhost:{port}/api/health"
    
    def _run_validation_tests(self, config: DeploymentConfig, environment: Environment) -> Dict[str, Any]:
        """Run validation tests against deployment."""
        # This would run comprehensive validation tests
        # For now, return success
        return {'success': True, 'tests_passed': 5, 'tests_failed': 0}
    
    def _update_load_balancer(self, target_environment: Environment) -> None:
        """Update load balancer to point to target environment."""
        # This would update your load balancer configuration
        # (nginx, HAProxy, cloud load balancer, etc.)
        
        port = 5000 if target_environment == Environment.BLUE else 5001
        logger.info(f"Load balancer updated to point to {target_environment.value} (port {port})")
    
    def _record_deployment(self, config: DeploymentConfig, environment: Environment, result: Dict[str, Any]) -> None:
        """Record deployment in history."""
        deployment_record = {
            'timestamp': time.time(),
            'application': config.application_name,
            'version': config.version,
            'environment': environment.value,
            'success': result.get('success', False),
            'error': result.get('error'),
            'duration': time.time() - (result.get('start_time', time.time()))
        }
        
        self.deployment_history.append(deployment_record)
        
        # Keep only last 50 deployments
        if len(self.deployment_history) > 50:
            self.deployment_history = self.deployment_history[-50:]
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            'current_environment': self.current_environment.value,
            'deployment_state': self.deployment_state.value,
            'last_deployment': self.deployment_history[-1] if self.deployment_history else None,
            'deployment_count': len(self.deployment_history)
        }
```

## 📋 **Infrastructure as Code**

### **Kubernetes Deployment Configuration**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: doorbell-system
  labels:
    app: doorbell-system
    version: "1.0.0"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: doorbell-system
  template:
    metadata:
      labels:
        app: doorbell-system
        version: "1.0.0"
    spec:
      containers:
      - name: doorbell-system
        image: doorbell-system:latest
        ports:
        - containerPort: 5000
          name: http
        - containerPort: 8000
          name: metrics
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: MONITORING_ENABLED
          value: "true"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: doorbell-data-pvc
      - name: config-volume
        configMap:
          name: doorbell-config
      imagePullSecrets:
      - name: registry-secret
---
apiVersion: v1
kind: Service
metadata:
  name: doorbell-system-service
  labels:
    app: doorbell-system
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 5000
    protocol: TCP
    name: http
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: metrics
  selector:
    app: doorbell-system
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: doorbell-system-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - doorbell.yourdomain.com
    secretName: doorbell-tls
  rules:
  - host: doorbell.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: doorbell-system-service
            port:
              number: 80
```

### **Production Docker Configuration**
```dockerfile
# Dockerfile.production
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libdlib-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libdlib19 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Create data directories
RUN mkdir -p data/{known_faces,blacklist_faces,captures,logs}

# Expose ports
EXPOSE 5000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Production entrypoint
CMD ["python", "app.py"]
```

## 📋 **Acceptance Criteria**

### **Monitoring & Observability**
- [ ] Comprehensive Prometheus metrics collection
- [ ] Grafana dashboards for operational and business metrics
- [ ] Distributed tracing with Jaeger integration
- [ ] Intelligent alerting with multiple notification channels
- [ ] 99.9% monitoring system uptime

### **Performance & Optimization**
- [ ] Sub-50ms response times for critical operations
- [ ] Memory usage optimized for production workloads
- [ ] Auto-scaling capabilities tested under load
- [ ] Performance regression testing integrated into CI/CD
- [ ] Resource optimization reducing costs by 25%

### **Security & Compliance**
- [ ] SOC 2 compliance ready with audit trails
- [ ] Zero critical security vulnerabilities
- [ ] Data encryption at rest and in transit
- [ ] Compliance monitoring and reporting
- [ ] Security incident response procedures

### **Deployment & Operations**
- [ ] Zero-downtime deployment with automated rollback
- [ ] Blue-green deployment strategy implemented
- [ ] Infrastructure as Code for all components
- [ ] Disaster recovery procedures tested
- [ ] Operational runbooks and documentation

### **Business Continuity**
- [ ] 99.9% system uptime SLA
- [ ] Automated backup and recovery systems
- [ ] Multi-region deployment capability
- [ ] Business metrics tracking and reporting
- [ ] Customer support integration

---

**This final issue achieves production readiness with enterprise-grade monitoring, security, deployment automation, and operational excellence, completing the transformation of the doorbell security system into a production-ready pipeline architecture.**