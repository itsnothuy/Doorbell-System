#!/usr/bin/env python3
"""
Production Monitoring System

Comprehensive monitoring infrastructure for production deployment with
metrics collection, alerting, and observability.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Third-party monitoring integrations (optional)
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

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
        self.enabled = config.get("monitoring", {}).get("enabled", True)

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

        if self.enabled and PROMETHEUS_AVAILABLE:
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
                labels=["event_type", "status"],
            ),
            MetricDefinition(
                name="pipeline_processing_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Pipeline processing duration",
                labels=["stage", "worker_id"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            ),
            MetricDefinition(
                name="pipeline_queue_size",
                metric_type=MetricType.GAUGE,
                description="Current pipeline queue sizes",
                labels=["queue_name", "worker_type"],
            ),
            MetricDefinition(
                name="pipeline_worker_health",
                metric_type=MetricType.GAUGE,
                description="Pipeline worker health status (1=healthy, 0=unhealthy)",
                labels=["worker_id", "worker_type"],
            ),
            # Face Recognition Metrics
            MetricDefinition(
                name="face_recognition_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total face recognition requests",
                labels=["result_type", "confidence_bucket"],
            ),
            MetricDefinition(
                name="face_recognition_accuracy",
                metric_type=MetricType.GAUGE,
                description="Face recognition accuracy percentage",
                labels=["model_version"],
            ),
            MetricDefinition(
                name="known_faces_database_size",
                metric_type=MetricType.GAUGE,
                description="Number of known faces in database",
            ),
            # System Resource Metrics
            MetricDefinition(
                name="system_cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                description="System CPU usage percentage",
            ),
            MetricDefinition(
                name="system_memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="System memory usage in bytes",
            ),
            MetricDefinition(
                name="system_disk_usage_percent",
                metric_type=MetricType.GAUGE,
                description="System disk usage percentage",
                labels=["mount_point"],
            ),
            # Business Metrics
            MetricDefinition(
                name="doorbell_triggers_total",
                metric_type=MetricType.COUNTER,
                description="Total doorbell trigger events",
                labels=["trigger_source", "time_of_day"],
            ),
            MetricDefinition(
                name="notifications_sent_total",
                metric_type=MetricType.COUNTER,
                description="Total notifications sent",
                labels=["channel", "notification_type", "status"],
            ),
            MetricDefinition(
                name="user_sessions_active",
                metric_type=MetricType.GAUGE,
                description="Active user sessions on web interface",
            ),
            # Error and Security Metrics
            MetricDefinition(
                name="errors_total",
                metric_type=MetricType.COUNTER,
                description="Total system errors",
                labels=["component", "error_type", "severity"],
            ),
            MetricDefinition(
                name="security_events_total",
                metric_type=MetricType.COUNTER,
                description="Security-related events",
                labels=["event_type", "source", "action_taken"],
            ),
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
                duration=120.0,
            ),
            AlertRule(
                name="CriticalCPUUsage",
                condition="system_cpu_usage_percent > 95",
                severity=AlertSeverity.CRITICAL,
                description="System CPU usage is critically high",
                threshold=95.0,
                duration=60.0,
            ),
            AlertRule(
                name="HighMemoryUsage",
                condition="system_memory_usage_bytes > 0.85 * system_memory_total_bytes",
                severity=AlertSeverity.WARNING,
                description="System memory usage is high",
                threshold=0.85,
                duration=180.0,
            ),
            AlertRule(
                name="PipelineWorkerDown",
                condition="pipeline_worker_health == 0",
                severity=AlertSeverity.CRITICAL,
                description="Pipeline worker is unhealthy",
                threshold=0.0,
                duration=30.0,
            ),
            AlertRule(
                name="HighErrorRate",
                condition="rate(errors_total[5m]) > 10",
                severity=AlertSeverity.WARNING,
                description="High error rate detected",
                threshold=10.0,
                duration=300.0,
            ),
            AlertRule(
                name="LowFaceRecognitionAccuracy",
                condition="face_recognition_accuracy < 85",
                severity=AlertSeverity.WARNING,
                description="Face recognition accuracy has dropped",
                threshold=85.0,
                duration=600.0,
            ),
            AlertRule(
                name="SecurityEvent",
                condition="security_events_total > 0",
                severity=AlertSeverity.CRITICAL,
                description="Security event detected",
                threshold=0.0,
                duration=0.0,  # Immediate alert
            ),
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
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()

            # Start Prometheus HTTP server
            prometheus_port = self.config.get("monitoring", {}).get("prometheus_port", 8000)
            start_http_server(prometheus_port)

            logger.info(f"Monitoring system started on port {prometheus_port}")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            self.enabled = False

    def _initialize_prometheus(self) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        for metric_def in self.metrics_definitions:
            if metric_def.metric_type == MetricType.COUNTER:
                self.metrics[metric_def.name] = Counter(
                    metric_def.name, metric_def.description, metric_def.labels
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                self.metrics[metric_def.name] = Gauge(
                    metric_def.name, metric_def.description, metric_def.labels
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                self.metrics[metric_def.name] = Histogram(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    buckets=metric_def.buckets,
                )

    def _initialize_tracing(self) -> None:
        """Initialize distributed tracing."""
        if not TRACING_AVAILABLE:
            return

        try:
            # Configure OpenTelemetry
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)

            # Configure Jaeger exporter
            jaeger_config = self.config.get("tracing", {})
            if jaeger_config.get("enabled", False):
                jaeger_exporter = JaegerExporter(
                    agent_host_name=jaeger_config.get("agent_host", "localhost"),
                    agent_port=jaeger_config.get("agent_port", 6831),
                )

                span_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)

            self.tracer = tracer
            logger.info("Distributed tracing initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize tracing: {e}")

    def record_metric(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value."""
        if not self.enabled or metric_name not in self.metrics:
            return

        try:
            metric = self.metrics[metric_name]
            labels = labels or {}

            if hasattr(metric, "inc"):  # Counter
                metric.labels(**labels).inc(value)
            elif hasattr(metric, "set"):  # Gauge
                metric.labels(**labels).set(value)
            elif hasattr(metric, "observe"):  # Histogram
                metric.labels(**labels).observe(value)

        except Exception as e:
            logger.warning(f"Failed to record metric {metric_name}: {e}")

    def record_counter(
        self, metric_name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a counter metric."""
        self.record_metric(metric_name, increment, labels)

    def record_gauge(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a gauge metric."""
        self.record_metric(metric_name, value, labels)

    def record_histogram(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram metric."""
        self.record_metric(metric_name, value, labels)

    def start_trace(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        """Start a distributed trace."""
        if not self.enabled or not hasattr(self, "tracer"):
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
        if not PSUTIL_AVAILABLE:
            return

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
                    try:
                        usage = psutil.disk_usage(disk.mountpoint)
                        usage_percent = (usage.used / usage.total) * 100
                        self.record_gauge(
                            "system_disk_usage_percent",
                            usage_percent,
                            {"mount_point": disk.mountpoint},
                        )
                    except (PermissionError, OSError):
                        continue

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def check_alert_rules(self) -> None:
        """Check alert rules and trigger alerts if needed."""
        current_time = time.time()

        for rule in self.alert_rules:
            try:
                # Check if we should evaluate this rule (considering cooldown)
                alert_key = f"{rule.name}"

                if alert_key in self.active_alerts:
                    last_alert_time = self.active_alerts[alert_key].get("last_alert_time", 0)
                    if current_time - last_alert_time < rule.cooldown:
                        continue

                # Evaluate alert condition (simplified)
                if self._evaluate_alert_condition(rule):
                    self._trigger_alert(rule, current_time)

            except Exception as e:
                logger.warning(f"Failed to check alert rule {rule.name}: {e}")

    def _evaluate_alert_condition(self, rule: AlertRule) -> bool:
        """Evaluate if an alert condition is met."""
        if not PSUTIL_AVAILABLE:
            return False

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
            "rule_name": rule.name,
            "severity": rule.severity.value,
            "description": rule.description,
            "timestamp": current_time,
            "last_alert_time": current_time,
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
            "enabled": self.enabled,
            "running": self.running,
            "active_alerts": len(self.active_alerts),
            "metrics_collected": len(self.metrics),
            "alert_rules": len(self.alert_rules),
            "system_health": self._get_system_health(),
        }

    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health summary."""
        if not PSUTIL_AVAILABLE:
            return {"health_score": 0.0}

        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "health_score": self._calculate_health_score(cpu_percent, memory.percent),
            }
        except Exception as e:
            logger.warning(f"Failed to get system health: {e}")
            return {"health_score": 0.0}

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
