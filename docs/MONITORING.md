# Monitoring and Observability Guide

## Overview

This guide covers the comprehensive monitoring and observability stack for the Doorbell Security System in production environments.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Application Layer                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │  Pipeline  │  │    Web     │  │  Hardware  │        │
│  │Orchestrator│  │ Interface  │  │   Layer    │        │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘        │
│        │               │               │                │
│        └───────────────┼───────────────┘                │
│                        │                                │
└────────────────────────┼────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼───────┐ ┌─────▼──────┐ ┌───────▼────────┐
│   Metrics     │ │  Logging   │ │    Tracing     │
│               │ │            │ │                │
│ - Prometheus  │ │ - JSON     │ │ - OpenTelemetry│
│ - Custom      │ │ - Audit    │ │ - Jaeger       │
└───────┬───────┘ └─────┬──────┘ └───────┬────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                ┌───────▼────────┐
                │  Visualization │
                │                │
                │   - Grafana    │
                │   - Dashboards │
                └────────────────┘
```

## Metrics Collection

### Prometheus Metrics

#### System Metrics

**CPU Usage**
```promql
# Current CPU usage
system_cpu_usage_percent

# Alert on high CPU
system_cpu_usage_percent > 80
```

**Memory Usage**
```promql
# Memory usage in bytes
system_memory_usage_bytes

# Memory usage percentage
(system_memory_usage_bytes / system_memory_total_bytes) * 100
```

**Disk Usage**
```promql
# Disk usage by mount point
system_disk_usage_percent{mount_point="/"}

# Alert on low disk space
system_disk_usage_percent > 85
```

#### Pipeline Metrics

**Event Processing**
```promql
# Events processed per second
rate(pipeline_events_total[1m])

# Events by type
sum(pipeline_events_total) by (event_type)

# Success rate
sum(rate(pipeline_events_total{status="success"}[5m])) /
sum(rate(pipeline_events_total[5m]))
```

**Processing Duration**
```promql
# Average processing time
histogram_quantile(0.5, pipeline_processing_duration_seconds_bucket)

# 95th percentile
histogram_quantile(0.95, pipeline_processing_duration_seconds_bucket)

# Processing time by stage
histogram_quantile(0.95, pipeline_processing_duration_seconds_bucket) by (stage)
```

**Queue Sizes**
```promql
# Current queue sizes
pipeline_queue_size

# Queue backlog
sum(pipeline_queue_size) by (queue_name)
```

**Worker Health**
```promql
# Healthy workers
sum(pipeline_worker_health == 1)

# Unhealthy workers
sum(pipeline_worker_health == 0)

# Worker health by type
sum(pipeline_worker_health) by (worker_type)
```

#### Face Recognition Metrics

**Recognition Requests**
```promql
# Total requests
sum(face_recognition_requests_total)

# Requests per minute
rate(face_recognition_requests_total[1m]) * 60

# Requests by result type
sum(face_recognition_requests_total) by (result_type)

# High confidence recognitions
sum(face_recognition_requests_total{confidence_bucket="high"})
```

**Accuracy**
```promql
# Current accuracy
face_recognition_accuracy

# Accuracy trend
avg_over_time(face_recognition_accuracy[1h])
```

**Database Size**
```promql
# Number of known faces
known_faces_database_size

# Growth rate
rate(known_faces_database_size[1d])
```

#### Business Metrics

**Doorbell Triggers**
```promql
# Triggers per hour
rate(doorbell_triggers_total[1h]) * 3600

# Triggers by time of day
sum(doorbell_triggers_total) by (time_of_day)

# Peak hours
topk(3, sum(rate(doorbell_triggers_total[1h])) by (time_of_day))
```

**Notifications**
```promql
# Successful notifications
sum(notifications_sent_total{status="success"})

# Failed notifications
sum(notifications_sent_total{status="failure"})

# Success rate
sum(rate(notifications_sent_total{status="success"}[5m])) /
sum(rate(notifications_sent_total[5m]))

# Notifications by channel
sum(notifications_sent_total) by (channel)
```

**Active Sessions**
```promql
# Current active users
user_sessions_active

# Session trend
avg_over_time(user_sessions_active[1h])
```

#### Error Metrics

**Error Rates**
```promql
# Total errors
sum(errors_total)

# Error rate per minute
rate(errors_total[1m]) * 60

# Errors by component
sum(errors_total) by (component)

# Errors by severity
sum(errors_total) by (severity)
```

**Security Events**
```promql
# Security events count
sum(security_events_total)

# Events by type
sum(security_events_total) by (event_type)

# Recent security events
increase(security_events_total[5m])
```

### Custom Metrics Collection

```python
from monitoring.metrics.metrics_collector import MetricsCollector

# Initialize collector
collector = MetricsCollector(monitoring_system)

# Collect pipeline metrics
collector.collect_pipeline_metrics({
    "events_processed": 100,
    "event_type": "doorbell_press",
    "queue_sizes": {
        "frame_capture": 5,
        "face_detection": 3,
        "recognition": 2
    },
    "worker_health": {
        "worker_1": True,
        "worker_2": True,
        "worker_3": False
    }
})

# Collect face recognition metrics
collector.collect_face_recognition_metrics({
    "result_type": "known_person",
    "confidence": 0.95,
    "known_faces_count": 25,
    "accuracy": 98.5,
    "model_version": "1.0"
})

# Collect doorbell metrics
collector.collect_doorbell_metrics({
    "trigger_source": "button_press"
})

# Collect notification metrics
collector.collect_notification_metrics({
    "channel": "telegram",
    "type": "face_recognized",
    "status": "success"
})

# Collect error metrics
collector.collect_error_metrics({
    "component": "face_detector",
    "error_type": "timeout",
    "severity": "warning"
})

# Collect security metrics
collector.collect_security_metrics({
    "event_type": "blacklist_match",
    "source": "face_recognition",
    "action_taken": "alert_sent"
})
```

## Structured Logging

### JSON Log Format

```python
from monitoring.logging.structured_logging import get_structured_logger

logger = get_structured_logger("doorbell.pipeline")

# Set context for all subsequent logs
logger.set_context(
    request_id="req_123",
    user_id="user_456",
    session_id="sess_789"
)

# Log with additional context
logger.info("Processing event", 
    event_id="evt_001",
    event_type="doorbell_press"
)

# Log performance
logger.log_performance(
    operation="face_recognition",
    duration_ms=45.2,
    success=True,
    confidence=0.95
)

# Log HTTP requests
logger.log_request(
    method="POST",
    path="/api/recognize",
    status_code=200,
    duration_ms=120.5,
    user_agent="Mozilla/5.0"
)

# Log errors with exception
try:
    process_frame()
except Exception as e:
    logger.error("Frame processing failed", exception=e,
        frame_id="frame_123",
        retry_count=3
    )
```

### Log Output

```json
{
  "timestamp": "2025-01-29T04:20:12.950Z",
  "level": "INFO",
  "logger": "doorbell.pipeline",
  "message": "Processing event",
  "module": "event_processor",
  "function": "process_event",
  "line": 45,
  "context": {
    "request_id": "req_123",
    "user_id": "user_456",
    "session_id": "sess_789",
    "event_id": "evt_001",
    "event_type": "doorbell_press"
  }
}
```

## Audit Logging

### Security Event Logging

```python
from monitoring.logging.audit_logger import AuditLogger

audit_logger = AuditLogger()

# Log authentication
audit_logger.log_authentication(
    user="admin",
    success=True,
    method="password",
    ip_address="192.168.1.100"
)

# Log authorization
audit_logger.log_authorization(
    user="admin",
    resource="face_database",
    granted=True,
    permission="write"
)

# Log data access
audit_logger.log_data_access(
    user="system",
    resource="known_faces/alice.jpg",
    operation="read"
)

# Log data modification
audit_logger.log_data_modification(
    user="admin",
    resource="known_faces",
    operation="add",
    changes={"added": "new_person.jpg"}
)

# Log security event
audit_logger.log_security_event(
    event_type="blacklist_match",
    severity="critical",
    description="Blacklisted person detected at door",
    person_id="blacklist_123",
    confidence=0.92
)

# Log system change
audit_logger.log_system_change(
    component="pipeline",
    change_type="configuration",
    description="Updated face recognition tolerance",
    old_value=0.6,
    new_value=0.5
)

# Log face recognition
audit_logger.log_face_recognition(
    result_type="known_person",
    confidence=0.95,
    person_id="alice",
    timestamp=datetime.now()
)

# Log notification
audit_logger.log_notification(
    channel="telegram",
    recipient="admin",
    notification_type="face_recognized",
    success=True,
    message_id="msg_123"
)
```

### Audit Log Verification

```python
# Verify audit log integrity
if audit_logger.verify_integrity():
    print("Audit log integrity verified - no tampering detected")
else:
    print("WARNING: Audit log integrity check failed!")

# Verify specific range
if audit_logger.verify_integrity(start_index=0, end_index=100):
    print("First 100 entries verified")
```

## Alerting

### Alert Rules

#### Critical Alerts

```yaml
# High CPU Usage (Critical)
- alert: CriticalCPUUsage
  expr: system_cpu_usage_percent > 95
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Critical CPU usage detected"
    description: "CPU usage is {{ $value }}%"

# Worker Down
- alert: PipelineWorkerDown
  expr: pipeline_worker_health == 0
  for: 30s
  labels:
    severity: critical
  annotations:
    summary: "Pipeline worker is unhealthy"
    description: "Worker {{ $labels.worker_id }} is down"

# Security Event
- alert: SecurityEventDetected
  expr: increase(security_events_total[1m]) > 0
  labels:
    severity: critical
  annotations:
    summary: "Security event detected"
    description: "{{ $value }} security events in the last minute"
```

#### Warning Alerts

```yaml
# High Memory Usage
- alert: HighMemoryUsage
  expr: (system_memory_usage_bytes / system_memory_total_bytes) > 0.85
  for: 3m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage"
    description: "Memory usage is {{ $value | humanizePercentage }}"

# High Error Rate
- alert: HighErrorRate
  expr: rate(errors_total[5m]) > 10
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High error rate detected"
    description: "{{ $value }} errors per second"

# Low Face Recognition Accuracy
- alert: LowRecognitionAccuracy
  expr: face_recognition_accuracy < 85
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Face recognition accuracy dropped"
    description: "Accuracy is {{ $value }}%"
```

### Notification Channels

Configure notification channels in `monitoring_config.py`:

```python
# Environment variable format
# NOTIFICATION_CHANNELS="slack:webhook_url,email:admin@example.com,pagerduty:service_key"

monitoring_config = {
    "alerting": {
        "notification_channels": [
            {"type": "slack", "config": "webhook_url"},
            {"type": "email", "config": "admin@example.com"},
            {"type": "pagerduty", "config": "service_key"}
        ]
    }
}
```

## Grafana Dashboards

### System Overview Dashboard

**Panels:**
1. System Health Score (Gauge)
2. CPU Usage (Time Series)
3. Memory Usage (Time Series)
4. Disk Usage (Bar Gauge)
5. Active Workers (Stat)
6. Error Rate (Time Series)

### Pipeline Performance Dashboard

**Panels:**
1. Events Processed (Time Series)
2. Processing Duration (Heatmap)
3. Queue Sizes (Time Series)
4. Worker Health (Status History)
5. Success Rate (Gauge)
6. Throughput (Stat)

### Face Recognition Dashboard

**Panels:**
1. Recognition Requests (Time Series)
2. Accuracy Trend (Time Series)
3. Confidence Distribution (Pie Chart)
4. Result Types (Bar Chart)
5. Database Size (Stat)
6. Recognition Rate (Gauge)

### Business Metrics Dashboard

**Panels:**
1. Doorbell Triggers (Time Series)
2. Triggers by Time of Day (Heatmap)
3. Notifications Sent (Time Series)
4. Notification Success Rate (Gauge)
5. Active Sessions (Time Series)
6. Popular Times (Bar Chart)

## Distributed Tracing

### OpenTelemetry Integration

```python
from monitoring.metrics.prometheus_config import ProductionMonitoringSystem

monitoring = ProductionMonitoringSystem(config)

# Start a trace
span = monitoring.start_trace(
    "process_doorbell_event",
    attributes={
        "event_id": "evt_123",
        "event_type": "button_press"
    }
)

try:
    # Process event
    result = process_event(event_data)
    
    # End trace successfully
    monitoring.end_trace(span, status="ok")
    
except Exception as e:
    # End trace with error
    monitoring.end_trace(span, status="error", error=str(e))
```

### Jaeger Configuration

```yaml
# In docker-compose.yml
jaeger:
  image: jaegertracing/all-in-one:latest
  environment:
    - COLLECTOR_ZIPKIN_HOST_PORT=:9411
  ports:
    - "5775:5775/udp"
    - "6831:6831/udp"
    - "6832:6832/udp"
    - "5778:5778"
    - "16686:16686"
    - "14268:14268"
    - "14250:14250"
    - "9411:9411"
```

## Monitoring Best Practices

### 1. Metric Naming Conventions

- Use lowercase with underscores: `system_cpu_usage_percent`
- Include units in name: `_bytes`, `_seconds`, `_percent`
- Use consistent prefixes: `pipeline_`, `system_`, `face_`

### 2. Alert Thresholds

- Set appropriate thresholds based on baseline metrics
- Use `for` clause to avoid alert fatigue
- Implement alert cooldown periods

### 3. Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error events that might still allow application to continue
- **CRITICAL**: Critical events that require immediate attention

### 4. Metric Retention

- Short-term (1h): 15s resolution
- Medium-term (1d): 1m resolution
- Long-term (30d): 5m resolution

### 5. Dashboard Design

- Use consistent color schemes
- Group related metrics
- Include trend indicators
- Add threshold lines
- Use appropriate visualization types

## Troubleshooting

### Metrics Not Updating

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify application metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus logs
docker-compose logs prometheus
```

### High Cardinality Issues

```python
# Avoid high cardinality labels
# BAD: user_id, request_id, timestamp
# GOOD: request_type, status, component

# Use label limits
metrics_config = {
    "max_label_cardinality": 1000
}
```

### Log Volume

```python
# Use sampling for high-volume logs
if random.random() < 0.1:  # 10% sampling
    logger.debug("High-frequency event", ...)

# Use log levels appropriately
logger.debug(...)  # Development only
logger.info(...)   # Important events
logger.error(...)  # Errors only
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Structured Logging Best Practices](https://www.loggly.com/ultimate-guide/python-logging-basics/)
