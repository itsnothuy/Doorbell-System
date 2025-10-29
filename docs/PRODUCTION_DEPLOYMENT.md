# Production Deployment Guide

## Overview

This guide covers deploying the Doorbell Security System to production environments with comprehensive monitoring, zero-downtime deployments, and automated operations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Blue-Green Deployments](#blue-green-deployments)
7. [Health Checks](#health-checks)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 1.29+ (for Docker deployment)
- Kubernetes 1.20+ (for K8s deployment)
- kubectl configured (for K8s deployment)

### Docker Production Deployment

```bash
# Clone repository
git clone https://github.com/itsnothuy/Doorbell-System.git
cd Doorbell-System

# Set environment variables
export ENVIRONMENT=production
export SECRET_KEY="your-secure-secret-key"
export MONITORING_ENABLED=True

# Build and deploy
cd infrastructure/docker
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose -f docker-compose.production.yml ps
curl http://localhost:5000/health
curl http://localhost:8000/metrics  # Prometheus metrics
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace doorbell-system

# Apply configurations
kubectl apply -f infrastructure/kubernetes/deployment.yaml
kubectl apply -f infrastructure/kubernetes/service.yaml
kubectl apply -f infrastructure/kubernetes/ingress.yaml

# Verify deployment
kubectl get pods -n doorbell-system
kubectl get services -n doorbell-system
```

## Architecture

### Production Components

```
┌─────────────────────────────────────────────────────────┐
│                   Load Balancer / Ingress                │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼───────┐ ┌─────▼──────┐ ┌───────▼───────┐
│ Blue Instance │ │Green Instance│ │ Monitoring    │
│               │ │              │ │ Stack         │
│ - Application │ │- Application │ │               │
│ - Metrics     │ │- Metrics     │ │ - Prometheus  │
│ - Health      │ │- Health      │ │ - Grafana     │
└───────────────┘ └──────────────┘ └───────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                ┌────────▼────────┐
                │   Persistence   │
                │                 │
                │ - Event DB      │
                │ - Face DB       │
                │ - Logs          │
                └─────────────────┘
```

### Key Features

- **Zero-Downtime Deployments**: Blue-green deployment strategy
- **Auto-Healing**: Health checks with automatic restart
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **Structured Logging**: JSON logs with correlation IDs
- **Audit Trail**: Tamper-evident security logging
- **Auto-Scaling**: HPA for Kubernetes deployments

## Docker Deployment

### Production Docker Image

The production Dockerfile uses multi-stage builds for optimization:

```dockerfile
# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base
# ... system dependencies

# Stage 2: Python dependencies
FROM base as dependencies
# ... pip install

# Stage 3: Production image
FROM base as production
# ... application code
```

**Key Features:**
- Optimized layer caching
- Non-root user execution
- Health check integration
- Minimal image size

### Docker Compose Stack

The production stack includes:

```yaml
services:
  doorbell-system:    # Main application
  prometheus:         # Metrics collection
  grafana:           # Visualization
```

**Environment Variables:**

```bash
# Application
ENVIRONMENT=production
DEBUG=False
SECRET_KEY=your-secure-key

# Monitoring
MONITORING_ENABLED=True
PROMETHEUS_PORT=8000
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
WORKER_PROCESSES=4
WORKER_THREADS=2
MAX_QUEUE_SIZE=1000
```

### Scaling Docker Deployment

```bash
# Scale application instances
docker-compose -f docker-compose.production.yml up -d --scale doorbell-system=3

# View logs
docker-compose -f docker-compose.production.yml logs -f doorbell-system

# Restart services
docker-compose -f docker-compose.production.yml restart doorbell-system
```

## Kubernetes Deployment

### Deployment Configuration

The Kubernetes deployment includes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: doorbell-system
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

**Key Features:**
- Rolling updates with zero downtime
- Resource limits and requests
- Liveness and readiness probes
- Persistent volume claims

### Resource Management

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

### Health Checks

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 10
  periodSeconds: 5
```

### Horizontal Pod Autoscaling

```bash
# Create HPA
kubectl autoscale deployment doorbell-system \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n doorbell-system

# View HPA status
kubectl get hpa -n doorbell-system
```

## Monitoring and Observability

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`

**Key Metrics:**

- `pipeline_events_total` - Total pipeline events processed
- `pipeline_processing_duration_seconds` - Processing time histogram
- `system_cpu_usage_percent` - CPU utilization
- `system_memory_usage_bytes` - Memory usage
- `face_recognition_requests_total` - Recognition requests
- `doorbell_triggers_total` - Doorbell events
- `errors_total` - System errors

### Grafana Dashboards

Access Grafana at `http://localhost:3000`

Default credentials:
- Username: `admin`
- Password: `admin` (change on first login)

**Pre-configured Dashboards:**
- System Overview
- Pipeline Performance
- Face Recognition Metrics
- Business Metrics

### Structured Logging

Logs are output in JSON format:

```json
{
  "timestamp": "2025-01-29T04:20:12.950Z",
  "level": "INFO",
  "logger": "doorbell.pipeline",
  "message": "Event processed successfully",
  "context": {
    "event_id": "evt_123",
    "duration_ms": 45.2,
    "stage": "face_recognition"
  }
}
```

### Audit Logging

Security events are logged with tamper-evident hashing:

```json
{
  "timestamp": "2025-01-29T04:20:12.950Z",
  "event_type": "face_recognition",
  "actor": "system",
  "action": "recognize_face",
  "resource": "face_database",
  "result": "known_person",
  "hash": "a3f2..."
}
```

## Blue-Green Deployments

### Overview

Blue-green deployment enables zero-downtime releases:

1. Deploy new version to inactive environment (green)
2. Run health checks and validation
3. Switch traffic to new environment
4. Keep old environment (blue) for instant rollback

### Using the Blue-Green Deployer

```python
from production.deployment.blue_green_deployer import BlueGreenDeployer

# Initialize deployer
config = {"deployment_dir": "/opt/doorbell-system"}
deployer = BlueGreenDeployer(config)

# Deploy new version
success = deployer.deploy(
    version="1.1.0",
    source_path="/path/to/deployment/package"
)

# Check status
status = deployer.get_deployment_status()
print(f"Active environment: {status['active_environment']}")

# Rollback if needed
if not success:
    deployer.rollback()
```

### Automated Rollback

The system automatically rolls back on:
- Health check failures
- Validation failures
- High error rates (>10%)
- Performance degradation (>2x baseline)

```python
from production.deployment.rollback_manager import RollbackManager

rollback_mgr = RollbackManager(config, deployer)

# Automatic rollback monitoring
metrics = {
    "error_rate": 0.15,  # 15% error rate
    "avg_response_time": 2500  # 2.5s response time
}

if rollback_mgr.auto_rollback_if_needed("1.1.0", metrics):
    print("Automatic rollback triggered")
```

## Health Checks

### System Health Checker

```python
from production.health.health_checker import HealthChecker

health_checker = HealthChecker(config)

# Get overall health
report = health_checker.get_health_report()
print(f"Overall status: {report['overall_status']}")
print(f"Healthy checks: {report['summary']['healthy']}")

# Custom health checks
def check_database():
    # Custom check logic
    return HealthCheckResult(
        check_name="database",
        status=HealthStatus.HEALTHY,
        message="Database responding",
        timestamp=datetime.now()
    )

health_checker.register_check("database", check_database)
```

### Health Check Endpoints

**Application Health:**
```bash
curl http://localhost:5000/health
# Response:
{
  "status": "healthy",
  "checks": {
    "system_resources": "healthy",
    "application_components": "healthy",
    "database_connectivity": "healthy"
  }
}
```

**Metrics Health:**
```bash
curl http://localhost:8000/metrics
# Prometheus format metrics
```

## Security

### Security Best Practices

1. **Environment Variables**: Never commit secrets
   ```bash
   export SECRET_KEY="$(openssl rand -hex 32)"
   export DB_PASSWORD="secure-password"
   ```

2. **TLS/SSL**: Enable HTTPS in production
   ```yaml
   # In Kubernetes ingress
   tls:
   - hosts:
     - doorbell.example.com
     secretName: doorbell-tls
   ```

3. **Network Policies**: Restrict inter-pod communication
   ```bash
   kubectl apply -f network-policies.yaml
   ```

4. **RBAC**: Use role-based access control
   ```bash
   kubectl apply -f rbac.yaml
   ```

### Audit Logging

All security events are logged:

```python
from monitoring.logging.audit_logger import AuditLogger

audit_logger = AuditLogger()

# Log security event
audit_logger.log_security_event(
    event_type="unauthorized_access",
    severity="warning",
    description="Failed authentication attempt"
)

# Verify log integrity
if audit_logger.verify_integrity():
    print("Audit log integrity verified")
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker-compose logs doorbell-system

# Check health
docker-compose exec doorbell-system /health_check.sh
```

#### 2. High Memory Usage

```bash
# Check metrics
curl http://localhost:8000/metrics | grep memory

# Scale up resources
kubectl set resources deployment doorbell-system \
  --limits=memory=4Gi
```

#### 3. Deployment Failed

```bash
# Check deployment history
kubectl rollout history deployment/doorbell-system

# Rollback to previous version
kubectl rollout undo deployment/doorbell-system
```

#### 4. Metrics Not Available

```bash
# Check Prometheus connection
docker-compose exec prometheus wget -O- http://doorbell-system:8000/metrics

# Restart Prometheus
docker-compose restart prometheus
```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=True  # Only in non-production environments!

# Restart application
docker-compose restart doorbell-system
```

### Performance Profiling

```python
# Enable performance monitoring
from monitoring.metrics.metrics_collector import MetricsCollector

metrics = MetricsCollector(monitoring_system)

# Collect pipeline metrics
metrics.collect_pipeline_metrics({
    "events_processed": 100,
    "queue_sizes": {"frame_capture": 5, "face_detection": 3}
})
```

## Next Steps

- Review [Operational Runbooks](OPERATIONAL_RUNBOOKS.md)
- Configure [Alerting](ALERTING.md)
- Setup [Disaster Recovery](DISASTER_RECOVERY.md)
- Implement [Auto-Scaling](AUTO_SCALING.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/itsnothuy/Doorbell-System/issues
- Documentation: https://github.com/itsnothuy/Doorbell-System/docs

## License

MIT License - See LICENSE file for details.
