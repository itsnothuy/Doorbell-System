# Production Readiness Quick Reference

## Installation

```bash
# Install with production dependencies
pip install -e ".[production]"

# Or with all dependencies
pip install -e ".[all]"

# Or from requirements file
pip install -r requirements-production.txt
```

## Quick Start

### Docker Deployment

```bash
cd infrastructure/docker
docker-compose -f docker-compose.production.yml up -d
```

**Access:**
- Application: http://localhost:5000
- Metrics: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Kubernetes Deployment

```bash
kubectl apply -f infrastructure/kubernetes/
kubectl get pods
```

## Monitoring

### Key Metrics Endpoints

```bash
# Application metrics (Prometheus format)
curl http://localhost:8000/metrics

# Health check
curl http://localhost:5000/health

# Deployment status
curl http://localhost:5000/api/deployment/status
```

### Essential Queries

```promql
# CPU Usage
system_cpu_usage_percent

# Memory Usage  
system_memory_usage_bytes

# Event Processing Rate
rate(pipeline_events_total[1m])

# Error Rate
rate(errors_total[5m])
```

## Production Configuration

### Environment Variables

```bash
# Application
export ENVIRONMENT=production
export DEBUG=False
export SECRET_KEY="your-secure-key"

# Monitoring
export MONITORING_ENABLED=True
export PROMETHEUS_PORT=8000
export LOG_LEVEL=INFO
export LOG_FORMAT=json

# Performance
export WORKER_PROCESSES=4
export WORKER_THREADS=2
export MAX_QUEUE_SIZE=1000
```

### Configuration Files

- `config/production/production_settings.py` - Production settings
- `config/production/monitoring_config.py` - Monitoring configuration
- `infrastructure/docker/prometheus.yml` - Prometheus configuration
- `infrastructure/kubernetes/deployment.yaml` - K8s deployment

## Deployment

### Blue-Green Deployment

```python
from production.deployment.blue_green_deployer import BlueGreenDeployer

deployer = BlueGreenDeployer({"deployment_dir": "/opt/doorbell-system"})
deployer.deploy("1.1.0", "/path/to/package")
```

### Rollback

```python
from production.deployment.rollback_manager import RollbackManager

rollback_mgr = RollbackManager(config, deployer)
rollback_mgr.trigger_rollback(
    RollbackReason.VALIDATION_FAILED,
    current_version="1.1.0"
)
```

## Health Checks

### System Health

```python
from production.health.health_checker import HealthChecker

checker = HealthChecker(config)
report = checker.get_health_report()
print(f"Status: {report['overall_status']}")
```

### Custom Health Checks

```python
def check_database():
    return HealthCheckResult(
        check_name="database",
        status=HealthStatus.HEALTHY,
        message="OK",
        timestamp=datetime.now()
    )

checker.register_check("database", check_database)
```

## Logging

### Structured Logging

```python
from monitoring.logging.structured_logging import get_structured_logger

logger = get_structured_logger("app")
logger.set_context(request_id="req_123")
logger.info("Event processed", event_id="evt_456")
```

### Audit Logging

```python
from monitoring.logging.audit_logger import AuditLogger

audit = AuditLogger()
audit.log_face_recognition(
    result_type="known_person",
    confidence=0.95,
    person_id="alice"
)
```

## Metrics Collection

### Application Metrics

```python
from monitoring.metrics.metrics_collector import MetricsCollector

collector = MetricsCollector(monitoring_system)

# Pipeline metrics
collector.collect_pipeline_metrics({
    "events_processed": 100,
    "event_type": "doorbell_press"
})

# Face recognition metrics
collector.collect_face_recognition_metrics({
    "result_type": "known_person",
    "confidence": 0.95
})

# Error metrics
collector.collect_error_metrics({
    "component": "pipeline",
    "error_type": "timeout",
    "severity": "warning"
})
```

## Troubleshooting

### Check Logs

```bash
# Docker
docker-compose logs -f doorbell-system

# Kubernetes
kubectl logs -f deployment/doorbell-system
```

### Check Metrics

```bash
# System metrics
curl http://localhost:8000/metrics | grep system_

# Pipeline metrics
curl http://localhost:8000/metrics | grep pipeline_

# Error metrics
curl http://localhost:8000/metrics | grep errors_
```

### Verify Health

```bash
# Application health
curl http://localhost:5000/health

# Container health (Docker)
docker-compose exec doorbell-system /health_check.sh

# Pod health (Kubernetes)
kubectl describe pod <pod-name>
```

### Restart Services

```bash
# Docker
docker-compose restart doorbell-system

# Kubernetes
kubectl rollout restart deployment/doorbell-system
```

## Common Issues

### High CPU Usage

```bash
# Check current usage
curl http://localhost:8000/metrics | grep cpu_usage_percent

# Scale up (Docker)
docker-compose up -d --scale doorbell-system=3

# Scale up (Kubernetes)
kubectl scale deployment doorbell-system --replicas=5
```

### High Memory Usage

```bash
# Check memory
curl http://localhost:8000/metrics | grep memory_usage_bytes

# Increase limits (Kubernetes)
kubectl set resources deployment doorbell-system --limits=memory=4Gi
```

### Container Won't Start

```bash
# Check logs
docker-compose logs doorbell-system

# Check health
docker-compose exec doorbell-system /health_check.sh

# Restart
docker-compose restart doorbell-system
```

### Metrics Not Available

```bash
# Verify endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Restart Prometheus
docker-compose restart prometheus
```

## Security

### Generate Secrets

```bash
# Secret key
export SECRET_KEY="$(openssl rand -hex 32)"

# Password
export DB_PASSWORD="$(openssl rand -base64 32)"
```

### Verify Audit Logs

```python
from monitoring.logging.audit_logger import AuditLogger

audit = AuditLogger()
if audit.verify_integrity():
    print("Audit logs verified - no tampering")
else:
    print("WARNING: Audit log tampering detected!")
```

## Alert Rules

### Critical Alerts

- CPU > 95% for 1m
- Worker down for 30s
- Security event detected

### Warning Alerts

- CPU > 80% for 2m
- Memory > 85% for 3m
- Error rate > 10/min for 5m
- Face recognition accuracy < 85% for 10m

## Performance Optimization

### Recommended Settings

```bash
# Docker
WORKER_PROCESSES=4
WORKER_THREADS=2

# Kubernetes
replicas: 3
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi
```

### Scaling Guidelines

- **Light load**: 1-2 replicas, 512Mi RAM
- **Medium load**: 3-4 replicas, 1Gi RAM
- **Heavy load**: 5-10 replicas, 2Gi RAM

## Backup and Recovery

### Backup

```bash
# Manual backup
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Automated backup (set in cron)
0 2 * * * /opt/doorbell-system/scripts/backup.sh
```

### Restore

```bash
# Stop application
docker-compose down

# Restore data
tar -xzf backup-20250129.tar.gz

# Start application
docker-compose up -d
```

## Support

- **Documentation**: `/docs/PRODUCTION_DEPLOYMENT.md`
- **Monitoring Guide**: `/docs/MONITORING.md`
- **GitHub Issues**: https://github.com/itsnothuy/Doorbell-System/issues

## Checklist

### Pre-Deployment

- [ ] Set `SECRET_KEY` environment variable
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Set `DEBUG=False`
- [ ] Configure monitoring
- [ ] Setup backup schedule
- [ ] Test health checks
- [ ] Configure alerts

### Post-Deployment

- [ ] Verify application health
- [ ] Check metrics endpoint
- [ ] Verify Prometheus scraping
- [ ] Test alert notifications
- [ ] Verify audit logging
- [ ] Test rollback procedure
- [ ] Document any customizations

## Quick Links

- [Full Deployment Guide](PRODUCTION_DEPLOYMENT.md)
- [Monitoring Guide](MONITORING.md)
- [Infrastructure as Code](../infrastructure/)
- [Configuration](../config/production/)
