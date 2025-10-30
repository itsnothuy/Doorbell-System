# Issue #24: Production Monitoring and Analytics Dashboard

## Overview
Implement a comprehensive production monitoring and analytics dashboard that provides real-time system health monitoring, performance analytics, security event tracking, and operational insights for the Doorbell Security System in production environments.

## Problem Statement
While the system has robust testing and demo capabilities, it lacks comprehensive production monitoring that provides:
- Real-time system health and performance monitoring
- Security event analytics and trend analysis
- Operational insights for system optimization
- Proactive alerting for system issues
- Historical data analysis and reporting
- User activity and system usage analytics

## Success Criteria
- [ ] Real-time system health monitoring with customizable dashboards
- [ ] Performance metrics collection and trend analysis
- [ ] Security event tracking and analytics
- [ ] Automated alerting for system anomalies
- [ ] Historical data analysis and reporting
- [ ] User activity and usage analytics
- [ ] Production deployment health validation
- [ ] Integration with monitoring tools (Prometheus, Grafana, ELK stack)

## Technical Requirements

### 1. Monitoring Core (`src/monitoring/core.py`)

```python
#!/usr/bin/env python3
"""
Production Monitoring Core

Core monitoring infrastructure for system health, performance, and security tracking.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import psutil
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricDefinition:
    """Definition of a monitoring metric."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None


@dataclass
class Alert:
    """System alert definition."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float
    source_component: str
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """System health status."""
    overall_status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)


class MonitoringCore:
    """Core monitoring system for production environments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_registry = {}
        self.alert_handlers: List[Callable] = []
        self.health_checkers: Dict[str, Callable] = {}
        self.running = False
        
        # Prometheus metrics
        self.prometheus_enabled = config.get("prometheus", {}).get("enabled", True)
        if self.prometheus_enabled:
            self._setup_prometheus_metrics()
        
        # Alert thresholds
        self.thresholds = config.get("thresholds", {})
        
        logger.info("Monitoring core initialized")
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        # System metrics
        self.system_cpu_usage = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
        self.system_memory_usage = Gauge('system_memory_usage_percent', 'System memory usage percentage')
        self.system_disk_usage = Gauge('system_disk_usage_percent', 'System disk usage percentage')
        
        # Application metrics
        self.face_detection_duration = Histogram(
            'face_detection_duration_seconds',
            'Face detection processing time',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        self.face_recognition_duration = Histogram(
            'face_recognition_duration_seconds', 
            'Face recognition processing time',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        self.pipeline_events_total = Counter(
            'pipeline_events_total',
            'Total pipeline events processed',
            ['event_type', 'status']
        )
        
        # Security metrics
        self.face_recognitions_total = Counter(
            'face_recognitions_total',
            'Total face recognitions',
            ['result_type', 'person_name']
        )
        self.security_alerts_total = Counter(
            'security_alerts_total',
            'Total security alerts',
            ['alert_type', 'severity']
        )
        
        # API metrics
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
    
    async def start_monitoring(self) -> None:
        """Start monitoring services."""
        self.running = True
        
        # Start Prometheus server if enabled
        if self.prometheus_enabled:
            prometheus_port = self.config.get("prometheus", {}).get("port", 8000)
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        
        # Start monitoring tasks
        tasks = [
            self._monitor_system_health(),
            self._monitor_application_metrics(),
            self._process_alerts()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring services."""
        self.running = False
        logger.info("Monitoring stopped")
    
    async def _monitor_system_health(self) -> None:
        """Monitor system health metrics."""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.system_memory_usage.set(memory_percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                self.system_disk_usage.set(disk_percent)
                
                # Check thresholds and generate alerts
                await self._check_system_thresholds(cpu_percent, memory_percent, disk_percent)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"System health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_application_metrics(self) -> None:
        """Monitor application-specific metrics."""
        while self.running:
            try:
                # Check pipeline health
                pipeline_health = await self._check_pipeline_health()
                
                # Check API health
                api_health = await self._check_api_health()
                
                # Check database health
                database_health = await self._check_database_health()
                
                # Aggregate health status
                overall_health = self._calculate_overall_health({
                    "pipeline": pipeline_health,
                    "api": api_health,
                    "database": database_health
                })
                
                # Store health status
                await self._store_health_status(overall_health)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Application monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _process_alerts(self) -> None:
        """Process and handle alerts."""
        while self.running:
            try:
                # Process pending alerts
                alerts = await self._get_pending_alerts()
                
                for alert in alerts:
                    await self._handle_alert(alert)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30)
    
    async def _check_system_thresholds(self, cpu: float, memory: float, disk: float) -> None:
        """Check system metrics against thresholds."""
        thresholds = self.thresholds.get("system", {})
        
        # CPU threshold
        cpu_threshold = thresholds.get("cpu_percent", 80)
        if cpu > cpu_threshold:
            alert = Alert(
                alert_id=f"high_cpu_{int(time.time())}",
                severity=AlertSeverity.WARNING if cpu < 90 else AlertSeverity.CRITICAL,
                title="High CPU Usage",
                description=f"CPU usage is {cpu:.1f}%, exceeding threshold of {cpu_threshold}%",
                timestamp=time.time(),
                source_component="system_monitor",
                metric_value=cpu,
                threshold_value=cpu_threshold
            )
            await self._emit_alert(alert)
        
        # Memory threshold
        memory_threshold = thresholds.get("memory_percent", 85)
        if memory > memory_threshold:
            alert = Alert(
                alert_id=f"high_memory_{int(time.time())}",
                severity=AlertSeverity.WARNING if memory < 95 else AlertSeverity.CRITICAL,
                title="High Memory Usage",
                description=f"Memory usage is {memory:.1f}%, exceeding threshold of {memory_threshold}%",
                timestamp=time.time(),
                source_component="system_monitor",
                metric_value=memory,
                threshold_value=memory_threshold
            )
            await self._emit_alert(alert)
        
        # Disk threshold
        disk_threshold = thresholds.get("disk_percent", 90)
        if disk > disk_threshold:
            alert = Alert(
                alert_id=f"high_disk_{int(time.time())}",
                severity=AlertSeverity.CRITICAL,
                title="High Disk Usage",
                description=f"Disk usage is {disk:.1f}%, exceeding threshold of {disk_threshold}%",
                timestamp=time.time(),
                source_component="system_monitor",
                metric_value=disk,
                threshold_value=disk_threshold
            )
            await self._emit_alert(alert)
    
    async def _check_pipeline_health(self) -> Dict[str, Any]:
        """Check pipeline component health."""
        try:
            # This would check actual pipeline components
            # For now, return mock health data
            return {
                "status": "healthy",
                "components": {
                    "frame_capture": {"status": "healthy", "fps": 25.0},
                    "motion_detection": {"status": "healthy", "latency": 0.1},
                    "face_detection": {"status": "healthy", "queue_size": 2},
                    "face_recognition": {"status": "healthy", "success_rate": 0.95},
                    "event_processor": {"status": "healthy", "throughput": 15.0}
                }
            }
        except Exception as e:
            logger.error(f"Pipeline health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API endpoint health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:5000/api/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"status": "healthy", "response_time": data.get("response_time", 0)}
                    else:
                        return {"status": "unhealthy", "http_status": response.status}
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            # This would check actual database connection
            # For now, return mock health data
            return {
                "status": "healthy",
                "connection_pool": {"active": 2, "idle": 8},
                "query_performance": {"avg_duration": 0.05}
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def _calculate_overall_health(self, component_health: Dict[str, Dict[str, Any]]) -> SystemHealth:
        """Calculate overall system health."""
        unhealthy_components = [
            name for name, health in component_health.items()
            if health.get("status") != "healthy"
        ]
        
        if not unhealthy_components:
            overall_status = "healthy"
        elif len(unhealthy_components) == 1:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return SystemHealth(
            overall_status=overall_status,
            timestamp=time.time(),
            components=component_health
        )
    
    async def _store_health_status(self, health: SystemHealth) -> None:
        """Store health status for historical analysis."""
        # This would store to database or time series database
        logger.debug(f"Health status: {health.overall_status}")
    
    async def _emit_alert(self, alert: Alert) -> None:
        """Emit alert to handlers."""
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    async def _get_pending_alerts(self) -> List[Alert]:
        """Get pending alerts for processing."""
        # This would query pending alerts from storage
        return []
    
    async def _handle_alert(self, alert: Alert) -> None:
        """Handle individual alert."""
        logger.info(f"Processing alert: {alert.title} ({alert.severity.value})")
        
        # Send to monitoring systems
        if self.prometheus_enabled:
            self.security_alerts_total.labels(
                alert_type=alert.source_component,
                severity=alert.severity.value
            ).inc()
    
    def record_face_detection(self, duration: float) -> None:
        """Record face detection performance."""
        if self.prometheus_enabled:
            self.face_detection_duration.observe(duration)
    
    def record_face_recognition(self, duration: float, result_type: str, person_name: str = "unknown") -> None:
        """Record face recognition performance."""
        if self.prometheus_enabled:
            self.face_recognition_duration.observe(duration)
            self.face_recognitions_total.labels(
                result_type=result_type,
                person_name=person_name
            ).inc()
    
    def record_pipeline_event(self, event_type: str, status: str) -> None:
        """Record pipeline event."""
        if self.prometheus_enabled:
            self.pipeline_events_total.labels(
                event_type=event_type,
                status=status
            ).inc()
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record API request metrics."""
        if self.prometheus_enabled:
            self.api_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            self.api_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
```

### 2. Analytics Dashboard (`src/monitoring/dashboard.py`)

```python
#!/usr/bin/env python3
"""
Analytics Dashboard

Web-based analytics dashboard for system monitoring and insights.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2

logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    """Web-based analytics dashboard."""
    
    def __init__(self, config: Dict[str, Any], monitoring_core):
        self.config = config
        self.monitoring_core = monitoring_core
        self.app = web.Application()
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8080)
        
        # Setup templates
        template_dir = Path(__file__).parent / "templates"
        aiohttp_jinja2.setup(
            self.app,
            loader=jinja2.FileSystemLoader(str(template_dir))
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Analytics dashboard initialized on {self.host}:{self.port}")
    
    def _setup_routes(self) -> None:
        """Setup web routes."""
        # Dashboard routes
        self.app.router.add_get('/', self.dashboard_home)
        self.app.router.add_get('/dashboard', self.dashboard_home)
        self.app.router.add_get('/health', self.health_dashboard)
        self.app.router.add_get('/performance', self.performance_dashboard)
        self.app.router.add_get('/security', self.security_dashboard)
        self.app.router.add_get('/alerts', self.alerts_dashboard)
        
        # API routes
        self.app.router.add_get('/api/metrics/system', self.api_system_metrics)
        self.app.router.add_get('/api/metrics/performance', self.api_performance_metrics)
        self.app.router.add_get('/api/metrics/security', self.api_security_metrics)
        self.app.router.add_get('/api/alerts/recent', self.api_recent_alerts)
        self.app.router.add_get('/api/health/status', self.api_health_status)
        
        # Static files
        self.app.router.add_static('/static/', path='src/monitoring/static/', name='static')
    
    async def start_server(self) -> None:
        """Start dashboard server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Analytics dashboard started on http://{self.host}:{self.port}")
    
    @aiohttp_jinja2.template('dashboard.html')
    async def dashboard_home(self, request: web.Request) -> Dict[str, Any]:
        """Main dashboard page."""
        # Get current system status
        system_health = await self._get_system_health()
        recent_alerts = await self._get_recent_alerts(limit=5)
        performance_summary = await self._get_performance_summary()
        
        return {
            'system_health': system_health,
            'recent_alerts': recent_alerts,
            'performance_summary': performance_summary,
            'dashboard_title': 'Doorbell Security System - Monitoring Dashboard'
        }
    
    @aiohttp_jinja2.template('health.html')
    async def health_dashboard(self, request: web.Request) -> Dict[str, Any]:
        """System health dashboard."""
        health_data = await self._get_detailed_health()
        
        return {
            'health_data': health_data,
            'dashboard_title': 'System Health Monitor'
        }
    
    @aiohttp_jinja2.template('performance.html')
    async def performance_dashboard(self, request: web.Request) -> Dict[str, Any]:
        """Performance analytics dashboard."""
        # Get time range from query parameters
        hours = int(request.query.get('hours', 24))
        performance_data = await self._get_performance_data(hours)
        
        return {
            'performance_data': performance_data,
            'time_range_hours': hours,
            'dashboard_title': 'Performance Analytics'
        }
    
    @aiohttp_jinja2.template('security.html')
    async def security_dashboard(self, request: web.Request) -> Dict[str, Any]:
        """Security events dashboard."""
        # Get time range from query parameters
        hours = int(request.query.get('hours', 24))
        security_data = await self._get_security_data(hours)
        
        return {
            'security_data': security_data,
            'time_range_hours': hours,
            'dashboard_title': 'Security Analytics'
        }
    
    @aiohttp_jinja2.template('alerts.html')
    async def alerts_dashboard(self, request: web.Request) -> Dict[str, Any]:
        """Alerts dashboard."""
        # Get pagination parameters
        page = int(request.query.get('page', 1))
        limit = int(request.query.get('limit', 50))
        
        alerts_data = await self._get_alerts_data(page, limit)
        
        return {
            'alerts_data': alerts_data,
            'current_page': page,
            'dashboard_title': 'System Alerts'
        }
    
    async def api_system_metrics(self, request: web.Request) -> web.Response:
        """API endpoint for system metrics."""
        try:
            metrics = await self._get_system_metrics()
            return web.json_response(metrics)
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def api_performance_metrics(self, request: web.Request) -> web.Response:
        """API endpoint for performance metrics."""
        try:
            hours = int(request.query.get('hours', 24))
            metrics = await self._get_performance_metrics(hours)
            return web.json_response(metrics)
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def api_security_metrics(self, request: web.Request) -> web.Response:
        """API endpoint for security metrics."""
        try:
            hours = int(request.query.get('hours', 24))
            metrics = await self._get_security_metrics(hours)
            return web.json_response(metrics)
        except Exception as e:
            logger.error(f"Failed to get security metrics: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def api_recent_alerts(self, request: web.Request) -> web.Response:
        """API endpoint for recent alerts."""
        try:
            limit = int(request.query.get('limit', 20))
            alerts = await self._get_recent_alerts(limit)
            return web.json_response({"alerts": alerts})
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def api_health_status(self, request: web.Request) -> web.Response:
        """API endpoint for health status."""
        try:
            health = await self._get_system_health()
            return web.json_response(health)
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        # This would query actual health data
        return {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "pipeline": {"status": "healthy", "uptime": "2d 5h 30m"},
                "api": {"status": "healthy", "response_time": 0.05},
                "database": {"status": "healthy", "connections": 8},
                "hardware": {"status": "healthy", "temperature": 45.2}
            }
        }
    
    async def _get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        # This would query actual alerts from database
        return [
            {
                "id": "alert_001",
                "severity": "warning",
                "title": "High Memory Usage",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "resolved": True
            },
            {
                "id": "alert_002", 
                "severity": "info",
                "title": "New Face Detected",
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                "resolved": False
            }
        ]
    
    async def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "face_detection_avg": 0.8,
            "face_recognition_avg": 1.2,
            "pipeline_throughput": 15.5,
            "api_response_time": 0.05,
            "system_load": 0.65
        }
    
    async def _get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information."""
        return {
            "system": {
                "cpu_usage": 45.2,
                "memory_usage": 68.5,
                "disk_usage": 34.7,
                "network_io": {"rx": 1250000, "tx": 890000}
            },
            "application": {
                "pipeline_status": "healthy",
                "active_workers": 4,
                "queue_sizes": {"detection": 2, "recognition": 1, "events": 0},
                "error_rate": 0.02
            }
        }
    
    async def _get_performance_data(self, hours: int) -> Dict[str, Any]:
        """Get performance data for time range."""
        # This would query actual performance metrics
        return {
            "time_range": f"{hours} hours",
            "face_detection": {
                "avg_duration": 0.8,
                "min_duration": 0.3,
                "max_duration": 2.1,
                "success_rate": 0.98
            },
            "face_recognition": {
                "avg_duration": 1.2,
                "min_duration": 0.5,
                "max_duration": 3.2,
                "success_rate": 0.94
            },
            "pipeline_throughput": {
                "avg_fps": 24.5,
                "events_per_hour": 145
            }
        }
    
    async def _get_security_data(self, hours: int) -> Dict[str, Any]:
        """Get security data for time range."""
        return {
            "time_range": f"{hours} hours",
            "face_recognitions": {
                "total": 45,
                "known_persons": 38,
                "unknown_persons": 7,
                "blacklisted": 0
            },
            "security_events": {
                "total_alerts": 3,
                "high_priority": 0,
                "medium_priority": 2,
                "low_priority": 1
            },
            "activity_timeline": [
                {"time": "2024-01-20 14:30", "event": "John Doe recognized", "type": "known"},
                {"time": "2024-01-20 13:45", "event": "Unknown person detected", "type": "unknown"},
                {"time": "2024-01-20 12:15", "event": "Jane Smith recognized", "type": "known"}
            ]
        }
    
    async def _get_alerts_data(self, page: int, limit: int) -> Dict[str, Any]:
        """Get alerts data with pagination."""
        # This would query actual alerts with pagination
        return {
            "alerts": [
                {
                    "id": f"alert_{i:03d}",
                    "severity": ["info", "warning", "critical"][i % 3],
                    "title": f"Alert {i}",
                    "description": f"Description for alert {i}",
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                    "resolved": i % 2 == 0
                }
                for i in range((page - 1) * limit, page * limit)
            ],
            "pagination": {
                "current_page": page,
                "total_pages": 10,
                "total_alerts": 500,
                "limit": limit
            }
        }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        import psutil
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict()
            }
        }
    
    async def _get_performance_metrics(self, hours: int) -> Dict[str, Any]:
        """Get performance metrics for time range."""
        # This would query time series data
        return {
            "time_range_hours": hours,
            "metrics": {
                "avg_face_detection_time": 0.8,
                "avg_face_recognition_time": 1.2,
                "pipeline_fps": 24.5,
                "api_response_time": 0.05
            }
        }
    
    async def _get_security_metrics(self, hours: int) -> Dict[str, Any]:
        """Get security metrics for time range."""
        return {
            "time_range_hours": hours,
            "metrics": {
                "total_recognitions": 45,
                "known_person_rate": 0.84,
                "unknown_person_rate": 0.16,
                "security_alerts": 3
            }
        }
```

### 3. Alert Management (`src/monitoring/alerts.py`)

```python
#!/usr/bin/env python3
"""
Alert Management System

Comprehensive alerting system for production monitoring.
"""

import asyncio
import json
import logging
import smtplib
from dataclasses import dataclass, field
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Any, Dict, List, Optional, Callable

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    type: str  # "email", "webhook", "slack", "telegram"
    config: Dict[str, Any]
    enabled: bool = True


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: str  # Python expression
    severity: str
    message_template: str
    channels: List[str]
    cooldown_minutes: int = 15
    enabled: bool = True


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.channels: Dict[str, NotificationChannel] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.cooldown_tracker: Dict[str, float] = {}
        
        # Load configuration
        self._load_channels()
        self._load_rules()
        
        logger.info("Alert manager initialized")
    
    def _load_channels(self) -> None:
        """Load notification channels from configuration."""
        channels_config = self.config.get("channels", {})
        
        for name, channel_config in channels_config.items():
            channel = NotificationChannel(
                name=name,
                type=channel_config["type"],
                config=channel_config.get("config", {}),
                enabled=channel_config.get("enabled", True)
            )
            self.channels[name] = channel
    
    def _load_rules(self) -> None:
        """Load alert rules from configuration."""
        rules_config = self.config.get("rules", {})
        
        for name, rule_config in rules_config.items():
            rule = AlertRule(
                name=name,
                condition=rule_config["condition"],
                severity=rule_config["severity"],
                message_template=rule_config["message_template"],
                channels=rule_config["channels"],
                cooldown_minutes=rule_config.get("cooldown_minutes", 15),
                enabled=rule_config.get("enabled", True)
            )
            self.rules[name] = rule
    
    async def process_metrics(self, metrics: Dict[str, Any]) -> None:
        """Process metrics and trigger alerts if needed."""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check if rule condition is met
                if self._evaluate_condition(rule.condition, metrics):
                    # Check cooldown
                    if self._is_in_cooldown(rule_name):
                        logger.debug(f"Alert rule {rule_name} in cooldown, skipping")
                        continue
                    
                    # Create alert
                    alert = {
                        "rule_name": rule_name,
                        "severity": rule.severity,
                        "message": self._format_message(rule.message_template, metrics),
                        "timestamp": asyncio.get_event_loop().time(),
                        "metrics": metrics
                    }
                    
                    # Send notifications
                    await self._send_alert(alert, rule.channels)
                    
                    # Update cooldown
                    self._update_cooldown(rule_name, rule.cooldown_minutes)
                    
            except Exception as e:
                logger.error(f"Error processing alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition against metrics."""
        try:
            # Create safe evaluation context
            context = {
                "metrics": metrics,
                "abs": abs,
                "max": max,
                "min": min,
                "len": len
            }
            
            # Evaluate condition
            return eval(condition, {"__builtins__": {}}, context)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if alert rule is in cooldown period."""
        if rule_name not in self.cooldown_tracker:
            return False
        
        current_time = asyncio.get_event_loop().time()
        cooldown_end = self.cooldown_tracker[rule_name]
        
        return current_time < cooldown_end
    
    def _update_cooldown(self, rule_name: str, cooldown_minutes: int) -> None:
        """Update cooldown timer for alert rule."""
        current_time = asyncio.get_event_loop().time()
        self.cooldown_tracker[rule_name] = current_time + (cooldown_minutes * 60)
    
    def _format_message(self, template: str, metrics: Dict[str, Any]) -> str:
        """Format alert message using template and metrics."""
        try:
            return template.format(**metrics)
        except Exception as e:
            logger.error(f"Error formatting message template: {e}")
            return f"Alert triggered (template formatting error: {e})"
    
    async def _send_alert(self, alert: Dict[str, Any], channels: List[str]) -> None:
        """Send alert to specified channels."""
        for channel_name in channels:
            if channel_name not in self.channels:
                logger.error(f"Unknown notification channel: {channel_name}")
                continue
            
            channel = self.channels[channel_name]
            if not channel.enabled:
                continue
            
            try:
                await self._send_to_channel(alert, channel)
                logger.info(f"Alert sent to {channel_name}: {alert['message']}")
                
            except Exception as e:
                logger.error(f"Failed to send alert to {channel_name}: {e}")
    
    async def _send_to_channel(self, alert: Dict[str, Any], channel: NotificationChannel) -> None:
        """Send alert to specific notification channel."""
        if channel.type == "email":
            await self._send_email_alert(alert, channel)
        elif channel.type == "webhook":
            await self._send_webhook_alert(alert, channel)
        elif channel.type == "slack":
            await self._send_slack_alert(alert, channel)
        elif channel.type == "telegram":
            await self._send_telegram_alert(alert, channel)
        else:
            raise ValueError(f"Unsupported channel type: {channel.type}")
    
    async def _send_email_alert(self, alert: Dict[str, Any], channel: NotificationChannel) -> None:
        """Send email alert."""
        config = channel.config
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(config['to_emails'])
        msg['Subject'] = f"[{alert['severity'].upper()}] Doorbell Security Alert"
        
        # Email body
        body = f"""
        Alert: {alert['message']}
        
        Severity: {alert['severity']}
        Timestamp: {alert['timestamp']}
        Rule: {alert['rule_name']}
        
        Metrics:
        {json.dumps(alert['metrics'], indent=2)}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587)) as server:
            if config.get('use_tls', True):
                server.starttls()
            
            if 'username' in config and 'password' in config:
                server.login(config['username'], config['password'])
            
            server.send_message(msg)
    
    async def _send_webhook_alert(self, alert: Dict[str, Any], channel: NotificationChannel) -> None:
        """Send webhook alert."""
        config = channel.config
        
        payload = {
            "alert": alert,
            "timestamp": alert['timestamp'],
            "severity": alert['severity'],
            "message": alert['message']
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
    
    async def _send_slack_alert(self, alert: Dict[str, Any], channel: NotificationChannel) -> None:
        """Send Slack alert."""
        config = channel.config
        
        # Slack message format
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9500", 
            "critical": "#ff0000",
            "emergency": "#8b0000"
        }
        
        payload = {
            "channel": config['channel'],
            "username": "Doorbell Security",
            "icon_emoji": ":rotating_light:",
            "attachments": [{
                "color": color_map.get(alert['severity'], "#cccccc"),
                "title": f"{alert['severity'].upper()} Alert",
                "text": alert['message'],
                "fields": [
                    {"title": "Rule", "value": alert['rule_name'], "short": True},
                    {"title": "Timestamp", "value": str(alert['timestamp']), "short": True}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config['webhook_url'],
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
    
    async def _send_telegram_alert(self, alert: Dict[str, Any], channel: NotificationChannel) -> None:
        """Send Telegram alert."""
        config = channel.config
        
        message = f"""
🚨 *{alert['severity'].upper()} Alert*

{alert['message']}

*Rule:* {alert['rule_name']}
*Time:* {alert['timestamp']}
        """
        
        payload = {
            "chat_id": config['chat_id'],
            "text": message,
            "parse_mode": "Markdown"
        }
        
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
```

## Acceptance Criteria

### Core Monitoring
- [ ] **Real-time Metrics**: System health, performance, and security metrics collection
- [ ] **Prometheus Integration**: Metrics export for Prometheus/Grafana integration  
- [ ] **Health Monitoring**: Component health checks with automated status determination
- [ ] **Performance Tracking**: Response time, throughput, and resource usage monitoring

### Analytics Dashboard
- [ ] **Web Interface**: Responsive web dashboard with real-time data
- [ ] **Multiple Views**: Health, performance, security, and alerts dashboards
- [ ] **Historical Analysis**: Time-series data visualization and trend analysis
- [ ] **API Endpoints**: RESTful API for metrics and dashboard data

### Alerting System
- [ ] **Rule-based Alerts**: Configurable alert rules with custom conditions
- [ ] **Multiple Channels**: Email, webhook, Slack, and Telegram notifications
- [ ] **Alert Management**: Cooldown periods, severity levels, and alert history
- [ ] **Escalation**: Automated escalation for critical alerts

### Production Integration
- [ ] **Containerization**: Docker support for monitoring components
- [ ] **Configuration**: Environment-based configuration management
- [ ] **Security**: Secure access controls for monitoring interfaces
- [ ] **Performance**: Minimal overhead monitoring system

## Implementation Plan

### Phase 1: Core Monitoring (Week 1)
1. Implement `MonitoringCore` with Prometheus metrics
2. Add system health monitoring and checks
3. Implement performance metrics collection
4. Add basic alerting infrastructure

### Phase 2: Analytics Dashboard (Week 2)
1. Create web-based analytics dashboard
2. Implement multiple dashboard views
3. Add real-time data visualization
4. Create API endpoints for dashboard data

### Phase 3: Alert Management (Week 3)
1. Implement comprehensive alert management system
2. Add multiple notification channels
3. Create rule-based alerting with cooldowns
4. Add alert history and management features

### Phase 4: Integration & Production (Week 4)
1. Docker containerization and deployment
2. Production configuration and security
3. Performance optimization and testing
4. Documentation and operational guides

## Dependencies

### Required Packages
```txt
prometheus-client>=0.17.0
aiohttp>=3.8.0
aiohttp-jinja2>=1.5.0
psutil>=5.9.0
jinja2>=3.1.0
```

### External Services
- Prometheus (metrics collection)
- Grafana (visualization)
- SMTP server (email alerts)
- Slack/Telegram (notifications)

---

**Estimated Effort**: 4 weeks (160 hours)
**Priority**: High (Essential for production operations)
**Dependencies**: Production deployment infrastructure