#!/bin/bash
# Container health check script

set -e

# Check if application is responding
if ! curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "Application health check failed"
    exit 1
fi

# Check if Prometheus metrics endpoint is responding
if ! curl -f http://localhost:8000/metrics > /dev/null 2>&1; then
    echo "Metrics endpoint health check failed"
    exit 1
fi

echo "Health check passed"
exit 0
