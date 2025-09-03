# Doorbell Security System Makefile

.PHONY: help install test start stop status clean lint format deps

# Default target
help:
	@echo "Doorbell Face Recognition Security System"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install system and dependencies"
	@echo "  test       - Run test suite"
	@echo "  start      - Start the service"
	@echo "  stop       - Stop the service"
	@echo "  status     - Show service status"
	@echo "  restart    - Restart the service"
	@echo "  logs       - Show recent logs"
	@echo "  clean      - Clean temporary files"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code"
	@echo "  deps       - Install/update dependencies"
	@echo "  backup     - Backup face databases"
	@echo "  restore    - Restore face databases"

# Installation
install:
	@echo "🔧 Installing Doorbell Security System..."
	chmod +x setup.sh
	./setup.sh

# Testing
test:
	@echo "🧪 Running tests..."
	./scripts/test.sh

test-python:
	@echo "🐍 Running Python unit tests..."
	source venv/bin/activate && python -m pytest tests/ -v

# Service management
start:
	@echo "▶️ Starting service..."
	./scripts/start.sh

stop:
	@echo "⏹️ Stopping service..."
	./scripts/stop.sh

restart: stop start

status:
	@echo "📊 Service status..."
	./scripts/status.sh

logs:
	@echo "📋 Recent logs..."
	tail -n 50 data/logs/doorbell.log

logs-follow:
	@echo "📋 Following logs..."
	tail -f data/logs/doorbell.log

# Development
lint:
	@echo "🔍 Running linter..."
	source venv/bin/activate && python -m flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

format:
	@echo "✨ Formatting code..."
	source venv/bin/activate && python -m black src/ tests/ --line-length=100

format-check:
	@echo "🔍 Checking code format..."
	source venv/bin/activate && python -m black src/ tests/ --line-length=100 --check

# Dependencies
deps:
	@echo "📦 Installing dependencies..."
	source venv/bin/activate && pip install -r requirements.txt

deps-update:
	@echo "📦 Updating dependencies..."
	source venv/bin/activate && pip install -r requirements.txt --upgrade

deps-freeze:
	@echo "📦 Freezing dependencies..."
	source venv/bin/activate && pip freeze > requirements.lock

# Face database management
add-face:
	@echo "👤 Adding new face to database..."
	@read -p "Enter person's name: " name; \
	read -p "Enter image path: " image; \
	cp "$$image" "data/known_faces/$$name.jpg" && \
	echo "Face added: $$name"

list-faces:
	@echo "👥 Known faces:"
	@ls -1 data/known_faces/ | sed 's/\.jpg$$//' | sed 's/^/  - /'
	@echo ""
	@echo "🚫 Blacklisted faces:"
	@ls -1 data/blacklist_faces/ | sed 's/\.jpg$$//' | sed 's/^/  - /' 2>/dev/null || echo "  (none)"

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "backup_$$timestamp.tar.gz" \
		data/known_faces/ \
		data/blacklist_faces/ \
		config/credentials_telegram.py \
		.env && \
	echo "Backup created: backup_$$timestamp.tar.gz"

restore:
	@echo "📥 Restoring from backup..."
	@read -p "Enter backup file path: " backup; \
	tar -xzf "$$backup" && \
	echo "Backup restored"

# Maintenance
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log.*" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

clean-logs:
	@echo "🧹 Cleaning old logs..."
	find data/logs/ -type f -name "*.log" -mtime +30 -delete
	find data/captures/ -type f -name "*.jpg" -mtime +7 -delete

clean-cache:
	@echo "🧹 Cleaning face recognition cache..."
	rm -f data/known_faces_cache.pkl
	rm -f data/blacklist_faces_cache.pkl

# System monitoring
monitor:
	@echo "📊 System monitoring..."
	@echo "Memory usage:"
	@free -h
	@echo ""
	@echo "Disk usage:"
	@df -h /
	@echo ""
	@echo "Temperature:"
	@vcgencmd measure_temp 2>/dev/null || echo "N/A"
	@echo ""
	@echo "Service status:"
	@systemctl is-active doorbell-security.service

# Update system
update:
	@echo "🔄 Updating system..."
	git pull origin main
	make deps-update
	make restart

# Development setup
dev-setup:
	@echo "🛠️ Setting up development environment..."
	source venv/bin/activate && pip install -r requirements.txt
	source venv/bin/activate && pip install black flake8 pytest pytest-cov
	pre-commit install 2>/dev/null || true

# Documentation
docs:
	@echo "📚 Opening documentation..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open README.md; \
	elif command -v open > /dev/null; then \
		open README.md; \
	else \
		echo "Please open README.md manually"; \
	fi

# Security scan
security-scan:
	@echo "🔒 Running security scan..."
	source venv/bin/activate && pip install safety bandit
	source venv/bin/activate && safety check
	source venv/bin/activate && bandit -r src/ -f json -o security_report.json

# Performance test
perf-test:
	@echo "⚡ Running performance test..."
	source venv/bin/activate && python -c "
import time
from src.face_manager import FaceManager
from src.camera_handler import MockCameraHandler

print('Testing face recognition performance...')
face_manager = FaceManager()
camera = MockCameraHandler()
camera.initialize()

start_time = time.time()
for i in range(10):
    image = camera.capture_image()
    faces = face_manager.detect_faces(image)
    print(f'Frame {i+1}: {len(faces)} faces detected')

total_time = time.time() - start_time
fps = 10 / total_time
print(f'Average FPS: {fps:.2f}')
"

# Docker support (if available)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t doorbell-security .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -it --rm --privileged -v /dev:/dev doorbell-security

# Create release
release:
	@echo "🚀 Creating release..."
	@read -p "Enter version (e.g., v1.0.0): " version; \
	git tag -a "$$version" -m "Release $$version" && \
	git push origin "$$version" && \
	echo "Release $$version created"
