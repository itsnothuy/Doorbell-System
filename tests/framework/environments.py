#!/usr/bin/env python3
"""
Test Environment Management

Containerized test isolation and cross-platform environment setup.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if docker is available
try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    logger.warning("Docker library not available, Docker environment setup disabled")


class TestEnvironmentManager:
    """Manage test environments and isolation."""

    def __init__(self) -> None:
        self.docker_client: Optional[Any] = None
        self.temp_dirs: List[Path] = []
        self.active_containers: List[str] = []

    async def setup_docker_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup Docker-based test environment."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError(
                "Docker library not available. Install with: pip install docker"
            )

        try:
            self.docker_client = docker.from_env()

            # Build test image if needed
            image_name = config.get("image", "doorbell-test:latest")
            await self._ensure_test_image(image_name)

            # Create test network
            network_name = "doorbell-test-network"
            await self._create_test_network(network_name)

            # Start required services
            containers = await self._start_test_services(config, network_name)
            self.active_containers.extend(containers)

            return {
                "type": "docker",
                "image": image_name,
                "network": network_name,
                "containers": containers,
            }

        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to setup Docker environment: {e}")

    async def setup_local_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup local test environment with isolation."""
        # Create temporary directory structure
        temp_root = Path(tempfile.mkdtemp(prefix="doorbell_test_"))
        self.temp_dirs.append(temp_root)

        # Setup test data directories
        test_dirs = {
            "data": temp_root / "data",
            "logs": temp_root / "logs",
            "config": temp_root / "config",
            "fixtures": temp_root / "fixtures",
        }

        for dir_path in test_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Copy test fixtures
        fixtures_src = Path("tests/fixtures")
        if fixtures_src.exists():
            shutil.copytree(fixtures_src, test_dirs["fixtures"], dirs_exist_ok=True)

        # Create test configuration
        test_config = self._create_test_config(test_dirs)
        config_file = test_dirs["config"] / "test_settings.json"
        config_file.write_text(json.dumps(test_config, indent=2))

        logger.info(f"Local test environment setup complete: {temp_root}")

        return {
            "type": "local",
            "root_dir": str(temp_root),
            "directories": {k: str(v) for k, v in test_dirs.items()},
            "config_file": str(config_file),
        }

    async def _ensure_test_image(self, image_name: str) -> None:
        """Ensure test Docker image exists."""
        if not self.docker_client:
            return

        try:
            self.docker_client.images.get(image_name)
            logger.info(f"Test image {image_name} already exists")
        except Exception:
            logger.info(f"Building test image {image_name}...")
            # Build test image
            dockerfile_content = """
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    libopencv-dev \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-test.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt -r requirements-test.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV TEST_MODE=true

# Default command
CMD ["python", "-m", "pytest", "tests/"]
"""

            # Build image
            build_context = Path.cwd()
            self.docker_client.images.build(
                path=str(build_context),
                dockerfile=dockerfile_content,
                tag=image_name,
                rm=True,
            )
            logger.info(f"Test image {image_name} built successfully")

    async def _create_test_network(self, network_name: str) -> None:
        """Create Docker test network."""
        if not self.docker_client:
            return

        try:
            self.docker_client.networks.get(network_name)
            logger.info(f"Test network {network_name} already exists")
        except Exception:
            self.docker_client.networks.create(network_name, driver="bridge")
            logger.info(f"Created test network {network_name}")

    async def _start_test_services(
        self, config: Dict[str, Any], network_name: str
    ) -> List[str]:
        """Start required test services in Docker."""
        # This can be extended to start database, cache, or other services
        containers = []
        logger.info("No additional test services configured")
        return containers

    def _create_test_config(self, test_dirs: Dict[str, Path]) -> Dict[str, Any]:
        """Create test-specific configuration."""
        return {
            "test_mode": True,
            "database": {
                "path": str(test_dirs["data"] / "test.db"),
                "type": "sqlite",
            },
            "logging": {
                "level": "DEBUG",
                "file": str(test_dirs["logs"] / "test.log"),
            },
            "face_recognition": {
                "tolerance": 0.6,
                "model": "hog",
                "known_faces_dir": str(test_dirs["data"] / "known_faces"),
                "blacklist_faces_dir": str(test_dirs["data"] / "blacklist_faces"),
            },
            "hardware": {
                "mock_mode": True,
                "camera_enabled": False,
                "gpio_enabled": False,
            },
            "web": {
                "host": "127.0.0.1",
                "port": 5000,
                "debug": True,
            },
        }

    async def cleanup(self) -> None:
        """Cleanup test environments."""
        # Stop Docker containers
        if self.docker_client:
            for container_id in self.active_containers:
                try:
                    container = self.docker_client.containers.get(container_id)
                    container.stop(timeout=10)
                    container.remove()
                    logger.info(f"Cleaned up container {container_id}")
                except Exception as e:
                    logger.warning(f"Could not cleanup container {container_id}: {e}")

        # Remove temporary directories
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Removed temporary directory {temp_dir}")

        self.active_containers.clear()
        self.temp_dirs.clear()


# Async context manager for test environments
class TestEnvironmentContext:
    """Context manager for test environment setup and cleanup."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.manager = TestEnvironmentManager()
        self.env_info: Optional[Dict[str, Any]] = None

    async def __aenter__(self) -> Dict[str, Any]:
        """Setup test environment on enter."""
        env_type = self.config.get("type", "local")

        if env_type == "docker" and DOCKER_AVAILABLE:
            self.env_info = await self.manager.setup_docker_environment(self.config)
        else:
            self.env_info = await self.manager.setup_local_environment(self.config)

        return self.env_info

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup test environment on exit."""
        await self.manager.cleanup()
