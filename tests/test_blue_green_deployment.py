#!/usr/bin/env python3
"""
Tests for Blue-Green Deployment Manager
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from production.deployment.blue_green_deployer import (
    BlueGreenDeployer,
    DeploymentEnvironment,
    DeploymentStatus,
    DeploymentInfo,
)


class TestBlueGreenDeployer:
    """Test suite for BlueGreenDeployer."""

    @pytest.fixture
    def temp_deployment_dir(self, tmp_path):
        """Create temporary deployment directory."""
        deployment_dir = tmp_path / "deployment"
        deployment_dir.mkdir()
        return deployment_dir

    @pytest.fixture
    def deployer_config(self, temp_deployment_dir):
        """Create test deployer configuration."""
        return {
            "deployment_dir": str(temp_deployment_dir),
        }

    @pytest.fixture
    def deployer(self, deployer_config):
        """Create test deployer instance."""
        return BlueGreenDeployer(deployer_config)

    @pytest.fixture
    def source_package(self, tmp_path):
        """Create a test source package."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create required files
        (source_dir / "app.py").touch()
        (source_dir / "src").mkdir()
        (source_dir / "config").mkdir()

        return source_dir

    def test_initialization(self, deployer, temp_deployment_dir):
        """Test deployer initializes correctly."""
        assert deployer.deployment_dir == temp_deployment_dir
        assert deployer.blue_dir.exists()
        assert deployer.green_dir.exists()
        assert deployer.active_environment in [
            DeploymentEnvironment.BLUE,
            DeploymentEnvironment.GREEN,
        ]

    def test_get_inactive_environment(self, deployer):
        """Test getting inactive environment."""
        if deployer.active_environment == DeploymentEnvironment.BLUE:
            assert deployer._get_inactive_environment() == DeploymentEnvironment.GREEN
        else:
            assert deployer._get_inactive_environment() == DeploymentEnvironment.BLUE

    def test_get_environment_dir(self, deployer):
        """Test getting environment directory."""
        blue_dir = deployer._get_environment_dir(DeploymentEnvironment.BLUE)
        green_dir = deployer._get_environment_dir(DeploymentEnvironment.GREEN)

        assert blue_dir == deployer.blue_dir
        assert green_dir == deployer.green_dir

    @patch.object(BlueGreenDeployer, "_health_check")
    @patch.object(BlueGreenDeployer, "_validate_deployment")
    def test_deploy_success(self, mock_validate, mock_health, deployer, source_package):
        """Test successful deployment."""
        mock_health.return_value = True
        mock_validate.return_value = True

        result = deployer.deploy("1.0.0", source_package)

        assert result is True
        assert len(deployer.deployment_history) > 0
        assert deployer.deployment_history[-1]["version"] == "1.0.0"
        assert deployer.deployment_history[-1]["status"] == "active"

    @patch.object(BlueGreenDeployer, "_health_check")
    def test_deploy_health_check_failed(self, mock_health, deployer, source_package):
        """Test deployment failure due to health check."""
        mock_health.return_value = False

        result = deployer.deploy("1.0.0", source_package)

        assert result is False
        assert len(deployer.deployment_history) > 0
        assert deployer.deployment_history[-1]["status"] == "failed"

    @patch.object(BlueGreenDeployer, "_health_check")
    @patch.object(BlueGreenDeployer, "_validate_deployment")
    def test_deploy_validation_failed(
        self, mock_validate, mock_health, deployer, source_package
    ):
        """Test deployment failure due to validation."""
        mock_health.return_value = True
        mock_validate.return_value = False

        result = deployer.deploy("1.0.0", source_package)

        assert result is False
        assert deployer.deployment_history[-1]["status"] == "failed"

    def test_deploy_dry_run(self, deployer, source_package):
        """Test dry run deployment."""
        result = deployer.deploy("1.0.0", source_package, dry_run=True)

        assert result is True
        # Verify no actual changes were made
        target_env = deployer._get_inactive_environment()
        target_dir = deployer._get_environment_dir(target_env)
        # In dry run, directory should be empty or not modified
        assert len(list(target_dir.iterdir())) == 0

    def test_health_check_missing_files(self, deployer):
        """Test health check with missing required files."""
        # Create incomplete deployment
        test_dir = deployer.blue_dir
        test_dir.mkdir(parents=True, exist_ok=True)

        result = deployer._health_check(test_dir)

        assert result is False

    def test_health_check_complete_deployment(self, deployer):
        """Test health check with complete deployment."""
        # Create complete deployment
        test_dir = deployer.blue_dir
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "app.py").touch()
        (test_dir / "src").mkdir()
        (test_dir / "config").mkdir()

        result = deployer._health_check(test_dir)

        assert result is True

    @patch.object(BlueGreenDeployer, "_health_check")
    @patch.object(BlueGreenDeployer, "_validate_deployment")
    def test_rollback(self, mock_validate, mock_health, deployer, source_package):
        """Test rollback functionality."""
        # First deploy to blue
        mock_health.return_value = True
        mock_validate.return_value = True
        deployer.deploy("1.0.0", source_package)

        initial_env = deployer.active_environment

        # Perform rollback
        result = deployer.rollback()

        assert result is True
        assert deployer.active_environment != initial_env
        assert deployer.deployment_history[-1]["status"] == "rolled_back"

    def test_get_deployment_status(self, deployer):
        """Test getting deployment status."""
        status = deployer.get_deployment_status()

        assert "active_environment" in status
        assert "blue_exists" in status
        assert "green_exists" in status
        assert "deployment_count" in status

    @patch.object(BlueGreenDeployer, "_health_check")
    @patch.object(BlueGreenDeployer, "_validate_deployment")
    def test_get_deployment_history(self, mock_validate, mock_health, deployer, source_package):
        """Test getting deployment history."""
        mock_health.return_value = True
        mock_validate.return_value = True

        # Perform multiple deployments
        deployer.deploy("1.0.0", source_package)
        deployer.deploy("1.1.0", source_package)

        history = deployer.get_deployment_history(limit=5)

        assert len(history) == 2
        assert history[-1]["version"] == "1.1.0"

    def test_deployment_history_persistence(self, deployer_config):
        """Test deployment history is persisted."""
        # Create deployer and perform deployment
        deployer1 = BlueGreenDeployer(deployer_config)
        initial_count = len(deployer1.deployment_history)

        # Create new deployer instance - should load existing history
        deployer2 = BlueGreenDeployer(deployer_config)

        assert len(deployer2.deployment_history) == initial_count


class TestDeploymentInfo:
    """Test suite for DeploymentInfo."""

    def test_deployment_info_creation(self):
        """Test creating deployment info."""
        info = DeploymentInfo(
            environment=DeploymentEnvironment.BLUE,
            version="1.0.0",
            timestamp=datetime.now(),
            status=DeploymentStatus.ACTIVE,
        )

        assert info.environment == DeploymentEnvironment.BLUE
        assert info.version == "1.0.0"
        assert info.status == DeploymentStatus.ACTIVE

    def test_deployment_info_to_dict(self):
        """Test converting deployment info to dictionary."""
        info = DeploymentInfo(
            environment=DeploymentEnvironment.BLUE,
            version="1.0.0",
            timestamp=datetime.now(),
            status=DeploymentStatus.ACTIVE,
            metadata={"key": "value"},
        )

        data = info.to_dict()

        assert data["environment"] == "blue"
        assert data["version"] == "1.0.0"
        assert data["status"] == "active"
        assert data["metadata"]["key"] == "value"


class TestDeploymentEnvironment:
    """Test suite for DeploymentEnvironment."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert DeploymentEnvironment.BLUE.value == "blue"
        assert DeploymentEnvironment.GREEN.value == "green"


class TestDeploymentStatus:
    """Test suite for DeploymentStatus."""

    def test_status_values(self):
        """Test status enum values."""
        assert DeploymentStatus.PENDING.value == "pending"
        assert DeploymentStatus.IN_PROGRESS.value == "in_progress"
        assert DeploymentStatus.VALIDATING.value == "validating"
        assert DeploymentStatus.ACTIVE.value == "active"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.ROLLED_BACK.value == "rolled_back"
