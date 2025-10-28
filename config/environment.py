#!/usr/bin/env python3
"""
Environment-Specific Configuration Management

Handles multiple environments with secure overrides and validation.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum

from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    @classmethod
    def from_string(cls, env_str: str) -> 'Environment':
        """Create Environment from string.
        
        Args:
            env_str: Environment name as string
            
        Returns:
            Environment: Environment enum value
            
        Raises:
            ValueError: If environment string is invalid
        """
        try:
            return cls(env_str.lower())
        except ValueError:
            raise ValueError(
                f"Invalid environment '{env_str}'. "
                f"Valid options: {', '.join(e.value for e in cls)}"
            )


class EnvironmentManager:
    """Manages environment-specific configurations."""
    
    def __init__(self, base_config_path: Optional[Path] = None):
        """Initialize environment manager.
        
        Args:
            base_config_path: Optional path to base configuration
        """
        self.base_config_path = base_config_path or Path(__file__).parent / "pipeline_config.py"
        self.current_env = self._detect_environment()
        
        logger.info(f"Environment manager initialized for: {self.current_env.value}")
    
    def load_environment_config(self) -> PipelineConfig:
        """Load configuration for current environment.
        
        Returns:
            PipelineConfig: Environment-specific configuration
        """
        # Load base configuration
        base_config = self._load_base_config()
        
        # Load environment-specific overrides
        env_overrides = self._load_environment_overrides()
        
        # Apply environment variables
        self._apply_env_variables(base_config, env_overrides)
        
        # Apply environment-specific settings
        self._apply_environment_defaults(base_config)
        
        # Validate environment-specific constraints
        self._validate_environment_config(base_config)
        
        return base_config
    
    def _detect_environment(self) -> Environment:
        """Auto-detect current environment from environment variables.
        
        Returns:
            Environment: Detected environment
        """
        # Check for explicit environment variable
        env_name = os.getenv('DOORBELL_ENV', os.getenv('ENV', 'development'))
        
        # Also check common CI/CD environment variables
        if os.getenv('CI'):
            env_name = 'testing'
        elif os.getenv('RAILWAY_ENVIRONMENT'):
            env_name = os.getenv('RAILWAY_ENVIRONMENT')
        elif os.getenv('RENDER'):
            env_name = 'production'
        elif os.getenv('VERCEL_ENV'):
            env_name = os.getenv('VERCEL_ENV')
        
        try:
            return Environment.from_string(env_name)
        except ValueError as e:
            logger.warning(f"{e}, defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_base_config(self) -> PipelineConfig:
        """Load base configuration.
        
        Returns:
            PipelineConfig: Base configuration
        """
        return PipelineConfig()
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load environment-specific configuration overrides.
        
        Returns:
            Dict[str, Any]: Environment-specific overrides
        """
        env_config_path = (
            self.base_config_path.parent / 
            "environments" / 
            f"env_{self.current_env.value}.py"
        )
        
        if env_config_path.exists():
            try:
                return self._load_config_from_file(env_config_path)
            except Exception as e:
                logger.warning(f"Failed to load environment config from {env_config_path}: {e}")
        
        return {}
    
    def _load_config_from_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from Python file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("env_config", config_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extract configuration from module
            config = {}
            for key in dir(module):
                if not key.startswith('_'):
                    config[key] = getattr(module, key)
            
            return config
        
        return {}
    
    def _apply_env_variables(
        self,
        config: PipelineConfig,
        env_overrides: Dict[str, Any]
    ) -> None:
        """Apply environment variable overrides to configuration.
        
        Args:
            config: Configuration to update
            env_overrides: Environment-specific overrides
        """
        # Define environment variable mappings
        env_mappings = {
            # Logging
            'DOORBELL_LOG_LEVEL': ('storage', 'log_level'),
            'LOG_LEVEL': ('storage', 'log_level'),
            
            # Performance
            'DOORBELL_WORKER_COUNT': ('face_detection', 'worker_count'),
            'WORKER_COUNT': ('face_detection', 'worker_count'),
            
            # Face recognition
            'DOORBELL_FACE_TOLERANCE': ('face_recognition', 'tolerance'),
            'FACE_RECOGNITION_TOLERANCE': ('face_recognition', 'tolerance'),
            
            # Frame capture
            'DOORBELL_CAMERA_FPS': ('frame_capture', 'fps'),
            'CAMERA_FPS': ('frame_capture', 'fps'),
            
            # Hardware
            'DOORBELL_PIN': ('hardware', 'doorbell_pin'),
            'DOORBELL_DETECTOR_TYPE': ('face_detection', 'detector_type'),
            
            # Web interface
            'WEB_PORT': ('web', 'port'),
            'PORT': ('web', 'port'),
            
            # Storage
            'DATA_PATH': ('storage', 'base_path'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                try:
                    # Get the section
                    section_obj = getattr(config, section, None)
                    if section_obj and hasattr(section_obj, key):
                        # Convert value to appropriate type
                        current_value = getattr(section_obj, key)
                        converted_value = self._convert_env_value(env_value, type(current_value))
                        setattr(section_obj, key, converted_value)
                        logger.debug(f"Applied env var {env_var} to {section}.{key}")
                except Exception as e:
                    logger.warning(f"Failed to apply env var {env_var}: {e}")
    
    def _convert_env_value(self, value: str, target_type: type) -> Any:
        """Convert environment variable string to target type.
        
        Args:
            value: String value from environment
            target_type: Target type to convert to
            
        Returns:
            Any: Converted value
        """
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        else:
            return value
    
    def _apply_environment_defaults(self, config: PipelineConfig) -> None:
        """Apply environment-specific default settings.
        
        Args:
            config: Configuration to update
        """
        if self.current_env == Environment.DEVELOPMENT:
            # Development: verbose logging, more workers
            config.storage.log_level = "DEBUG"
            config.monitoring.enabled = True
            config.monitoring.health_check_interval = 10.0
            
        elif self.current_env == Environment.TESTING:
            # Testing: minimal resources, fast startup
            config.face_detection.worker_count = 1
            config.face_recognition.worker_count = 1
            config.motion_detection.enabled = False
            config.notifications.enabled = False
            config.storage.log_level = "WARNING"
            
        elif self.current_env == Environment.STAGING:
            # Staging: production-like but with more logging
            config.storage.log_level = "INFO"
            config.monitoring.enabled = True
            config.storage.backup_enabled = True
            
        elif self.current_env == Environment.PRODUCTION:
            # Production: optimized for reliability
            config.storage.log_level = "INFO"
            config.monitoring.enabled = True
            config.storage.backup_enabled = True
            config.motion_detection.enabled = True
            config.notifications.rate_limiting['enabled'] = True
            
            # Ensure reasonable resource limits
            if config.face_detection.worker_count > 4:
                config.face_detection.worker_count = 4
            if config.face_recognition.worker_count > 3:
                config.face_recognition.worker_count = 3
    
    def _validate_environment_config(self, config: PipelineConfig) -> None:
        """Validate environment-specific configuration constraints.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration violates environment constraints
        """
        if self.current_env == Environment.PRODUCTION:
            # Production validation
            if config.storage.log_level == "DEBUG":
                logger.warning("DEBUG logging enabled in production")
            
            if not config.storage.backup_enabled:
                logger.warning("Backups disabled in production")
            
            if not config.monitoring.enabled:
                raise ValueError("Monitoring must be enabled in production")
        
        elif self.current_env == Environment.TESTING:
            # Testing validation
            if config.notifications.enabled:
                logger.warning("Notifications enabled in testing environment")
    
    def get_environment(self) -> Environment:
        """Get current environment.
        
        Returns:
            Environment: Current environment
        """
        return self.current_env
    
    def is_production(self) -> bool:
        """Check if running in production environment.
        
        Returns:
            bool: True if in production
        """
        return self.current_env == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment.
        
        Returns:
            bool: True if in development
        """
        return self.current_env == Environment.DEVELOPMENT
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about current environment.
        
        Returns:
            Dict[str, Any]: Environment information
        """
        return {
            'environment': self.current_env.value,
            'is_production': self.is_production(),
            'is_development': self.is_development(),
            'config_path': str(self.base_config_path),
            'env_vars': {
                'DOORBELL_ENV': os.getenv('DOORBELL_ENV'),
                'CI': os.getenv('CI'),
                'RAILWAY_ENVIRONMENT': os.getenv('RAILWAY_ENVIRONMENT'),
                'RENDER': os.getenv('RENDER'),
                'VERCEL_ENV': os.getenv('VERCEL_ENV'),
            }
        }


def load_config_for_environment(
    environment: Optional[Environment] = None
) -> PipelineConfig:
    """Load configuration for specified environment.
    
    Args:
        environment: Optional environment to load config for
        
    Returns:
        PipelineConfig: Environment-specific configuration
    """
    # Temporarily override environment if specified
    original_env = os.getenv('DOORBELL_ENV')
    
    try:
        if environment:
            os.environ['DOORBELL_ENV'] = environment.value
        
        manager = EnvironmentManager()
        return manager.load_environment_config()
        
    finally:
        # Restore original environment
        if original_env:
            os.environ['DOORBELL_ENV'] = original_env
        elif 'DOORBELL_ENV' in os.environ:
            del os.environ['DOORBELL_ENV']


def get_current_environment() -> Environment:
    """Get current environment.
    
    Returns:
        Environment: Current environment
    """
    manager = EnvironmentManager()
    return manager.get_environment()
