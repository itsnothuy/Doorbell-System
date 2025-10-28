#!/usr/bin/env python3
"""
Advanced Configuration Validation System

Comprehensive validation with schema support and dependency checking.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    path: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of validation error."""
        result = f"[{self.severity.upper()}] {self.path}: {self.message}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def __str__(self) -> str:
        """String representation of validation result."""
        lines = []
        if self.is_valid:
            lines.append("✓ Configuration is valid")
        else:
            lines.append("✗ Configuration validation failed")
        
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  {error}")
        
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  {warning}")
        
        return "\n".join(lines)


class ConfigValidator:
    """Advanced configuration validator with schema support."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize configuration validator.
        
        Args:
            schema_path: Optional path to JSON schema directory
        """
        self.schema_path = schema_path or Path(__file__).parent / "schemas"
        self.schemas: Dict[str, Any] = {}
        
        # Load schemas if available
        if JSONSCHEMA_AVAILABLE and self.schema_path.exists():
            self._load_schemas()
        
        # Dependency rules
        self.dependency_rules = self._load_dependency_rules()
    
    def validate_full_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Perform comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult: Result of validation
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        
        # Schema validation
        if JSONSCHEMA_AVAILABLE and self.schemas:
            schema_errors = self._validate_schema(config)
            errors.extend(schema_errors)
        
        # Dependency validation
        dep_errors, dep_warnings = self._validate_dependencies(config)
        errors.extend(dep_errors)
        warnings.extend(dep_warnings)
        
        # Hardware compatibility validation
        hw_errors, hw_warnings = self._validate_hardware_compatibility(config)
        errors.extend(hw_errors)
        warnings.extend(hw_warnings)
        
        # Performance validation
        perf_warnings = self._validate_performance_settings(config)
        warnings.extend(perf_warnings)
        
        # Resource validation
        resource_warnings = self._validate_resource_settings(config)
        warnings.extend(resource_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _load_schemas(self) -> None:
        """Load JSON schemas from schema directory."""
        try:
            import json
            
            if not self.schema_path.exists():
                logger.debug(f"Schema directory not found: {self.schema_path}")
                return
            
            for schema_file in self.schema_path.glob("*.json"):
                try:
                    with open(schema_file, 'r') as f:
                        schema_name = schema_file.stem
                        self.schemas[schema_name] = json.load(f)
                    logger.debug(f"Loaded schema: {schema_name}")
                except Exception as e:
                    logger.warning(f"Failed to load schema {schema_file}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load schemas: {e}")
    
    def _load_dependency_rules(self) -> Dict[str, Any]:
        """Load configuration dependency rules.
        
        Returns:
            Dict[str, Any]: Dependency rules
        """
        return {
            'gpu_detector_requires_cuda': {
                'condition': lambda cfg: cfg.get('face_detection', {}).get('detector_type') == 'gpu',
                'requirement': lambda cfg: cfg.get('hardware', {}).get('gpu_available', False),
                'error_message': 'GPU detector requires GPU hardware',
                'suggestion': 'Install CUDA or switch to CPU detector'
            },
            'edgetpu_detector_requires_hardware': {
                'condition': lambda cfg: cfg.get('face_detection', {}).get('detector_type') == 'edgetpu',
                'requirement': lambda cfg: cfg.get('hardware', {}).get('edgetpu_available', False),
                'error_message': 'EdgeTPU detector requires Coral EdgeTPU hardware',
                'suggestion': 'Connect EdgeTPU or switch to CPU detector'
            },
            'telegram_requires_credentials': {
                'condition': lambda cfg: cfg.get('notifications', {}).get('telegram_enabled', False),
                'requirement': lambda cfg: bool(cfg.get('notifications', {}).get('telegram_bot_token')),
                'error_message': 'Telegram notifications require bot token',
                'suggestion': 'Configure TELEGRAM_BOT_TOKEN environment variable or disable telegram'
            }
        }
    
    def _validate_schema(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration against JSON schema.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List[ValidationError]: List of schema validation errors
        """
        errors: List[ValidationError] = []
        
        if not JSONSCHEMA_AVAILABLE:
            return errors
        
        try:
            # Load main schema
            main_schema = self.schemas.get('main_config')
            if main_schema:
                jsonschema.validate(config, main_schema)
        except jsonschema.ValidationError as e:
            errors.append(ValidationError(
                path='.'.join(str(p) for p in e.absolute_path),
                message=e.message,
                severity='error',
                suggestion=self._get_schema_suggestion(e)
            ))
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
        
        return errors
    
    def _get_schema_suggestion(self, error: Any) -> Optional[str]:
        """Get suggestion for schema validation error.
        
        Args:
            error: Schema validation error
            
        Returns:
            Optional[str]: Suggestion text
        """
        if hasattr(error, 'validator'):
            if error.validator == 'type':
                return f"Expected type: {error.validator_value}"
            elif error.validator == 'minimum':
                return f"Value must be at least {error.validator_value}"
            elif error.validator == 'maximum':
                return f"Value must be at most {error.validator_value}"
            elif error.validator == 'required':
                return f"Required properties: {', '.join(error.validator_value)}"
        return None
    
    def _validate_dependencies(
        self,
        config: Dict[str, Any]
    ) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate configuration dependencies.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        
        # Check dependency rules
        for rule_name, rule in self.dependency_rules.items():
            try:
                if rule['condition'](config):
                    if not rule['requirement'](config):
                        errors.append(ValidationError(
                            path=rule_name,
                            message=rule['error_message'],
                            severity='error',
                            suggestion=rule.get('suggestion')
                        ))
            except Exception as e:
                logger.warning(f"Error checking dependency rule {rule_name}: {e}")
        
        return errors, warnings
    
    def _validate_hardware_compatibility(
        self,
        config: Dict[str, Any]
    ) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate hardware compatibility settings.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        
        # Check worker counts
        face_detection = config.get('face_detection', {})
        worker_count = face_detection.get('worker_count', 1)
        
        if worker_count > 8:
            warnings.append(ValidationError(
                path='face_detection.worker_count',
                message=f'High worker count ({worker_count}) may cause resource contention',
                severity='warning',
                suggestion='Consider reducing worker_count for better stability'
            ))
        
        # Check queue sizes
        max_queue_size = face_detection.get('max_queue_size', 100)
        if max_queue_size > 1000:
            warnings.append(ValidationError(
                path='face_detection.max_queue_size',
                message=f'Large queue size ({max_queue_size}) may consume excessive memory',
                severity='warning',
                suggestion='Consider reducing max_queue_size'
            ))
        
        return errors, warnings
    
    def _validate_performance_settings(
        self,
        config: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate performance-related settings.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List[ValidationError]: List of performance warnings
        """
        warnings: List[ValidationError] = []
        
        # Check FPS settings
        frame_capture = config.get('frame_capture', {})
        fps = frame_capture.get('fps', 10)
        
        if fps > 30:
            warnings.append(ValidationError(
                path='frame_capture.fps',
                message=f'High FPS ({fps}) may impact performance',
                severity='warning',
                suggestion='Consider reducing FPS to 30 or below for better stability'
            ))
        
        # Check motion detection
        motion = config.get('motion_detection', {})
        if not motion.get('enabled', False):
            face_detection = config.get('face_detection', {})
            if face_detection.get('enabled', True):
                warnings.append(ValidationError(
                    path='motion_detection.enabled',
                    message='Motion detection disabled - all frames will be processed',
                    severity='info',
                    suggestion='Enable motion detection to reduce CPU usage'
                ))
        
        # Check timeout settings
        face_recognition = config.get('face_recognition', {})
        timeout = face_recognition.get('timeout', 15.0)
        
        if timeout < 5.0:
            warnings.append(ValidationError(
                path='face_recognition.timeout',
                message=f'Short timeout ({timeout}s) may cause frequent failures',
                severity='warning',
                suggestion='Consider increasing timeout to at least 10 seconds'
            ))
        
        return warnings
    
    def _validate_resource_settings(
        self,
        config: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate resource allocation settings.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List[ValidationError]: List of resource warnings
        """
        warnings: List[ValidationError] = []
        
        # Check total worker count
        face_detection = config.get('face_detection', {})
        face_recognition = config.get('face_recognition', {})
        
        total_workers = (
            face_detection.get('worker_count', 2) +
            face_recognition.get('worker_count', 2)
        )
        
        if total_workers > 16:
            warnings.append(ValidationError(
                path='worker_configuration',
                message=f'Total worker count ({total_workers}) is very high',
                severity='warning',
                suggestion='Reduce total workers to avoid resource exhaustion'
            ))
        
        # Check cache settings
        cache_config = face_recognition.get('cache', {})
        cache_size = cache_config.get('cache_size', 1000) if isinstance(cache_config, dict) else 1000
        
        if cache_size > 10000:
            warnings.append(ValidationError(
                path='face_recognition.cache.cache_size',
                message=f'Large cache size ({cache_size}) may use significant memory',
                severity='warning',
                suggestion='Consider reducing cache size'
            ))
        
        return warnings
    
    def validate_section(
        self,
        section_name: str,
        section_config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a specific configuration section.
        
        Args:
            section_name: Name of the configuration section
            section_config: Configuration dictionary for the section
            
        Returns:
            ValidationResult: Result of validation
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        
        # Basic type validation
        if not isinstance(section_config, dict):
            errors.append(ValidationError(
                path=section_name,
                message=f'Configuration section must be a dictionary, got {type(section_config).__name__}',
                severity='error',
                suggestion='Check configuration format'
            ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


def create_validator(schema_path: Optional[Path] = None) -> ConfigValidator:
    """Create a configuration validator instance.
    
    Args:
        schema_path: Optional path to JSON schema directory
        
    Returns:
        ConfigValidator: Configured validator instance
    """
    return ConfigValidator(schema_path=schema_path)
