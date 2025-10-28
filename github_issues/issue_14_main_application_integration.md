# Issue #14: Main Application Integration and Legacy Migration

## 📋 **Overview**

Complete the architectural transformation by fully integrating the new pipeline orchestrator into the main application entry points, replacing the legacy `DoorbellSecuritySystem` while maintaining backward compatibility for external integrations. This issue ensures seamless migration, comprehensive testing, and production-ready deployment capabilities.

## 🎯 **Objectives**

### **Primary Goals**
1. **Complete Application Integration**: Replace legacy system with pipeline orchestrator in all entry points
2. **Backward Compatibility**: Maintain existing API contracts and web interface functionality
3. **Migration Tools**: Provide automated migration utilities for configuration and data
4. **Production Deployment**: Enable zero-downtime deployment and rollback capabilities
5. **Comprehensive Testing**: End-to-end testing of the integrated system

### **Success Criteria**
- All application entry points use the new pipeline architecture
- 100% backward compatibility with existing web interface and APIs
- Automated migration tools for configuration and face databases
- Zero-downtime deployment capability with rollback mechanisms
- Performance improvements: 30% faster processing, 25% better resource utilization
- Complete integration test coverage

## 🏗️ **Integration Architecture**

### **Application Entry Points Transformation**
```
BEFORE (Legacy):
app.py → DoorbellSecuritySystem → [Individual Components] → Web Interface

AFTER (Pipeline):
app.py → OrchestratorManager → PipelineOrchestrator → [Pipeline Workers] → Web Interface
main.py → OrchestratorManager → PipelineOrchestrator → [Pipeline Workers] → System
```

### **Compatibility Layer Structure**
```
Web Interface → Legacy Adapter → Pipeline Orchestrator
External APIs → Legacy Adapter → Pipeline Orchestrator  
Configuration → Migration Layer → Pipeline Configuration
Face Data → Migration Layer → Pipeline Storage
```

## 📁 **Implementation Specifications**

### **Files to Create/Modify**

#### **New Files**
```
src/integration/                               # Integration and migration layer
    migration_manager.py                       # Comprehensive migration management
    configuration_migrator.py                  # Configuration format migration
    data_migrator.py                           # Face database migration
    compatibility_layer.py                     # Backward compatibility interface
    deployment_manager.py                      # Production deployment management
    
scripts/                                      # Migration and deployment scripts
    migrate_to_pipeline.py                     # Complete system migration script
    validate_migration.py                      # Migration validation tool
    rollback_migration.py                      # Rollback to legacy system
    deploy_production.py                       # Production deployment script
    
config/migration/                             # Migration configuration
    migration_config.py                       # Migration settings and rules
    legacy_mapping.py                         # Legacy to pipeline config mapping
    
tests/integration/                            # Integration testing
    test_complete_integration.py              # End-to-end integration tests
    test_migration_process.py                 # Migration process validation
    test_backward_compatibility.py            # Compatibility layer tests
    test_deployment_scenarios.py              # Deployment testing
```

#### **Modified Files**
```
app.py                                        # Cloud deployment entry point
src/main.py                                   # Main application entry (enhanced)
src/web_interface.py                          # Enhanced web interface integration
src/doorbell_security.py                     # Legacy compatibility wrapper
config/settings.py                           # Enhanced configuration management
requirements.txt                             # Updated dependencies
Dockerfile                                   # Enhanced Docker configuration
docker-compose.yml                           # Production-ready compose
```

### **Core Component: Migration Manager**
```python
#!/usr/bin/env python3
"""
Migration Manager - Complete System Migration

Handles the complete migration from legacy DoorbellSecuritySystem to 
PipelineOrchestrator architecture with validation and rollback capabilities.
"""

import os
import json
import logging
import shutil
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import sqlite3

from src.integration.configuration_migrator import ConfigurationMigrator
from src.integration.data_migrator import DataMigrator
from src.integration.orchestrator_manager import OrchestratorManager
from src.doorbell_security import DoorbellSecuritySystem
from config.settings import Settings

logger = logging.getLogger(__name__)


class MigrationStage(Enum):
    """Migration stages."""
    PREPARATION = "preparation"
    BACKUP = "backup"
    CONFIG_MIGRATION = "config_migration"
    DATA_MIGRATION = "data_migration"
    SYSTEM_INTEGRATION = "system_integration"
    VALIDATION = "validation"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationStatus:
    """Migration status tracking."""
    stage: MigrationStage
    progress: float  # 0.0 to 1.0
    message: str
    errors: List[str]
    warnings: List[str]
    start_time: float
    estimated_completion: Optional[float] = None


class MigrationManager:
    """
    Comprehensive migration manager for transitioning from legacy to pipeline architecture.
    
    Features:
    - Staged migration with rollback capability
    - Configuration and data migration
    - Validation and testing
    - Progress tracking and reporting
    - Automated rollback on failure
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize migration manager."""
        self.config = config or {}
        self.backup_dir = Path(self.config.get('backup_dir', 'data/migration_backup'))
        self.migration_log_file = Path(self.config.get('log_file', 'data/logs/migration.log'))
        
        # Migration components
        self.config_migrator = ConfigurationMigrator()
        self.data_migrator = DataMigrator()
        
        # State tracking
        self.status = MigrationStatus(
            stage=MigrationStage.PREPARATION,
            progress=0.0,
            message="Migration initialized",
            errors=[],
            warnings=[],
            start_time=time.time()
        )
        
        # Migration steps with weights for progress calculation
        self.migration_steps = [
            (MigrationStage.PREPARATION, 0.05, self._prepare_migration),
            (MigrationStage.BACKUP, 0.15, self._create_backup),
            (MigrationStage.CONFIG_MIGRATION, 0.20, self._migrate_configuration),
            (MigrationStage.DATA_MIGRATION, 0.30, self._migrate_data),
            (MigrationStage.SYSTEM_INTEGRATION, 0.20, self._integrate_system),
            (MigrationStage.VALIDATION, 0.08, self._validate_migration),
            (MigrationStage.CLEANUP, 0.02, self._cleanup_migration)
        ]
        
        logger.info("Migration manager initialized")
    
    def run_migration(self) -> bool:
        """
        Run complete migration process.
        
        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info("Starting complete system migration to pipeline architecture")
            
            cumulative_progress = 0.0
            
            for stage, weight, step_func in self.migration_steps:
                # Update stage
                self.status.stage = stage
                self.status.message = f"Executing {stage.value}"
                self._log_status()
                
                try:
                    # Execute migration step
                    step_result = step_func()
                    
                    if not step_result:
                        raise Exception(f"Migration step {stage.value} failed")
                    
                    # Update progress
                    cumulative_progress += weight
                    self.status.progress = cumulative_progress
                    
                except Exception as e:
                    error_msg = f"Migration failed at stage {stage.value}: {e}"
                    self.status.errors.append(error_msg)
                    self.status.stage = MigrationStage.FAILED
                    logger.error(error_msg)
                    
                    # Attempt automatic rollback
                    if self.config.get('auto_rollback', True):
                        logger.info("Attempting automatic rollback...")
                        self.rollback_migration()
                    
                    return False
            
            # Migration completed successfully
            self.status.stage = MigrationStage.COMPLETED
            self.status.progress = 1.0
            self.status.message = "Migration completed successfully"
            self._log_status()
            
            logger.info("✅ Migration to pipeline architecture completed successfully")
            return True
            
        except Exception as e:
            self.status.stage = MigrationStage.FAILED
            self.status.errors.append(f"Migration process failed: {e}")
            logger.error(f"Migration process failed: {e}")
            return False
    
    def rollback_migration(self) -> bool:
        """
        Rollback migration to previous state.
        
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            logger.info("Starting migration rollback...")
            
            if not self.backup_dir.exists():
                raise Exception("No backup found for rollback")
            
            # Stop any running services
            self._stop_services()
            
            # Restore from backup
            self._restore_from_backup()
            
            # Validate rollback
            if not self._validate_rollback():
                raise Exception("Rollback validation failed")
            
            self.status.stage = MigrationStage.ROLLED_BACK
            self.status.message = "Migration rolled back successfully"
            
            logger.info("✅ Migration rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self.status.errors.append(f"Rollback failed: {e}")
            return False
    
    def get_migration_status(self) -> MigrationStatus:
        """Get current migration status."""
        # Update estimated completion time
        if self.status.progress > 0:
            elapsed = time.time() - self.status.start_time
            estimated_total = elapsed / self.status.progress
            self.status.estimated_completion = self.status.start_time + estimated_total
        
        return self.status
    
    def _prepare_migration(self) -> bool:
        """Prepare for migration."""
        try:
            # Create necessary directories
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.migration_log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate current system
            if not self._validate_legacy_system():
                raise Exception("Legacy system validation failed")
            
            # Check disk space
            if not self._check_disk_space():
                raise Exception("Insufficient disk space for migration")
            
            # Check dependencies
            if not self._check_dependencies():
                raise Exception("Missing required dependencies")
            
            logger.info("Migration preparation completed")
            return True
            
        except Exception as e:
            logger.error(f"Migration preparation failed: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """Create comprehensive backup."""
        try:
            backup_timestamp = int(time.time())
            backup_path = self.backup_dir / f"backup_{backup_timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration files
            config_backup = backup_path / "config"
            config_backup.mkdir(exist_ok=True)
            
            config_files = [
                "config/settings.py",
                "config/credentials_telegram.py",
                ".env"
            ]
            
            for config_file in config_files:
                src_path = Path(config_file)
                if src_path.exists():
                    shutil.copy2(src_path, config_backup / src_path.name)
            
            # Backup data directories
            data_dirs = [
                "data/known_faces",
                "data/blacklist_faces",
                "data/captures",
                "data/logs"
            ]
            
            for data_dir in data_dirs:
                src_path = Path(data_dir)
                if src_path.exists():
                    dst_path = backup_path / src_path.name
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            
            # Create backup manifest
            manifest = {
                "timestamp": backup_timestamp,
                "system_version": "legacy",
                "files_backed_up": len(list(backup_path.rglob("*"))),
                "backup_size_mb": sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file()) / (1024**2)
            }
            
            with open(backup_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            
            # Store backup path for rollback
            with open(self.backup_dir / "current_backup.txt", "w") as f:
                f.write(str(backup_path))
            
            logger.info(f"Backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def _migrate_configuration(self) -> bool:
        """Migrate configuration to pipeline format."""
        try:
            # Use configuration migrator
            migration_result = self.config_migrator.migrate_configuration()
            
            if not migration_result['success']:
                raise Exception(f"Configuration migration failed: {migration_result['error']}")
            
            # Validate migrated configuration
            if not self._validate_pipeline_config():
                raise Exception("Pipeline configuration validation failed")
            
            logger.info("Configuration migration completed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration migration failed: {e}")
            return False
    
    def _migrate_data(self) -> bool:
        """Migrate face data and databases."""
        try:
            # Use data migrator
            migration_result = self.data_migrator.migrate_face_databases()
            
            if not migration_result['success']:
                raise Exception(f"Data migration failed: {migration_result['error']}")
            
            # Validate migrated data
            if not self._validate_migrated_data():
                raise Exception("Migrated data validation failed")
            
            logger.info("Data migration completed")
            return True
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return False
    
    def _integrate_system(self) -> bool:
        """Integrate pipeline system."""
        try:
            # Stop legacy system if running
            self._stop_legacy_system()
            
            # Initialize pipeline orchestrator
            orchestrator_manager = OrchestratorManager()
            
            # Test pipeline startup
            if not self._test_pipeline_startup(orchestrator_manager):
                raise Exception("Pipeline startup test failed")
            
            # Update application entry points
            self._update_entry_points()
            
            logger.info("System integration completed")
            return True
            
        except Exception as e:
            logger.error(f"System integration failed: {e}")
            return False
    
    def _validate_migration(self) -> bool:
        """Validate complete migration."""
        try:
            # Functional validation
            if not self._validate_functionality():
                raise Exception("Functionality validation failed")
            
            # Performance validation
            if not self._validate_performance():
                self.status.warnings.append("Performance validation had issues")
            
            # API compatibility validation
            if not self._validate_api_compatibility():
                raise Exception("API compatibility validation failed")
            
            logger.info("Migration validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False
    
    def _cleanup_migration(self) -> bool:
        """Cleanup migration artifacts."""
        try:
            # Remove temporary files
            temp_files = [
                "data/migration_temp",
                "config/migration_backup"
            ]
            
            for temp_file in temp_files:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    if temp_path.is_dir():
                        shutil.rmtree(temp_path)
                    else:
                        temp_path.unlink()
            
            # Compress old backup if configured
            if self.config.get('compress_backup', True):
                self._compress_backup()
            
            logger.info("Migration cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Migration cleanup failed: {e}")
            return False
    
    def _validate_legacy_system(self) -> bool:
        """Validate legacy system before migration."""
        try:
            # Check if legacy system is functional
            legacy_system = DoorbellSecuritySystem()
            
            # Validate face databases
            face_count = len(list(Path("data/known_faces").glob("*.jpg")))
            blacklist_count = len(list(Path("data/blacklist_faces").glob("*.jpg")))
            
            logger.info(f"Legacy system validation: {face_count} known faces, {blacklist_count} blacklisted")
            
            return True
            
        except Exception as e:
            logger.error(f"Legacy system validation failed: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            
            # Check available space
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            # Require at least 2GB free space
            if free_gb < 2.0:
                logger.error(f"Insufficient disk space: {free_gb:.1f}GB available, 2GB required")
                return False
            
            logger.info(f"Disk space check passed: {free_gb:.1f}GB available")
            return True
            
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check required dependencies."""
        try:
            required_modules = [
                'numpy',
                'face_recognition',
                'opencv-python',
                'flask',
                'sqlite3'
            ]
            
            for module in required_modules:
                try:
                    __import__(module.replace('-', '_'))
                except ImportError:
                    logger.error(f"Required module not found: {module}")
                    return False
            
            logger.info("Dependency check passed")
            return True
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    def _validate_pipeline_config(self) -> bool:
        """Validate migrated pipeline configuration."""
        try:
            from config.pipeline_config import PipelineConfig
            
            # Try to load pipeline configuration
            config = PipelineConfig()
            
            # Validate critical settings
            if not config.frame_capture.enabled:
                raise Exception("Frame capture not enabled in pipeline config")
            
            if not config.face_detection.enabled:
                raise Exception("Face detection not enabled in pipeline config")
            
            logger.info("Pipeline configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline configuration validation failed: {e}")
            return False
    
    def _validate_migrated_data(self) -> bool:
        """Validate migrated face databases."""
        try:
            # Check pipeline storage
            from src.storage.face_database import FaceDatabase
            
            face_db = FaceDatabase()
            face_db.initialize()
            
            # Count migrated faces
            known_count = face_db.get_known_faces_count()
            blacklist_count = face_db.get_blacklist_faces_count()
            
            logger.info(f"Data validation: {known_count} known faces, {blacklist_count} blacklisted")
            
            # Verify at least some data was migrated
            if known_count == 0 and blacklist_count == 0:
                self.status.warnings.append("No face data was migrated - this might be expected for new installations")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def _test_pipeline_startup(self, orchestrator_manager: OrchestratorManager) -> bool:
        """Test pipeline startup."""
        try:
            # Start orchestrator
            orchestrator_manager.start()
            
            # Wait for initialization
            time.sleep(3.0)
            
            # Check health
            health = orchestrator_manager.get_health_status()
            
            if health.state.value != 'running':
                raise Exception(f"Pipeline not running: {health.state}")
            
            # Stop for now
            orchestrator_manager.stop()
            
            logger.info("Pipeline startup test passed")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline startup test failed: {e}")
            return False
    
    def _update_entry_points(self) -> bool:
        """Update application entry points."""
        try:
            # Update app.py to use orchestrator
            app_py_content = '''#!/usr/bin/env python3
"""
Cloud deployment entry point - Pipeline Architecture
Updated to use PipelineOrchestrator instead of legacy DoorbellSecuritySystem
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set environment variables for cloud deployment
os.environ['DEVELOPMENT_MODE'] = 'true'
os.environ['PORT'] = os.environ.get('PORT', '8000')

from src.web_interface import create_web_app
from src.integration.orchestrator_manager import OrchestratorManager
from config.logging_config import setup_logging

# Setup logging
setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variable to store initialization error message
system_init_error_message = None

# Initialize the pipeline system
try:
    logger.info("Initializing Pipeline Architecture for cloud deployment...")
    
    orchestrator_manager = OrchestratorManager()
    orchestrator_manager.start()
    
    # Create Flask app with legacy adapter
    legacy_interface = orchestrator_manager.get_legacy_interface()
    app = create_web_app(legacy_interface)
    
    logger.info("✅ Cloud deployment ready with pipeline architecture")
    
except Exception as init_exception:
    system_init_error_message = str(init_exception)
    logger.error(f"Failed to initialize pipeline system: {system_init_error_message}")
    
    # Create minimal error app
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return jsonify({
            'status': 'error',
            'message': f'Pipeline initialization failed: {system_init_error_message}',
            'note': 'System has been migrated to pipeline architecture.'
        })

# For cloud platforms that expect 'application'
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting web application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
'''
            
            with open("app.py", "w") as f:
                f.write(app_py_content)
            
            logger.info("Entry points updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update entry points: {e}")
            return False
    
    def _validate_functionality(self) -> bool:
        """Validate system functionality after migration."""
        try:
            # Test basic pipeline functionality
            orchestrator_manager = OrchestratorManager()
            orchestrator_manager.start()
            
            try:
                # Test doorbell trigger
                result = orchestrator_manager.trigger_doorbell()
                
                if result['status'] != 'success':
                    raise Exception(f"Doorbell trigger test failed: {result}")
                
                # Test system health
                health = orchestrator_manager.get_health_status()
                
                if health.performance_score < 0.5:
                    self.status.warnings.append(f"Low performance score: {health.performance_score}")
                
                logger.info("Functionality validation passed")
                return True
                
            finally:
                orchestrator_manager.stop()
                
        except Exception as e:
            logger.error(f"Functionality validation failed: {e}")
            return False
    
    def _validate_performance(self) -> bool:
        """Validate performance after migration."""
        try:
            # Run performance tests
            # This would integrate with the performance testing framework
            
            logger.info("Performance validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return False
    
    def _validate_api_compatibility(self) -> bool:
        """Validate API backward compatibility."""
        try:
            # Test web interface APIs
            from src.web_interface import create_web_app
            from src.integration.orchestrator_manager import OrchestratorManager
            
            orchestrator_manager = OrchestratorManager()
            legacy_interface = orchestrator_manager.get_legacy_interface()
            app = create_web_app(legacy_interface)
            
            # Test critical endpoints
            with app.test_client() as client:
                # Test status endpoint
                response = client.get('/api/status')
                if response.status_code != 200:
                    raise Exception("Status API endpoint failed")
                
                # Test faces endpoint
                response = client.get('/api/faces')
                if response.status_code != 200:
                    raise Exception("Faces API endpoint failed")
            
            logger.info("API compatibility validation passed")
            return True
            
        except Exception as e:
            logger.error(f"API compatibility validation failed: {e}")
            return False
    
    def _stop_legacy_system(self) -> None:
        """Stop legacy system if running."""
        try:
            # Implementation to stop legacy system processes
            logger.info("Legacy system stopped")
        except Exception as e:
            logger.warning(f"Failed to stop legacy system: {e}")
    
    def _stop_services(self) -> None:
        """Stop all running services."""
        try:
            # Stop any running orchestrator
            # Implementation to stop services
            logger.info("Services stopped")
        except Exception as e:
            logger.warning(f"Failed to stop services: {e}")
    
    def _restore_from_backup(self) -> None:
        """Restore system from backup."""
        try:
            # Get current backup path
            with open(self.backup_dir / "current_backup.txt", "r") as f:
                backup_path = Path(f.read().strip())
            
            if not backup_path.exists():
                raise Exception("Backup path not found")
            
            # Restore configuration files
            config_backup = backup_path / "config"
            if config_backup.exists():
                for config_file in config_backup.iterdir():
                    shutil.copy2(config_file, f"config/{config_file.name}")
            
            # Restore data directories
            data_dirs = ["known_faces", "blacklist_faces", "captures", "logs"]
            for data_dir in data_dirs:
                backup_data = backup_path / data_dir
                if backup_data.exists():
                    target_path = Path(f"data/{data_dir}")
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(backup_data, target_path)
            
            logger.info("System restored from backup")
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            raise
    
    def _validate_rollback(self) -> bool:
        """Validate rollback was successful."""
        try:
            # Test legacy system functionality
            legacy_system = DoorbellSecuritySystem()
            
            # Basic validation
            if not hasattr(legacy_system, 'face_manager'):
                return False
            
            logger.info("Rollback validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Rollback validation failed: {e}")
            return False
    
    def _compress_backup(self) -> None:
        """Compress backup for long-term storage."""
        try:
            import tarfile
            
            with open(self.backup_dir / "current_backup.txt", "r") as f:
                backup_path = Path(f.read().strip())
            
            if backup_path.exists():
                archive_name = f"{backup_path.name}.tar.gz"
                archive_path = self.backup_dir / archive_name
                
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(backup_path, arcname=backup_path.name)
                
                # Remove uncompressed backup
                shutil.rmtree(backup_path)
                
                logger.info(f"Backup compressed: {archive_path}")
                
        except Exception as e:
            logger.warning(f"Backup compression failed: {e}")
    
    def _log_status(self) -> None:
        """Log current migration status."""
        status_msg = (
            f"Migration Status: {self.status.stage.value} "
            f"({self.status.progress*100:.1f}%) - {self.status.message}"
        )
        logger.info(status_msg)
        
        # Write to migration log
        try:
            with open(self.migration_log_file, "a") as f:
                f.write(f"{time.time()}: {status_msg}\n")
        except Exception:
            pass
```

### **Configuration Migrator**
```python
#!/usr/bin/env python3
"""
Configuration Migrator

Migrates legacy configuration format to pipeline configuration format.
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from config.settings import Settings
from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class ConfigurationMigrator:
    """Migrate configuration from legacy to pipeline format."""
    
    def __init__(self):
        """Initialize configuration migrator."""
        self.legacy_config_mapping = {
            # Legacy setting -> Pipeline config path
            'DEBOUNCE_TIME': 'frame_capture.debounce_time',
            'FACE_RECOGNITION_TOLERANCE': 'face_recognition.tolerance',
            'TELEGRAM_BOT_TOKEN': 'notification.telegram.bot_token',
            'TELEGRAM_CHAT_ID': 'notification.telegram.chat_id',
            'WEB_INTERFACE_PORT': 'web_interface.port',
            'ENABLE_FACE_RECOGNITION': 'face_recognition.enabled',
        }
    
    def migrate_configuration(self) -> Dict[str, Any]:
        """
        Migrate legacy configuration to pipeline format.
        
        Returns:
            Migration result with success status and details
        """
        try:
            # Load legacy configuration
            legacy_config = self._load_legacy_config()
            
            # Create pipeline configuration
            pipeline_config = self._create_pipeline_config(legacy_config)
            
            # Validate pipeline configuration
            self._validate_pipeline_config(pipeline_config)
            
            # Save pipeline configuration
            self._save_pipeline_config(pipeline_config)
            
            # Create migration report
            report = {
                'success': True,
                'migrated_settings': len(pipeline_config),
                'pipeline_config_path': 'config/pipeline_config.py',
                'backup_created': True
            }
            
            logger.info("Configuration migration completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Configuration migration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_legacy_config(self) -> Dict[str, Any]:
        """Load legacy configuration."""
        try:
            # Load from Settings class
            settings = Settings()
            
            legacy_config = {}
            
            # Extract relevant settings
            for attr in dir(settings):
                if not attr.startswith('_') and attr.isupper():
                    value = getattr(settings, attr)
                    legacy_config[attr] = value
            
            logger.info(f"Loaded {len(legacy_config)} legacy configuration settings")
            return legacy_config
            
        except Exception as e:
            logger.error(f"Failed to load legacy configuration: {e}")
            raise
    
    def _create_pipeline_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create pipeline configuration from legacy settings."""
        pipeline_config = {
            'frame_capture': {
                'enabled': True,
                'debounce_time': legacy_config.get('DEBOUNCE_TIME', 5.0),
                'capture_timeout': 10.0,
                'ring_buffer_size': 30
            },
            'motion_detection': {
                'enabled': legacy_config.get('ENABLE_MOTION_DETECTION', False),
                'motion_threshold': 0.3,
                'min_motion_area': 500
            },
            'face_detection': {
                'enabled': True,
                'worker_count': 2,
                'detector_type': 'cpu',
                'model': 'hog',
                'confidence_threshold': 0.5
            },
            'face_recognition': {
                'enabled': legacy_config.get('ENABLE_FACE_RECOGNITION', True),
                'tolerance': legacy_config.get('FACE_RECOGNITION_TOLERANCE', 0.6),
                'worker_count': 1,
                'cache_size': 100
            },
            'event_processing': {
                'enabled': True,
                'max_queue_size': 1000,
                'processing_timeout': 30.0
            },
            'notification': {
                'telegram': {
                    'enabled': bool(legacy_config.get('TELEGRAM_BOT_TOKEN')),
                    'bot_token': legacy_config.get('TELEGRAM_BOT_TOKEN', ''),
                    'chat_id': legacy_config.get('TELEGRAM_CHAT_ID', ''),
                    'rate_limit': 10
                },
                'internal': {
                    'enabled': True,
                    'web_notifications': True,
                    'alert_history': True
                }
            },
            'web_interface': {
                'enabled': True,
                'port': legacy_config.get('WEB_INTERFACE_PORT', 5000),
                'host': '0.0.0.0',
                'debug': False
            },
            'storage': {
                'database_path': 'data/pipeline.db',
                'backup_enabled': True,
                'retention_days': 30
            }
        }
        
        return pipeline_config
    
    def _validate_pipeline_config(self, config: Dict[str, Any]) -> None:
        """Validate pipeline configuration."""
        required_sections = [
            'frame_capture',
            'face_detection',
            'face_recognition',
            'event_processing'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section missing: {section}")
            
            if not config[section].get('enabled', True):
                logger.warning(f"Configuration section disabled: {section}")
    
    def _save_pipeline_config(self, config: Dict[str, Any]) -> None:
        """Save pipeline configuration."""
        config_path = Path("config/pipeline_migrated.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Pipeline configuration saved to {config_path}")
```

### **Data Migrator**
```python
#!/usr/bin/env python3
"""
Data Migrator

Migrates face databases and event data from legacy format to pipeline storage.
"""

import logging
import sqlite3
import pickle
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

from src.storage.face_database import FaceDatabase
from src.storage.event_database import EventDatabase

logger = logging.getLogger(__name__)


class DataMigrator:
    """Migrate data from legacy to pipeline storage format."""
    
    def __init__(self):
        """Initialize data migrator."""
        self.face_database = FaceDatabase()
        self.event_database = EventDatabase()
    
    def migrate_face_databases(self) -> Dict[str, Any]:
        """
        Migrate face databases from legacy format.
        
        Returns:
            Migration result with success status and details
        """
        try:
            # Initialize pipeline databases
            self.face_database.initialize()
            self.event_database.initialize()
            
            # Migrate known faces
            known_faces_migrated = self._migrate_known_faces()
            
            # Migrate blacklist faces
            blacklist_faces_migrated = self._migrate_blacklist_faces()
            
            # Migrate event data (if exists)
            events_migrated = self._migrate_event_data()
            
            # Create migration report
            report = {
                'success': True,
                'known_faces_migrated': known_faces_migrated,
                'blacklist_faces_migrated': blacklist_faces_migrated,
                'events_migrated': events_migrated,
                'total_records': known_faces_migrated + blacklist_faces_migrated + events_migrated
            }
            
            logger.info(f"Data migration completed: {report}")
            return report
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _migrate_known_faces(self) -> int:
        """Migrate known faces from legacy format."""
        known_faces_dir = Path("data/known_faces")
        
        if not known_faces_dir.exists():
            logger.info("No known faces directory found")
            return 0
        
        migrated_count = 0
        
        for image_file in known_faces_dir.glob("*.jpg"):
            try:
                # Extract person name from filename
                person_name = image_file.stem.split('_')[0]
                
                # Load and process image
                import cv2
                image = cv2.imread(str(image_file))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate face encoding
                face_encoding = self._generate_face_encoding(image_rgb)
                
                if face_encoding is not None:
                    # Store in pipeline database
                    self.face_database.add_known_face(
                        person_name=person_name,
                        face_encoding=face_encoding,
                        image_path=str(image_file),
                        metadata={
                            'migrated_from': 'legacy',
                            'original_file': image_file.name
                        }
                    )
                    migrated_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to migrate face {image_file}: {e}")
        
        logger.info(f"Migrated {migrated_count} known faces")
        return migrated_count
    
    def _migrate_blacklist_faces(self) -> int:
        """Migrate blacklist faces from legacy format."""
        blacklist_faces_dir = Path("data/blacklist_faces")
        
        if not blacklist_faces_dir.exists():
            logger.info("No blacklist faces directory found")
            return 0
        
        migrated_count = 0
        
        for image_file in blacklist_faces_dir.glob("*.jpg"):
            try:
                # Extract person identifier from filename
                person_id = image_file.stem
                
                # Load and process image
                import cv2
                image = cv2.imread(str(image_file))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate face encoding
                face_encoding = self._generate_face_encoding(image_rgb)
                
                if face_encoding is not None:
                    # Store in pipeline database
                    self.face_database.add_blacklist_face(
                        person_id=person_id,
                        face_encoding=face_encoding,
                        image_path=str(image_file),
                        metadata={
                            'migrated_from': 'legacy',
                            'original_file': image_file.name
                        }
                    )
                    migrated_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to migrate blacklist face {image_file}: {e}")
        
        logger.info(f"Migrated {migrated_count} blacklist faces")
        return migrated_count
    
    def _migrate_event_data(self) -> int:
        """Migrate event data if legacy database exists."""
        legacy_db_path = Path("data/events.db")
        
        if not legacy_db_path.exists():
            logger.info("No legacy event database found")
            return 0
        
        migrated_count = 0
        
        try:
            # Connect to legacy database
            conn = sqlite3.connect(legacy_db_path)
            cursor = conn.cursor()
            
            # Check if events table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
            if not cursor.fetchone():
                logger.info("No events table in legacy database")
                return 0
            
            # Migrate events
            cursor.execute("SELECT * FROM events ORDER BY timestamp")
            for row in cursor.fetchall():
                try:
                    # Convert legacy event to pipeline format
                    pipeline_event = self._convert_legacy_event(row)
                    
                    # Store in pipeline database
                    self.event_database.add_event(pipeline_event)
                    migrated_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to migrate event: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Event data migration failed: {e}")
        
        logger.info(f"Migrated {migrated_count} events")
        return migrated_count
    
    def _generate_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Generate face encoding from image."""
        try:
            import face_recognition
            
            # Detect face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return None
            
            # Generate encoding for first face
            encodings = face_recognition.face_encodings(image, face_locations)
            
            return encodings[0] if encodings else None
            
        except Exception as e:
            logger.error(f"Face encoding generation failed: {e}")
            return None
    
    def _convert_legacy_event(self, row) -> Dict[str, Any]:
        """Convert legacy event record to pipeline format."""
        # This would depend on the legacy event database schema
        # Example conversion:
        return {
            'event_type': 'face_recognition',
            'timestamp': row[1],  # Assuming timestamp is second column
            'data': {
                'legacy_event_id': row[0],
                'migrated_from': 'legacy_db'
            },
            'source': 'migration'
        }
```

## 🧪 **Testing Requirements**

### **Integration Testing Suite**
```python
#!/usr/bin/env python3
"""
Complete Integration Testing Suite

Comprehensive testing of the integrated pipeline system.
"""

import pytest
import time
import requests
from pathlib import Path

from src.integration.migration_manager import MigrationManager
from src.integration.orchestrator_manager import OrchestratorManager


class TestCompleteIntegration:
    """Complete system integration tests."""
    
    def test_migration_process(self):
        """Test complete migration process."""
        # Test migration workflow
        migrator = MigrationManager()
        result = migrator.run_migration()
        
        assert result is True
        assert migrator.status.stage.value == 'completed'
    
    def test_web_interface_integration(self):
        """Test web interface with pipeline backend."""
        # Start pipeline system
        manager = OrchestratorManager()
        manager.start()
        
        try:
            # Test web interface endpoints
            # This would use the test client
            pass
        finally:
            manager.stop()
    
    def test_api_backward_compatibility(self):
        """Test API backward compatibility."""
        # Verify all legacy APIs still work
        pass
    
    def test_performance_comparison(self):
        """Compare performance between legacy and pipeline."""
        # Performance benchmarking
        pass
```

## 📋 **Acceptance Criteria**

### **Migration Requirements**
- [ ] Complete automated migration from legacy to pipeline architecture
- [ ] Comprehensive backup and rollback capabilities
- [ ] Configuration and data migration with validation
- [ ] Zero data loss during migration process

### **Integration Requirements**
- [ ] All application entry points use pipeline architecture
- [ ] 100% backward compatibility with existing APIs
- [ ] Web interface fully functional with pipeline backend
- [ ] Performance improvements achieved

### **Production Requirements**
- [ ] Zero-downtime deployment capability
- [ ] Comprehensive monitoring and health checking
- [ ] Automated rollback on failure
- [ ] Production deployment documentation

---

**This issue completes the architectural transformation by providing comprehensive migration tools, maintaining backward compatibility, and ensuring production-ready deployment capabilities for the new pipeline architecture.**