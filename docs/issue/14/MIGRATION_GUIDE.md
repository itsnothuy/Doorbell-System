# Migration Guide: Legacy to Pipeline Architecture

This guide helps you migrate your Doorbell Security System from the legacy architecture to the new pipeline orchestrator architecture.

## 🎯 Overview

The migration process is automated and includes:
- Comprehensive backup creation
- Configuration and data migration
- System integration and validation
- Rollback capability if needed
- Zero-downtime deployment support

## 📋 Prerequisites

### System Requirements
- **Disk Space**: At least 1GB free
- **Python**: 3.11 or higher
- **Dependencies**: All requirements.txt packages installed

### Pre-Migration Checklist
- [ ] System backup created manually (recommended)
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Sufficient disk space available
- [ ] No critical processes running on the system

## 🚀 Migration Process

### Step 1: Validate System Compatibility

Check if your system is ready for migration:

```bash
python scripts/validate_migration.py
```

This will check:
- Pipeline health
- Migration compatibility
- Data integrity
- Dependencies

### Step 2: Perform Migration

Run the automated migration script:

```bash
python scripts/migrate_to_pipeline.py
```

The migration process includes 7 stages:

1. **Preparation** (~5s)
   - Validates current system
   - Checks disk space and dependencies
   - Creates necessary directories

2. **Backup** (~30s)
   - Backs up configuration files
   - Backs up data directories
   - Creates backup manifest
   - Stores backup location

3. **Configuration Migration** (~5s)
   - Maps legacy settings to pipeline format
   - Validates pipeline configuration
   - Sets up environment variables

4. **Data Migration** (~10s)
   - Ensures pipeline directories exist
   - Migrates known faces
   - Migrates blacklist faces
   - Preserves capture history

5. **System Integration** (~10s)
   - Stops legacy system (if running)
   - Tests pipeline startup
   - Verifies orchestrator health

6. **Validation** (~15s)
   - Functional validation
   - Performance validation
   - API compatibility validation

7. **Cleanup** (~2s)
   - Removes temporary files
   - Optional backup compression

**Total Time**: ~75 seconds

### Step 3: Validate Migration

After migration, validate that everything works:

```bash
python scripts/validate_migration.py
```

This performs comprehensive checks:
- Pipeline health verification
- Orchestrator startup test
- Web interface testing
- Data integrity check

### Step 4: Test the System

Start the system to ensure it's working:

```bash
# Test main application
python src/main.py

# Test web interface
python app.py
```

Access the web interface at `http://localhost:5000`

## 🔄 Rollback (If Needed)

If you encounter issues, you can rollback to the pre-migration state:

```bash
python scripts/rollback_migration.py
```

This will:
- Restore configuration files
- Restore data directories
- Restore event database
- Validate rollback success

⚠️ **Warning**: Rollback will restore to pre-migration state. Any changes made after migration will be lost.

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker-compose build

# Start the system
docker-compose up -d

# Check logs
docker-compose logs -f doorbell-security

# Stop the system
docker-compose down
```

### Production Deployment

For production with Nginx reverse proxy:

```bash
docker-compose --profile production up -d
```

## 🔍 Troubleshooting

### Migration Fails at Preparation Stage

**Issue**: Insufficient disk space or missing dependencies

**Solution**:
```bash
# Check disk space
df -h

# Install dependencies
pip install -r requirements.txt

# Retry migration
python scripts/migrate_to_pipeline.py
```

### Migration Fails at Backup Stage

**Issue**: Permission issues or I/O errors

**Solution**:
```bash
# Check directory permissions
ls -la data/

# Ensure you have write permissions
chmod -R u+w data/

# Retry migration
python scripts/migrate_to_pipeline.py
```

### Migration Fails at Integration Stage

**Issue**: Pipeline startup fails

**Solution**:
```bash
# Check logs
tail -f data/logs/migration.log

# Validate configuration
python -c "from config.pipeline_config import PipelineConfig; print('Config OK')"

# Check dependencies
python scripts/validate_migration.py
```

### Web Interface Not Working

**Issue**: Legacy adapter or web interface issues

**Solution**:
```bash
# Test legacy adapter
python -c "from src.integration.legacy_adapter import LegacyAdapter; print('Adapter OK')"

# Test web interface
curl http://localhost:5000/api/status

# Check logs
tail -f data/logs/doorbell.log
```

## 📊 Migration Logs

All migration activities are logged to:
- **Migration Log**: `data/logs/migration.log`
- **Application Log**: `data/logs/doorbell.log`

Check these logs for detailed information about the migration process.

## 🔧 Advanced Options

### Custom Backup Location

Edit the migration script to use a custom backup location:

```python
migration_config = {
    'backup_dir': '/path/to/custom/backup',
    'log_file': 'data/logs/migration.log',
    'auto_rollback': True,
    'compress_backup': True  # Compress backup to save space
}
```

### Disable Auto-Rollback

If you want to handle rollback manually:

```python
migration_config = {
    'auto_rollback': False  # Disable automatic rollback on failure
}
```

### Skip Validation

For testing purposes, you can skip certain validation steps by modifying `src/integration/migration_manager.py`.

⚠️ **Not recommended for production**

## 📚 Additional Resources

### Documentation
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY_ISSUE_14.md`
- **Pipeline Architecture**: See existing implementation summaries
- **Configuration Guide**: `config/README.md` (if available)

### Scripts
- **Migration**: `scripts/migrate_to_pipeline.py`
- **Validation**: `scripts/validate_migration.py`
- **Rollback**: `scripts/rollback_migration.py`
- **Deployment**: `scripts/deploy_production.py`

### Tests
- **Integration Tests**: `tests/integration/`
- **Run Tests**: `pytest tests/integration/ -v`

## 🆘 Support

If you encounter issues not covered in this guide:

1. Check the logs: `data/logs/migration.log`
2. Review the implementation summary: `IMPLEMENTATION_SUMMARY_ISSUE_14.md`
3. Run validation: `python scripts/validate_migration.py`
4. Check GitHub issues: [Report an issue](https://github.com/itsnothuy/Doorbell-System/issues)

## ✅ Post-Migration Checklist

After successful migration:

- [ ] Migration validation passed
- [ ] System starts correctly
- [ ] Web interface accessible
- [ ] API endpoints responding
- [ ] Face recognition working
- [ ] Doorbell trigger functional
- [ ] Data integrity verified
- [ ] Performance acceptable

## 🎉 Success!

Once migration is complete and validated, your system is running on the new pipeline architecture with:

- ✅ Improved performance
- ✅ Better scalability
- ✅ Enhanced monitoring
- ✅ Modular design
- ✅ Easier maintenance

Enjoy your upgraded Doorbell Security System!
