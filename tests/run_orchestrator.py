#!/usr/bin/env python3
"""
Test Orchestrator CLI Runner

Command-line interface for the comprehensive testing framework orchestrator.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.framework.orchestrator import main

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
