#!/usr/bin/env python3
"""
Run End-to-End Demo

Quick script to run various demo modes of the Doorbell Security System.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.orchestrator import DemoOrchestrator


def main():
    """Main entry point for running demos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Doorbell Security System - End-to-End Demo Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete demo (automated)
  %(prog)s
  
  # Run interactive demo
  %(prog)s --interactive
  
  # Run quick 5-minute demo
  %(prog)s --quick
  
  # Save report to file
  %(prog)s --output demo_report.json
  
  # Run specific flow
  %(prog)s --flow setup
  %(prog)s --flow operations
  %(prog)s --flow advanced
  %(prog)s --flow admin
  %(prog)s --flow troubleshooting
        """
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run demo in interactive mode with user prompts'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick 5-minute demo'
    )
    parser.add_argument(
        '--flow',
        choices=['setup', 'operations', 'advanced', 'admin', 'troubleshooting'],
        help='Run specific demo flow only'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save demo report to JSON file'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create orchestrator
    orchestrator = DemoOrchestrator(interactive=args.interactive)
    
    # Run specific flow if requested
    if args.flow:
        from demo.flows.initial_setup import InitialSetupFlow
        from demo.flows.daily_operation import DailyOperationFlow
        from demo.flows.advanced_features import AdvancedFeaturesFlow
        from demo.flows.administration import AdministrationFlow
        from demo.flows.troubleshooting import TroubleshootingFlow
        
        flow_map = {
            'setup': InitialSetupFlow(),
            'operations': DailyOperationFlow(),
            'advanced': AdvancedFeaturesFlow(),
            'admin': AdministrationFlow(),
            'troubleshooting': TroubleshootingFlow()
        }
        
        flow = flow_map[args.flow]
        results = flow.run_demo()
        
    # Run quick demo
    elif args.quick:
        results = orchestrator.run_quick_demo()
    
    # Run complete demo
    else:
        results = orchestrator.run_complete_demo()
    
    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Demo report saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
