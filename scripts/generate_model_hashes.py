#!/usr/bin/env python3
"""
Model Hash Generator

Utility script to generate SHA-256 hashes for model files.
This should be run by administrators when adding new models to the registry.

Usage:
    python scripts/generate_model_hashes.py <model_directory>
    
Example:
    python scripts/generate_model_hashes.py data/models/
"""

import sys
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any


def calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Calculate cryptographic hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, sha512, md5)
        
    Returns:
        Hexadecimal hash string
    """
    # Use whitelist approach for security
    ALLOWED_ALGORITHMS = {
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
        'md5': hashlib.md5
    }
    
    if algorithm not in ALLOWED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Allowed: {list(ALLOWED_ALGORITHMS.keys())}")
    
    hash_func = ALLOWED_ALGORITHMS[algorithm]()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def generate_model_info(model_file: Path, model_format: str = None, 
                       version: str = '1.0.0', base_url: str = 'https://models.doorbell-system.com') -> Dict[str, Any]:
    """
    Generate model information including hash and size.
    
    Args:
        model_file: Path to model file
        model_format: Model format (auto-detected from extension if not provided)
        version: Model version (default: '1.0.0')
        base_url: Base URL for model downloads
        
    Returns:
        Dictionary with model information
    """
    if model_format is None:
        # Auto-detect format from extension
        ext = model_file.suffix.lower()
        format_map = {
            '.onnx': 'onnx',
            '.pb': 'tensorflow',
            '.tflite': 'tflite',
            '.dat': 'dlib',
            '.h5': 'keras',
            '.pt': 'pytorch',
            '.pth': 'pytorch'
        }
        model_format = format_map.get(ext, 'unknown')
    
    # Calculate hash
    sha256_hash = calculate_file_hash(model_file, 'sha256')
    
    # Get file size
    file_size = model_file.stat().st_size
    
    # Generate model name from filename (without extension)
    model_name = model_file.stem
    
    return {
        'name': model_name,
        'sha256_hash': sha256_hash,
        'size': file_size,
        'format': model_format,
        'version': version,
        'file': str(model_file),
        'url': f'{base_url}/{model_name}.{model_format}'
    }


def generate_python_code(model_info: Dict[str, Any]) -> str:
    """
    Generate Python code for ModelInfo definition.
    
    Args:
        model_info: Model information dictionary
        
    Returns:
        Python code string
    """
    code = f"""'{model_info['name']}': ModelInfo(
    name='{model_info['name']}',
    url='{model_info['url']}',
    sha256_hash='{model_info['sha256_hash']}',
    size={model_info['size']},
    format='{model_info['format']}',
    version='{model_info['version']}'
),"""
    return code


def main():
    """Main function to generate model hashes."""
    parser = argparse.ArgumentParser(
        description='Generate SHA-256 hashes for model files'
    )
    parser.add_argument(
        'model_directory',
        type=str,
        help='Directory containing model files'
    )
    parser.add_argument(
        '--format',
        type=str,
        default=None,
        help='Model format (auto-detected if not specified)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for generated Python code (prints to stdout if not specified)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='1.0.0',
        help='Model version (default: 1.0.0)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='https://models.doorbell-system.com',
        help='Base URL for model downloads (default: https://models.doorbell-system.com)'
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_directory)
    
    if not model_dir.exists():
        print(f"Error: Directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not model_dir.is_dir():
        print(f"Error: Not a directory: {model_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find all model files
    model_extensions = ['.onnx', '.pb', '.tflite', '.dat', '.h5', '.pt', '.pth']
    model_files = []
    
    for ext in model_extensions:
        model_files.extend(model_dir.glob(f'*{ext}'))
    
    if not model_files:
        print(f"No model files found in {model_dir}", file=sys.stderr)
        print(f"Looking for files with extensions: {', '.join(model_extensions)}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(model_files)} model file(s)\n")
    
    # Generate information for each model
    all_model_info = []
    
    for model_file in sorted(model_files):
        print(f"Processing: {model_file.name}")
        
        try:
            model_info = generate_model_info(
                model_file, 
                args.format,
                version=args.version,
                base_url=args.base_url
            )
            all_model_info.append(model_info)
            
            print(f"  - Size: {model_info['size']:,} bytes")
            print(f"  - SHA-256: {model_info['sha256_hash']}")
            print(f"  - Format: {model_info['format']}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            print()
    
    # Generate Python code
    if all_model_info:
        print("=" * 80)
        print("Generated Python Code for MODEL_REGISTRY:")
        print("=" * 80)
        print("\nMODEL_REGISTRY = {")
        
        for model_info in all_model_info:
            print("    " + generate_python_code(model_info))
        
        print("}\n")
        
        # Save to file if specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                f.write("MODEL_REGISTRY = {\n")
                for model_info in all_model_info:
                    f.write("    " + generate_python_code(model_info) + "\n")
                f.write("}\n")
            
            print(f"Code saved to: {output_path}")
    
    print(f"\nProcessed {len(all_model_info)} model(s) successfully")
    
    # Print verification instructions
    print("\n" + "=" * 80)
    print("Verification Instructions:")
    print("=" * 80)
    print("1. Copy the generated MODEL_REGISTRY code above")
    print("2. Update src/detectors/model_manager.py with the new model definitions")
    print("3. Ensure all model files are uploaded to the secure model repository")
    print("4. Verify model URLs are correct and accessible")
    print("5. Run integrity verification: python -c 'from src.detectors.model_manager import ModelManager; ModelManager().verify_all_models()'")


if __name__ == '__main__':
    main()
