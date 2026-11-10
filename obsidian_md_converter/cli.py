#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.links import build_link_mapping
from obsidian_md_converter.pipeline import process_mappings
from obsidian_md_converter.utils import enforce_dir


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert Obsidian notes to standard markdown')
    parser.add_argument('--validate', action='store_true',
                       help='Only validate files without writing output')
    parser.add_argument('--force', action='store_true',
                       help='Create output directories if they do not exist')
    parser.add_argument('--fix', action='store_true',
                       help='Automatically resolve warnings in source files')
    parser.add_argument('--obsidian-root', type=str,
                       help='Override Obsidian vault root directory')
    parser.add_argument('--output-root', type=str,
                       help='Override output root directory')
    parser.add_argument('--obsidian-image-dir', type=str,
                       help='Override Obsidian image directory (relative to obsidian-root)')
    parser.add_argument('--output-image-dir', type=str,
                       help='Override output image directory (relative to output-root)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: config.yaml next to the package)')
    args = parser.parse_args()

    # Load configuration (YAML + CLI overrides merged in one step)
    config_path = Path(args.config).resolve() if args.config else None
    config = Config(
        config_path=config_path,
        obsidian_root=args.obsidian_root,
        output_root=args.output_root,
        obsidian_image_dir=args.obsidian_image_dir,
        output_image_dir=args.output_image_dir,
        validate_only=args.validate or None,
        force_create=args.force or None,
        fix_source=args.fix or None,
    )
    print(f"Loaded configuration from {config._config_dir}")

    if not config.validate_only and config.force_create:
        enforce_dir(config.output_root)
        enforce_dir(config.output_image_dir)

    # Print header
    if config.validate_only:
        print("Running validation mode...")
    else:
        print("Starting the transfer process...")

    # Build link mapping for all notes
    print("\nBuilding link mapping...")
    link_mapping = build_link_mapping(config)
    print(f"Built mapping for {len(link_mapping)} notes")

    # Process all files
    errors, warnings, total_files = process_mappings(link_mapping, config)

    # Print summary
    print("\n" + "="*60)
    if len(errors) > 0:
        print(f"{'Validation' if config.validate_only else 'Transfer'} FAILED")
        print(f"   {len(errors)} error(s):")
        for error in errors[:10]:
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
        if len(warnings) > 0:
            print(f"   {len(warnings)} warning(s)")
        sys.exit(1)
    elif len(warnings) > 0:
        print(f"{'Validation' if config.validate_only else 'Transfer'} completed with warnings")
        for warning in warnings:
            print(f"   - {warning}")
        print(f"   {total_files} files processed")
        sys.exit(0)
    else:
        print(f"{'Validation' if config.validate_only else 'Transfer'} completed successfully!")
        print(f"   {total_files} files processed")
        sys.exit(0)

if __name__ == "__main__":
    main()
