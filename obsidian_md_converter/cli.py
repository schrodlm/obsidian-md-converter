#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.pipeline import run
from obsidian_md_converter.problems import Problem
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

    problems, total_files = run(config)
    print("\n" + "="*60)
    lines, exit_code = report(problems, total_files, config.validate_only)
    print("\n".join(lines))
    sys.exit(exit_code)


def report(problems: list[Problem], total_files: int, validate_only: bool) -> tuple[list[str], int]:
    """
    What a run prints, and its exit code: like a compiler, every problem in vault order, then the counts.

    Example:
        notes/Hello.md:3: warning: Missing image: gone.png; writing its alt text instead [missing-image]
        Transfer completed: 0 error(s), 1 warning(s), 12 files processed
    """
    errors = sum(problem.is_error for problem in problems)
    counts = f"{errors} error(s), {len(problems) - errors} warning(s)"
    mode = 'Validation' if validate_only else 'Transfer'

    lines = [str(problem) for problem in sorted(problems, key=_location)]
    if not errors:
        return lines + [f"{mode} completed: {counts}, {total_files} files processed"], 0
    lines.append(f"{mode} FAILED: {counts}")
    if not validate_only:
        lines.append("Nothing was written to the site.")
    return lines, 1


def _location(problem: Problem) -> tuple:
    """Problems about the whole run first, then by note and line; one place keeps the order its checks ran in."""
    return problem.note is not None, str(problem.note or ''), problem.line or 0


if __name__ == "__main__":
    main()
