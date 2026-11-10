"""Converting every mapped folder: select its notes, check them, and write each one to the site."""

import os
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.errors import ConversionError
from obsidian_md_converter.front_matter import create_front_matter
from obsidian_md_converter.headings import HeadingAnchors
from obsidian_md_converter.images import validate_images
from obsidian_md_converter.links import LinkTarget, NoteLinks
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.naming import slugify
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.publishing import find_output_clashes, get_directory_md_files
from obsidian_md_converter.references import rewrite_references
from obsidian_md_converter import scanning


# DANGEROUS method, use with care!
# Usable only in this directory
def remove_contents_of(directory: OutputPath):
    if not directory.is_relative_to(OutputPath._root):
        raise RuntimeError("Trying to remove contents outside of this project! Aborted.")

    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            (Path(root) / name).unlink()
        for name in dirs:
            (Path(root) / name).rmdir()


def validate_file_readable(filepath: ObsidianPath):
    """Check if file is readable. Raises ConversionError if not."""
    try:
        filepath.read_text(encoding='utf-8')
    except Exception as e:
        raise ConversionError(filepath, f"File not readable: {e}")


def process_file(source_filepath: ObsidianPath, target_directory: OutputPath, link_mapping: dict[str, LinkTarget], config: Config):
    """
    Process or validate a single file.

    Args:
        validate_only: If True, only validate without writing. If False, do full conversion.
        fix_source: If True, automatically fix warnings in the source file.

    Returns:
        List of warning messages (may be empty).
    """
    file_warnings = []

    # Always validate first
    validate_file_readable(source_filepath)
    content = source_filepath.read_text(encoding='utf-8')
    validate_images(scanning.scan(content), config.obsidian_image_dir, source_filepath)

    # Parse front matter once
    fm = create_front_matter(content, {'front_matter': config.front_matter_config})

    # Check for tabs in front matter
    if fm.fix_tabs():
        if config.fix_source:
            source_filepath.write_text(fm.serialize(), encoding='utf-8')
            file_warnings.append(f"{source_filepath.name}: Fixed tabs in front matter YAML")
        else:
            file_warnings.append(f"{source_filepath.name}: Tabs in front matter YAML (use --fix to resolve)")

    if fm.yaml_error:
        raise ConversionError(source_filepath, f"Invalid front matter YAML: {fm.yaml_error}")

    # The site generator's naming convention applies to the neutral name; breaking it fails validation too.
    try:
        output_filename = fm.apply_slug_transform(slugify(source_filepath))
    except ValueError as e:
        raise ConversionError(filepath=str(source_filepath), reason=str(e))

    # If validate_only, we're done
    if config.validate_only:
        return file_warnings

    # Otherwise, do the full conversion
    original_filename = Path(source_filepath).name

    try:
        dst = target_directory / output_filename
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Apply front matter processing
        #TODO: Move logic to to fm.try_apply()
        if config.front_matter_enabled:
            # Determine title from original filename (before slugification)
            if config.preserve_original_title and original_filename:
                title = Path(original_filename).stem
            else:
                title = Path(output_filename).stem.replace('-', ' ').title()

            fm.ensure(title)

        # Single write to output
        dst.write_text(fm.serialize(), encoding='utf-8')

        links = NoteLinks(
            link_mapping=link_mapping,
            own_anchors=HeadingAnchors(fm.body),
            source_name=source_filepath.name,
            warnings=file_warnings,
        )
        rewrite_references(dst, config.obsidian_image_dir, config.output_image_dir, links)
    except ConversionError as e:
        if dst.exists():
            dst.unlink()
        raise e

    return file_warnings


def process_mappings(link_mapping: dict[str, LinkTarget], config: Config):
    """
    Process all source->destination mappings.

    Args:
        link_mapping: Dictionary mapping note filenames to their paths
        validate_only: If True, only validate without writing
        force_create: If True, create output directories if they don't exist
        fix_source: If True, automatically fix warnings in source files

    Returns:
        Tuple of (errors, warnings, total_files)
    """
    errors = []
    warnings = []
    total_files = 0

    # Images are rebuilt each run like pages, so an image no published note references leaves the site.
    if not config.validate_only and config.output_image_dir.exists():
        print(f"\nClearing images: {config.output_image_dir}")
        remove_contents_of(config.output_image_dir)

    for mapping in config.source_mappings:
        mapping : Mapping
        if not mapping.source.exists():
            errors.append(f"Source directory not found: {mapping.source}")
            continue

        if not config.validate_only:
            if not mapping.destination.exists():
                if config.force_create:
                    print(f"Creating output directory: {mapping.destination}")
                    mapping.destination.mkdir(parents=True, exist_ok=True)
                else:
                    errors.append(f"Output directory not found: {mapping.destination} (use --force to create it)")
                    continue
            print(f"\nProcessing: {mapping.source} -> {mapping.destination}")
            remove_contents_of(mapping.destination)
        else:
            print(f"\nValidating: {mapping.source}")

        source_files = get_directory_md_files(mapping.source)
        print(f"Found {len(source_files)} markdown files")
        total_files += len(source_files)

        clashes = find_output_clashes(source_files, mapping)
        for output, clashing_sources in clashes.items():
            names = " and ".join(f"'{source.name}'" for source in clashing_sources)
            errors.append(f"{names} would both be written to {output}; rename one of them")
        clashing_files = {source for clashing_sources in clashes.values() for source in clashing_sources}

        processed = 0
        for source_file in source_files:
            if source_file in clashing_files:
                continue
            target_directory = None if config.validate_only else mapping.output_directory_for(source_file)
            try:
                file_warnings = process_file(
                    ObsidianPath(source_file),
                    target_directory,
                    link_mapping,
                    config
                )
                warnings.extend(file_warnings)
                processed += 1
                if not config.validate_only:
                    print(f"Transferred {source_file.name}. [{processed}/{len(source_files)}]")
            except ConversionError as e:
                errors.append(f"{source_file.name}: {e.reason}")
                if not config.validate_only:
                    print(f"Transfer failed for {e.filepath}")
                    print(f"Reason: {e.reason}")

        if config.validate_only:
            print(f"Sucessfully validated {processed}/{len(source_files)} files")

    return (errors, warnings, total_files)
