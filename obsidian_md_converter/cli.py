#!/usr/bin/env python3

import os
import re
import argparse
import sys
import urllib.parse

from dataclasses import dataclass
from pathlib import Path
from obsidian_md_converter.front_matter import create_front_matter, FrontMatter
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.errors import ConversionError
from obsidian_md_converter.config import Config
from obsidian_md_converter.utils import enforce_dir, slugify_text
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.headings import HeadingAnchors


@dataclass
class LinkTarget:
    """Where a note is published, and the anchors its headings get there."""
    url: str
    anchors: HeadingAnchors


@dataclass
class NoteLinks:
    """What rewriting one note's links needs: every published note, the note's own headings, and where to report problems."""
    link_mapping: dict[str, LinkTarget]
    own_anchors: HeadingAnchors
    source_name: str
    warnings: list[str]

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

def get_directory_md_files(directory: Path) -> list[Path]:
    """The markdown files a mapping publishes: every one under its source directory, subdirectories included."""
    return [path for path in directory.rglob("*.md") if is_published(path)]

def is_published(path: Path) -> bool:
    """A note is published unless it is a draft (`draft: true`) or a placeholder with nothing below its front matter."""
    fm = FrontMatter(path.read_text(encoding='utf-8'))
    return fm.get('draft') is not True and bool(fm.body.strip())

def slugify(filepath: ObsidianPath) -> str:
    """
    Transform a filename into its neutral, URL-safe output name.

    The name is the same for every site generator; a generator's own naming
    convention (such as Jekyll's dated post filenames) is applied when the file is written.

    Args:
        filepath: ObsidianPath object pointing to the file to convert

    Returns:
        A sanitized filename with:
        - Only lowercase letters, numbers, and hyphens
        - No special characters or punctuation
        - Spaces converted to hyphens
        - Multiple hyphens collapsed
        - Leading/trailing hyphens removed

    Example:
        "Hello World!.md" -> "hello-world.md"

    Raises:
        ConversionError: if the name has no letters or digits to build the slug from.
    """
    stem_slug = slugify_text(Path(filepath).stem)
    if not stem_slug:
        raise ConversionError(filepath, "its name has no letters or digits to build a URL from; rename it")
    return stem_slug + Path(filepath).suffix.strip().lower()


def transform_md_match(match: re.Match, src_dir: Path = None, dest_dir: Path = None):
    """
    Transforms standard Markdown links and images - ![]() or []() - processing images for output directory.

    Handles the following cases:
    - Image references (![alt](path)): the path is URL-decoded and resolved in the vault's
      image directory, the image is copied, and the reference points at its site-absolute URL
    - External links and images: Leaves unchanged
    - Document links [](): Leaves unchanged

    Returns:
        Transformed markdown string or original string if no transformation needed

    Examples:
        Input:  ![Figure 1](splay%20results.png)
        Output: ![Figure 1](/assets/img/splay%20results.png)

        Input:  ![logo](https://example.com/logo.png)
        Output: ![logo](https://example.com/logo.png)
    """
    is_image = match.group(1) == '!'
    alt_text = match.group(2)
    url = match.group(3)
    
    if is_image and not url.startswith(('http://', 'https://')):
        img_path = Path(urllib.parse.unquote(url))
        src_path = src_dir / img_path
        dst_path = dest_dir / img_path

        ensure_image_available(src_path, dst_path)
        return f"![{alt_text}]({output_image_url(dst_path, OutputPath._root)})"
    return match.group(0)  # Leave external links and doc links unchanged

def transform_obsidian_match(match, src_dir: Path = None, dest_dir: Path = None, links: NoteLinks = None):
    """
    Transforms Obsidian-style links ([[ ]]) into standard markdown format.

    Args:
        match: re.Match object from Obsidian link pattern
        src_dir: Source directory for images (from config)
        dest_dir: Destination directory for images (from config)
        links: The published notes and the current note's headings, for document links

    Returns:
        Transformed markdown string or display text

    Examples:
        # Obsidian image with dimensions
        Input:  ![[images/logo.png|200]]
        Output: ![Image](../assets/images/logo.png){:width="200"}

        # Obsidian document link with display text
        Input:  [[README.md|Readme File]]
        Output: [Readme File](/notes/readme/)

        # Obsidian image with alt text
        Input:  ![[logo.png|Company Logo]]
        Output: ![Company Logo](../assets/logo.png)
    """
    is_image = match.group(1) == '!'
    full_ref = match.group(2)
    if is_image:
        new_content, relative_src_path = transform_image_ref(full_ref, dest_dir, OutputPath._root)
        dst_path = dest_dir / relative_src_path
        src_path = src_dir / relative_src_path

        ensure_image_available(
            src_path,
            dst_path
        )
        return new_content
    else:
        return transform_md_ref(full_ref, links)
    
def transform_references(filepath: OutputPath, src_dir: ObsidianPath, dest_dir: OutputPath, links: NoteLinks = None):
    """
    Transform Obsidian-style references to standard markdown format.
    Handles both document links and image references.

    Document links are converted to standard markdown links to the published notes in links.
    A link to a note that is not published becomes its display text.

    Image references are transferred from their source to the output image directory (from config)
        Image reference can be in form:
        1. ![Alt text](path/to/image.png)
        2. ![[image.png]]                           - has to be in image folder (specified in Obsidian config)
        3. ![[path/to/image.png|200]]               - fixed width
        4. ![[path/to/image.png|200x100]]           - fixed width and height
        5. ![[path/to/image.png|My Alt Text]]       - alt text
        6. ![[path/to/image.png|My Alt Text|200]]   - alt text + resize
    """
    content = filepath.read_text(encoding='utf-8')

    # Patterns for different reference types
    obsidian_link_pattern = re.compile(r'(!?)\[\[([^\]\[]+)\]\]')  # ![[ ]] or [[ ]]
    md_link_pattern = re.compile(r'(!?)\[([^\]]+)\]\(([^)]+)\)')    # ![]() or []()

    # Use partial to pass parameters to transform functions
    #TODO: Introduce some form of transformer that will invoke specific functions on what was matched
    """
    I would like to have classes like ClassicMarkdownConverter, PDFConverter.
    Images will have to be embedded into some format I think.
    """
    from functools import partial
    transform_md_with_params = partial(transform_md_match, src_dir=src_dir, dest_dir=dest_dir)
    content = md_link_pattern.sub(transform_md_with_params, content)

    transform_obsidian_with_params = partial(transform_obsidian_match, src_dir=src_dir, dest_dir=dest_dir, links=links)
    content = obsidian_link_pattern.sub(transform_obsidian_with_params, content)

    filepath.write_text(content, encoding='utf-8')

def transform_md_ref(full_ref: str, links: NoteLinks = None) -> str:
    """
    Transform an Obsidian document link to a standard markdown link.

    Args:
        full_ref: The content inside [[...]]
        links: The published notes and the current note's headings

    Returns:
        Markdown link if the note is published, otherwise the display text.
        A link to a heading the note does not have points at the page and is reported.

    Examples:
        [[2PC]] -> [2PC](/notes/2pc.html)
        [[2PC|Two Phase Commit]] -> [Two Phase Commit](/notes/2pc.html)
        [[tmux#Prefix key]] -> [tmux > Prefix key](/notes/tmux.html#prefix-key)
        [[#Prefix key]] -> [Prefix key](#prefix-key)
        [[Unknown Note]] -> Unknown Note
    """
    parts = [part.strip() for part in full_ref.split('|')]
    note_title, _, subpath = parts[0].partition('#')
    note_title = note_title.strip()
    # A nested subpath (#H1#H2) names its innermost heading last.
    heading = subpath.split('#')[-1].strip()

    if len(parts) > 1:
        display_text = parts[1]
    elif note_title and heading:
        display_text = f"{note_title} > {heading}"
    else:
        display_text = note_title or heading

    if links is None:
        return display_text

    if not note_title:
        anchor = links.own_anchors.anchor_for(heading) if heading else None
        if anchor is None:
            links.warnings.append(f"{links.source_name}: [[{full_ref}]] names heading '{heading}', which this note does not have")
            return display_text
        return f"[{display_text}](#{anchor})"

    target = links.link_mapping.get(note_title) or links.link_mapping.get(note_title.lower())
    if target is None:
        return display_text
    if not heading:
        return f"[{display_text}]({target.url})"

    anchor = target.anchors.anchor_for(heading)
    if anchor is None:
        links.warnings.append(
            f"{links.source_name}: [[{full_ref}]] names heading '{heading}', which '{note_title}' does not have; linking to the page"
        )
        return f"[{display_text}]({target.url})"
    return f"[{display_text}]({target.url}#{anchor})"

def transform_image_ref(full_ref: str, new_parent_dir: OutputPath, root: OutputPath) -> str:
    """
    Transform Obsidian-style image references to standard Markdown.
    Handles all these cases:
    1. ![[image.png]]                   → ![Image](image.png)
    2. ![[image.png|200]]               → ![Image](image.png){:width="200"}
    3. ![[image.png|200x100]]           → ![Image](image.png){:width="200" height="100"}
    4. ![[image.png|alt text]]          → ![alt text](image.png)
    5. ![[image.png|alt text|200]]      → ![alt text](image.png){:width="200"}
    6. ![[subdir/image.png]]            → ![Image](subdir/image.png)
    7. ![[image.png|alt text|200x100]]  → ![alt text](image.png){:width="200" height="100"}
    """
    # Split into components and strip whitespace
    parts = [part.strip() for part in full_ref.split('|')]
    relative_img_path = Path(parts[0])
    
    # Default values
    alt_text = "Image"
    width = None
    height = None
    
    # Process additional parameters
    for part in parts[1:]:
        if 'x' in part and all(s.isdigit() for s in part.split('x')):
            # Case 3 & 7: Dimensions (200x100)
            width, height = part.split('x')[:2]
        elif part.isdigit():
            # Case 2 & 5: Single dimension (200)
            width = part
        else:
            # Case 4 & 5 & 7: Alt text (non-numeric)
            alt_text = part
    
    # Build the image tag - 
    img_tag = f"![{alt_text}]({output_image_url(new_parent_dir / relative_img_path, root)})"

    # Add dimensions if specified
    if width and height:
        img_tag = f'{img_tag}{{:width="{width}" height="{height}"}}'
    elif width:
        img_tag = f'{img_tag}{{:width="{width}"}}'
    
    return img_tag, relative_img_path
    

def output_image_url(output_path: OutputPath, root: OutputPath) -> str:
    """The site-absolute URL of a copied image, percent-encoded so names with spaces stay one link target."""
    return '/' + urllib.parse.quote(str(Path(output_path).relative_to(root)))

def copy_file(src: Path, dst: Path):
    # TODO: Src should always be in obsidian root while dst should always be in output path
    dst.write_bytes(src.read_bytes())

def ensure_image_available(obsidian_img_path: Path, output_img_path: Path) -> bool:
    """Copy image if needed, returns success status"""
    if not obsidian_img_path.exists():
        raise ConversionError(obsidian_img_path, "Obsidian image path does not exist.")
    if not output_img_path.parent.exists():
        output_img_path.parent.mkdir(parents=True)
    copy_file(obsidian_img_path, output_img_path)
    return True

def build_link_mapping(config: Config) -> dict[str, LinkTarget]:
    """
    Build a mapping of note titles/filenames to where they are published.
    This enables conversion of [[wikilinks]] to proper markdown links.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        Dictionary mapping note title (without extension) to its output URL and heading anchors
        Example: {'2PC': LinkTarget(url='/notes/2pc.html', anchors=...)}
    """
    link_mapping = {}

    for mapping in config.source_mappings:
        mapping : Mapping
        if not mapping.source.exists():
            continue

        # Get all markdown files
        md_files = get_directory_md_files(mapping.source)

        for md_file in md_files:
            # Get the note title (filename without extension)
            note_title = md_file.stem

            content = md_file.read_text(encoding='utf-8')
            fm = FrontMatter(content)
            try:
                output_slug = slugify(ObsidianPath(md_file))
            except ConversionError:
                # Its conversion reports the unusable name; nothing can link to it.
                continue
            # Remove .md extension for URL; a file in a subdirectory keeps it in its URL
            output_slug_no_ext = (mapping.subdirectory_for(md_file) / Path(output_slug).stem).as_posix()

            # Build output URL based on config format
            # Remove leading underscore from collection name for URL (Jekyll convention)
            collection_name = mapping.destination.name.lstrip('_')
            output_url = config.url_format.format(collection=collection_name, slug=output_slug_no_ext)

            # Map both the original title and lowercase version
            target = LinkTarget(url=output_url, anchors=HeadingAnchors(fm.body))
            link_mapping[note_title] = target
            link_mapping[note_title.lower()] = target

    return link_mapping


def validate_file_readable(filepath: ObsidianPath):
    """Check if file is readable. Raises ConversionError if not."""
    try:
        filepath.read_text(encoding='utf-8')
    except Exception as e:
        raise ConversionError(filepath, f"File not readable: {e}")

def validate_images_in_content(content: str, obsidian_image_dir: ObsidianPath, source_file: ObsidianPath):
    """
    Validate all image references in markdown content.
    Raises ConversionError with all missing images listed if any are found.
    """
    missing_images = []

    # Check Obsidian-style images: ![[image.png]]
    obsidian_image_pattern = re.compile(r'!\[\[([^\]]+)\]\]')
    for match in obsidian_image_pattern.finditer(content):
        img_ref = match.group(1).split('|')[0]  # Remove size/alt text
        img_path = obsidian_image_dir / img_ref
        if not img_path.exists():
            missing_images.append(img_ref)

    # Check standard markdown images: ![alt](image.png)
    md_image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    for match in md_image_pattern.finditer(content):
        img_path_str = match.group(2)
        # Only validate local images (not URLs)
        if not (img_path_str.startswith('http://') or img_path_str.startswith('https://')):
            img_path = obsidian_image_dir / urllib.parse.unquote(img_path_str)
            if not img_path.exists():
                missing_images.append(img_path_str)

    if missing_images:
        images_str = ", ".join(missing_images)
        raise ConversionError(source_file, f"Missing image(s): {images_str}")

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
    validate_images_in_content(content, config.obsidian_image_dir, source_filepath)

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
        transform_references(dst, config.obsidian_image_dir, config.output_image_dir, links=links)
    except ConversionError as e:
        if dst.exists():
            dst.unlink()
        raise e

    return file_warnings

def find_output_clashes(source_files: list[Path], mapping: Mapping) -> dict[Path, list[Path]]:
    """Group the source files that share a neutral output path, keeping only the groups of two or more."""
    sources_by_output: dict[Path, list[Path]] = {}
    for source_file in source_files:
        try:
            output = mapping.subdirectory_for(source_file) / slugify(ObsidianPath(source_file))
        except ConversionError:
            # Its conversion reports the unusable name.
            continue
        sources_by_output.setdefault(output, []).append(source_file)
    return {output: sources for output, sources in sources_by_output.items() if len(sources) > 1}

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
