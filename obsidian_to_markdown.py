#!/usr/bin/env python3

import os
import unicodedata
import re
import yaml

from typing import Union, TypeVar, Type, Optional
from pathlib import Path
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "config.yaml"

# Load configuration
def load_config():
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to script directory
    config['_resolved_paths'] = {
        'obsidian_root': (SCRIPT_DIR / config['paths']['obsidian_root']).resolve(),
        'output_root': (SCRIPT_DIR / config['paths']['output_root']).resolve(),
        'obsidian_image_dir': None,  # Will be set after reading config
        'output_image_dir': None,
    }

    # Resolve image directories
    obsidian_root = config['_resolved_paths']['obsidian_root']
    output_root = config['_resolved_paths']['output_root']
    config['_resolved_paths']['obsidian_image_dir'] = obsidian_root / config['images']['source_dir']
    config['_resolved_paths']['output_image_dir'] = output_root / config['images']['destination_dir']

    # Validate directories exist
    if not obsidian_root.is_dir():
        raise FileNotFoundError(f"Obsidian root directory not found: {obsidian_root}")
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root directory not found: {output_root}")

    return config

# Global config (will be loaded in main())
CONFIG = None


T = TypeVar('T', bound='BaseValidatedPath')

class PublishTransformError(Exception):
    """Exception raised when a file cannot be transformed for publishing."""
    
    def __init__(self, filepath: str, reason: str):
        """
        Args:
            file_path: Path to the file that failed transformation
            reason: Explanation of why the transformation failed
            original_exception: Optional original exception that caused the failure
        """
        self.filepath = filepath
        self.reason = reason
        message = f"Failed to transform '{filepath}': {reason}"
        super().__init__(message)

class BaseValidatedPath:
    """Base class for validated path wrappers that implements os.PathLike protocol."""
    _root: Path  # Must be set in child classes or via configure_root()

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path).expanduser().absolute()
        self._validate()

    def _validate(self):
        """Ensure path is within the configured root."""
        try:
            if not self._path.is_relative_to(self.__class__._root):
                raise ValueError(f"Path {self._path} is outside {self.__class__.__name__} root")
        except (FileNotFoundError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")

    # Overrding root - for testing purposes
    @classmethod
    def configure_root(cls: Type[T], root: Union[str, Path]) -> None:
        """Configure the root directory for validation."""
        cls._root = Path(root).resolve()

    @property
    def path(self) -> Path:
        """Access the raw Path object when needed."""
        return self._path

    def __fspath__(self):
        """Return the file system path representation (os.PathLike protocol)."""
        return str(self._path)

    def __truediv__(self, other):
        """Support the / operator for path concatenation."""
        return self.__class__(self._path / other)

    # Delegate all other attributes to the wrapped Path
    def __getattr__(self, name):
        return getattr(self._path, name)

class ObsidianPath(BaseValidatedPath):
    """Path guaranteed to be within the Obsidian vault."""
    _root = None  # Will be set from config in main()

class OutputPath(BaseValidatedPath):
    """Path guaranteed to be within the output site."""
    _root = None  # Will be set from config in main()

# Old functions removed - now using config-based approach

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

"""
Retrieves the files that will be converted to markdown and published to the output directory
- currently it does assume that Publish subdirectories will not contain any nested subdirectories, if such a feature is wanted, it needs to be implemented here first and foremost
"""
def get_directory_md_files(directory: Path) -> list[Path]:
    return list(directory.glob("*.md"))



def read_md_metadata(markdown_path: Path):
    """Extract metadata from markdown file's front matter.

    Args:
        markdown_path: Path to a markdown file
    
    Returns:
        Dictionary containung the metadata key-value pairs.
        Returns empty dict if no metadata is found or file can't be read.
    """
    metadata = {}
    try:
        with markdown_path.open('r', encoding='utf-8') as file:
            lines = iter(file)

            #Find the opening "---"
            for line in lines:
                if line.strip() == "---":
                    break
            # No front matter found
            else:
                return metadata

            for line in lines:
                line = line.strip()
                if line == "---":
                    return metadata
                if not line or ':' not in line:
                    continue #Skip empty or invalid lines
                
                key,value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
    except (IOError, UnicodeDecodeError):
        return {}
    return {}


def parse_date(date_str: str) -> Optional[datetime.date]:
    """
    Parse a date string in various formats into a datetime.date object.
    
    Supported formats:
    - YYYY-MM-DD (2024-12-20)
    - DD.MM.YYYY (20.12.2024)
    - DD/MM/YYYY (20/12/2024)
    - MM/DD/YYYY (12/20/2024)
    - Month DD, YYYY (December 20, 2024)
    - DD Month YYYY (20 December 2024)
    - YYYYMMDD (20241220)
    
    Returns:
        datetime.date object if parsing succeeds, None otherwise
    """
    if not date_str:
        return None
    date_str = date_str.strip()

    # Try common formats in order
    formats = [
        '%Y-%m-%d',    # 2024-12-20
        '%d.%m.%Y',    # 20.12.2024
        '%d/%m/%Y',    # 20/12/2024
        '%m/%d/%Y',    # 12/20/2024
        '%B %d, %Y',   # December 20, 2024
        '%d %B %Y',    # 20 December 2024
        '%Y%m%d',      # 20241220
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def slugify(filepath: ObsidianPath):
    """
    Transform a filename into a URL-safe slug.

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
    NOTICE:
        For Jekyll's native '_posts' collection it must follow a specific naming convention: YYYY-MM-DD-title.markdown,
        A simple parsing of markdown's metadata is necessary in order to handle this convention.
    """
    # Remove the file extension temporarily
    stem = Path(filepath).stem.strip()
    ext = Path(filepath).suffix.strip()

    # Normalize unicode characters (convert é to e, etc.)
    stem = unicodedata.normalize('NFKD', stem)
    
    # Convert to ASCII, ignoring non-ASCII chars
    stem = stem.encode('ascii', 'ignore').decode('ascii')
    
    # Replace various special characters with hyphens
    stem = re.sub(r'[^\w\s-]', '', stem)  # Remove remaining non-word chars
    stem = re.sub(r'[\s_-]+', '-', stem)  # Convert spaces/underscores to hyphens
    
    # Convert to lowercase and reattach extension
    slug = stem.lower() + ext.lower()

    # Special case for post layouts
    metadata = read_md_metadata(filepath)
    if metadata.get("layout") == "post":
        date_str = metadata.get("date")
        if not date_str:
            raise PublishTransformError(
                filepath=str(filepath),
                reason="Missing date field for post layout"
            )
        parsed_date = parse_date(date_str)
        if parsed_date is not None:
            date_prefix = f"{parsed_date.year:04d}-{parsed_date.month:02d}-{parsed_date.day:02d}"
            slug = f"{date_prefix}-{slug}"
        else:
            raise PublishTransformError(
                filepath=str(filepath),
                reason="Invalid date front matter provided for a post layout."
            )
            
    return slug


def transform_md_match(match: re.Match, src_dir: Path = None, dest_dir: Path = None):
    """
    Transforms standard Markdown links and images - ![]() or []() - processing images for output directory.

    Handles the following cases:
    - Image references (![alt](path)): 
        * Copies images to destination directory
        * Updates paths to be relative to destination
        * Returns alt text if image unavailable
    - External links [](): Leaves unchanged
    - Document links [](): Leaves unchanged (per project requirements)

    Returns:
        Transformed markdown string or original string if no transformation needed

    Examples:
        # Local image transformation
        Input:  ![logo](images/logo.png)
        Output: ![logo]($(dst_dir)/images/logo.png)

        # External image (unchanged)
        Input:  ![logo](https://example.com/logo.png)
        Output: ![logo](https://example.com/logo.png)

        # Document link (unchanged per current requirements)
        Input:  [Readme](README.md)
        Output: [Readme](README.md)

        # Missing image returns alt text
        Input:  ![missing](nonexistent.png)
        Output: missing
    """
    is_image = match.group(1) == '!'
    alt_text = match.group(2)
    url = match.group(3)
    
    if is_image and not url.startswith(('http://', 'https://')):
        img_path = Path(url)
        src_path = src_dir / img_path
        dst_path = dest_dir / img_path
        
        if ensure_image_available(src_path, dst_path):
            rel_path = dst_path.relative_to(dest_dir)
            return f"![{alt_text}]({rel_path})"
        return alt_text
    return match.group(0)  # Leave external links and doc links unchanged

def transform_obsidian_match(match, src_dir: Path = None, dest_dir: Path = None, link_mapping: dict = None):
    """
    Transforms Obsidian-style links ([[ ]]) into standard markdown format.

    Args:
        match: re.Match object from Obsidian link pattern
        src_dir: Source directory for images (from config)
        dest_dir: Destination directory for images (from config)
        link_mapping: Optional dict mapping note titles to output URLs

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
        return transform_md_ref(full_ref, link_mapping) #transform document links using mapping
    
def transform_references(filepath: OutputPath, src_dir: ObsidianPath, dest_dir: OutputPath, link_mapping: dict = None):
    """
    Transform Obsidian-style references to standard markdown format.
    Handles both document links and image references.

    Document links are converted to standard markdown links using the provided link_mapping.
    If no mapping is provided, falls back to extracting display text only.

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
    from functools import partial
    transform_md_with_params = partial(transform_md_match, src_dir=src_dir, dest_dir=dest_dir)
    content = md_link_pattern.sub(transform_md_with_params, content)

    transform_obsidian_with_params = partial(transform_obsidian_match, src_dir=src_dir, dest_dir=dest_dir, link_mapping=link_mapping)
    content = obsidian_link_pattern.sub(transform_obsidian_with_params, content)

    filepath.write_text(content, encoding='utf-8')

def transform_md_ref(full_ref: str, link_mapping: dict = None) -> str:
    """
    Transform Obsidian document links to standard markdown links.

    Args:
        full_ref: The content inside [[...]]
        link_mapping: Optional dict mapping note titles to output URLs

    Returns:
        Markdown link if mapping found, otherwise display text or empty string

    Examples:
        [[2PC]] with mapping -> [2PC](/notes/2pc/)
        [[2PC|Two Phase Commit]] with mapping -> [Two Phase Commit](/notes/2pc/)
        [[Unknown Note]] without mapping -> Unknown Note (display text only)
    """
    parts = [part.strip() for part in full_ref.split('|')]
    note_title = parts[0]  # The actual note reference
    display_text = parts[1] if len(parts) > 1 else note_title  # Display text or title

    # If we have a link mapping, try to convert to output link
    if link_mapping:
        # Try exact match first, then lowercase
        output_url = link_mapping.get(note_title) or link_mapping.get(note_title.lower())

        if output_url:
            return f"[{display_text}]({output_url})"

    # If no mapping found, return just the display text (backward compatible)
    return display_text

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
    img_tag = f"![{alt_text}](/{(new_parent_dir/relative_img_path).relative_to(root)})"

    # Add dimensions if specified
    if width and height:
        img_tag = f'{img_tag}{{:width="{width}" height="{height}"}}'
    elif width:
        img_tag = f'{img_tag}{{:width="{width}"}}'
    
    return img_tag, relative_img_path
    

def copy_file(src: Path, dst: Path):
    dst.write_bytes(src.read_bytes())

def ensure_image_available(obsidian_img_path: Path, output_img_path: Path) -> bool:
    """Copy image if needed, returns success status"""
    if not obsidian_img_path.exists():
        raise PublishTransformError(obsidian_img_path, "Obsidian image path does not exist.")
    if not output_img_path.parent.exists():
        output_img_path.parent.mkdir(parents=True)
    copy_file(obsidian_img_path, output_img_path)
    return True

def build_link_mapping(config):
    """
    Build a mapping of note titles/filenames to their output URLs.
    This enables conversion of [[wikilinks]] to proper markdown links.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        Dictionary mapping note title (without extension) to output URL
        Example: {'2PC': '/notes/2pc/', 'NPRG077': '/courses/nprg077-...'}
    """
    link_mapping = {}
    url_format = config['links']['url_format']

    for mapping in config['source_mappings']:
        # Skip if link transformation is disabled
        if not mapping.get('transform_links', False):
            continue

        source_dir = ObsidianPath._root / mapping['source']

        if not source_dir.exists():
            continue

        # Get all markdown files
        md_files = get_directory_md_files(source_dir)

        for md_file in md_files:
            # Get the note title (filename without extension)
            note_title = md_file.stem

            # Generate the output slug
            try:
                output_slug = slugify(ObsidianPath(md_file))
                # Remove .md extension for URL
                output_slug_no_ext = Path(output_slug).stem

                # Build output URL based on config format
                # Remove leading underscore from collection name for URL (Jekyll convention)
                collection_name = mapping['destination'].lstrip('_')
                output_url = url_format.format(collection=collection_name, slug=output_slug_no_ext)

                # Map both the original title and lowercase version
                link_mapping[note_title] = output_url
                link_mapping[note_title.lower()] = output_url
            except PublishTransformError:
                # Skip files that can't be slugified
                continue

    return link_mapping

def ensure_front_matter(filepath: OutputPath, original_filename: str = None):
    """
    Ensure a markdown file has front matter with proper title (if enabled in config).
    If file has front matter but no title, add title from original filename.
    If file has no front matter, create minimal front matter with title.

    Args:
        filepath: Path to the output markdown file
        original_filename: Original filename from Obsidian (before slugification) to preserve proper title with diacritics
    """
    # Check if front matter processing is enabled
    if not CONFIG.get('front_matter', {}).get('enabled', True):
        return  # Skip front matter processing if disabled

    content = filepath.read_text(encoding='utf-8')

    # Determine title based on preserve_original_title config
    if CONFIG.get('front_matter', {}).get('preserve_original_title', True) and original_filename:
        title = Path(original_filename).stem  # Remove .md extension but keep original formatting
    else:
        title = filepath.stem.replace('-', ' ').title()

    # Check if file already has front matter
    if content.startswith('---'):
        # Parse existing front matter
        import re
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        if match:
            front_matter = match.group(1)
            body = match.group(2)

            # Check if title already exists
            if re.search(r'^title:', front_matter, re.MULTILINE):
                return  # Already has title, don't modify

            # Add title to existing front matter (even if auto_add is false, since front matter exists)
            new_front_matter = f"---\n{front_matter}\ntitle: \"{title}\"\n---\n{body}"
            filepath.write_text(new_front_matter, encoding='utf-8')
        return

    # No front matter exists - check if auto-add is enabled
    if not CONFIG.get('front_matter', {}).get('auto_add', True):
        return  # Skip adding new front matter if auto-add is disabled

    # Add minimal front matter with configured default layout
    default_layout = CONFIG.get('front_matter', {}).get('default_layout', 'note')
    front_matter = f"---\nlayout: {default_layout}\ntitle: \"{title}\"\n---\n\n"
    new_content = front_matter + content
    filepath.write_text(new_content, encoding='utf-8')

def transfer_publish_file(source_filepath: ObsidianPath, target_directory: OutputPath, obsidian_image_dir: ObsidianPath, output_image_dir: OutputPath, link_mapping: dict = None):
    output_filename = slugify(source_filepath)

    # Store original filename to preserve title with diacritics
    original_filename = Path(source_filepath).name

    try:
        dst = target_directory / output_filename
        copy_file(source_filepath, dst)

        # Ensure the file has front matter with proper title (required for static site generators)
        ensure_front_matter(dst, original_filename)

        transform_references(dst, obsidian_image_dir, output_image_dir, link_mapping=link_mapping)
    except PublishTransformError as e:
        #Remove the already copied file from the output directory
        dst.unlink()
        raise e

def main():
    global CONFIG
    print("Starting the transfer process...")

    # Load configuration
    print("Loading configuration...")
    CONFIG = load_config()
    print(f"✓ Loaded configuration from {CONFIG_FILE}")

    # Set path roots for validation classes from config
    ObsidianPath._root = CONFIG['_resolved_paths']['obsidian_root']
    OutputPath._root = CONFIG['_resolved_paths']['output_root']

    # Phase 1: Build link mapping for all notes
    print("\nPhase 1: Building link mapping...")
    link_mapping = build_link_mapping(CONFIG)
    print(f"✓ Built mapping for {len(link_mapping)} notes")

    # Phase 2: Process each source directory mapping
    print("\nPhase 2: Transferring and transforming files...")
    for mapping in CONFIG['source_mappings']:
        source_dir = ObsidianPath._root / mapping['source']
        output_dir = OutputPath._root / mapping['destination']

        # Skip if source directory doesn't exist
        if not source_dir.exists():
            print(f"Skipping {mapping['source']} (directory not found)")
            continue

        if not output_dir.exists():
            print(f"Warning: Output directory {mapping['destination']} does not exist, skipping")
            continue

        print(f"\nProcessing: {mapping['source']} -> {mapping['destination']}")

        # Remove contents of output subdirectory
        remove_contents_of(OutputPath(output_dir))

        # Get all markdown files from source
        source_files = get_directory_md_files(source_dir)
        print(f"Found {len(source_files)} markdown files")

        # Get image directories from config
        obsidian_image_dir = CONFIG['_resolved_paths']['obsidian_image_dir']
        output_image_dir = CONFIG['_resolved_paths']['output_image_dir']

        # Transfer and transform files
        published = 0
        for source_file in source_files:
            try:
                transfer_publish_file(
                    ObsidianPath(source_file),
                    OutputPath(output_dir),
                    obsidian_image_dir,
                    output_image_dir,
                    link_mapping
                )
                published += 1
                print(f"Transferred {source_file.name}. [{published}/{len(source_files)}]")
            except PublishTransformError as e:
                print(f"Transfer failed for {e.filepath}")
                print(f"Reason: {e.reason}")

    print("\n✓ Transfer process completed!")

if __name__ == "__main__":
    main()
