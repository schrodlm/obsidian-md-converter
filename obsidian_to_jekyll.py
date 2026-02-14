#!/usr/bin/env python3

import os
import unicodedata
import re

from typing import Union, TypeVar, Type, Optional
from pathlib import Path
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

JEKYLL_ROOT = (SCRIPT_DIR / "../.jekyll_repository/").resolve()
OBSIDIAN_ROOT = (SCRIPT_DIR / "..").resolve()

PUBLISH_DIR = OBSIDIAN_ROOT / "publish"
OBSIDIAN_IMAGE_DIR = OBSIDIAN_ROOT / "assets" / "images"
JEKYLL_IMAGE_DIR = JEKYLL_ROOT / "assets" / "img"

# Source directory to Jekyll collection mappings
# Format: {source_relative_path: jekyll_collection_name}
# Convention: If a directory contains 'front_page.md', it will be used as intro text
#             for that collection's index page in Jekyll
SOURCE_MAPPINGS = {
    'uni/courses': '_courses',
    'notes': '_notes',
    'publish/posts': '_posts',
    'publish/projects': '_projects'
}

#Check if directories exist:
assert JEKYLL_ROOT.is_dir()
assert OBSIDIAN_ROOT.is_dir()
assert PUBLISH_DIR.is_dir()
assert OBSIDIAN_IMAGE_DIR.is_dir()
assert JEKYLL_IMAGE_DIR.is_dir()
assert JEKYLL_IMAGE_DIR.is_relative_to(JEKYLL_ROOT)
assert OBSIDIAN_IMAGE_DIR.is_relative_to(OBSIDIAN_ROOT)


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
    _root = OBSIDIAN_ROOT

class JekyllPath(BaseValidatedPath):
    """Path guaranteed to be within the Jekyll site."""
    _root = JEKYLL_ROOT

# Example: "$PUBLISH_DIR/Posts" -> "$JEKYLL_DIR/_posts"
def get_jekyll_directory(publish_subdir: ObsidianPath, jekyll_root: JekyllPath = JEKYLL_ROOT, publish_dir: ObsidianPath = PUBLISH_DIR) -> JekyllPath:
    if publish_subdir.parent != publish_dir:
        raise RuntimeError(f"Provided directory \"{publish_subdir}\" is not part of publish directory")
    # All capital to lower-case + add "_" to the start
    jekyll_root = Path(jekyll_root / str("_" + publish_subdir.name.lower()))
    #3. Check if it exists in Jekyll dir structure
    if not jekyll_root.is_dir():
        raise RuntimeError(f"Directory {jekyll_root} does not exist in Jekyll directory.")
    return jekyll_root

def get_publish_subdirectories(publish_dir: ObsidianPath = PUBLISH_DIR):
    subdirectories = []
    for file in publish_dir.iterdir():
        if not file.is_dir():
            raise RuntimeError(f"File {file.name} located in the publish directory")
        subdirectories.append(file)
    return subdirectories

# DANGEROUS method, use with care!
# Usable only in this directory
def remove_contents_of(directory: JekyllPath):
    if not directory.is_relative_to(JEKYLL_ROOT):
        raise RuntimeError("Trying to remove contents outside of this project! Aborted.")

    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            (Path(root) / name).unlink()
        for name in dirs:
            (Path(root) / name).rmdir()

"""
Retrieves the publish files that will be converted to jekyll-friendly files and published to the web
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
    Transform a filename into a URL-safe slug following Jekyll conventions.

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
        For Jekyll native '_posts' layout it must follow a specific naming convention: YYYY-MM-DD-title.markdown,
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


def transform_md_match(match: re.Match, src_dir: ObsidianPath = OBSIDIAN_IMAGE_DIR, dest_dir: JekyllPath = JEKYLL_IMAGE_DIR):
    """
    Transforms standard Markdown links and images - ![]() or []() - into Jekyll-compatible format.

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

def transform_obsidian_match(match, src_dir: ObsidianPath = OBSIDIAN_IMAGE_DIR, dest_dir: JekyllPath = JEKYLL_IMAGE_DIR, link_mapping: dict = None):
    """
    Transforms Obsidian-style links ([[ ]]) into Jekyll-compatible format.

    Args:
        match: re.Match object from Obsidian link pattern
        src_dir: Source directory for images (default: OBSIDIAN_IMAGE_DIR)
        dest_dir: Destination directory for images (default: JEKYLL_IMAGE_DIR)
        link_mapping: Optional dict mapping note titles to Jekyll URLs

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
        new_content, relative_src_path = transform_image_ref(full_ref, dest_dir, JEKYLL_ROOT)
        dst_path = dest_dir / relative_src_path
        src_path = src_dir / relative_src_path

        ensure_image_available(
            src_path,
            dst_path
        )
        return new_content
    else:
        return transform_md_ref(full_ref, link_mapping) #transform document links using mapping
    
def transform_references(filepath: JekyllPath, src_dir: ObsidianPath = OBSIDIAN_IMAGE_DIR, dest_dir: JekyllPath = JEKYLL_IMAGE_DIR, link_mapping: dict = None):
    """
    Transform Obsidian-style references to Jekyll-compatible format.
    Handles both document links and image references.

    Document links are now converted to Jekyll links using the provided link_mapping.
    If no mapping is provided, falls back to extracting display text only.

    Image references are transferred from their source to JEKYLL_IMAGE_DIR
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

    # Transform all reference types
    content = md_link_pattern.sub(transform_md_match, content)

    # Use partial to pass link_mapping to transform_obsidian_match
    from functools import partial
    transform_with_mapping = partial(transform_obsidian_match, src_dir=src_dir, dest_dir=dest_dir, link_mapping=link_mapping)
    content = obsidian_link_pattern.sub(transform_with_mapping, content)

    filepath.write_text(content, encoding='utf-8')

def transform_md_ref(full_ref: str, link_mapping: dict = None) -> str:
    """
    Transform Obsidian document links to Jekyll markdown links.

    Args:
        full_ref: The content inside [[...]]
        link_mapping: Optional dict mapping note titles to Jekyll URLs

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

    # If we have a link mapping, try to convert to Jekyll link
    if link_mapping:
        # Try exact match first, then lowercase
        jekyll_url = link_mapping.get(note_title) or link_mapping.get(note_title.lower())

        if jekyll_url:
            return f"[{display_text}]({jekyll_url})"

    # If no mapping found, return just the display text (backward compatible)
    return display_text

def transform_image_ref(full_ref: str, new_parent_dir: JekyllPath = JEKYLL_IMAGE_DIR, root: JekyllPath = JEKYLL_ROOT) -> str:
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

def ensure_image_available(obsidian_img_path: Path, jekyll_img_path: Path) -> bool:
    """Copy image if needed, returns success status"""
    if not obsidian_img_path.exists():
        raise PublishTransformError(obsidian_img_path, "Obsidian image path does not exist.")
    if not jekyll_img_path.parent.exists():
        jekyll_img_path.parent.mkdir(parents=True)
    copy_file(obsidian_img_path, jekyll_img_path)
    return True

def build_link_mapping():
    """
    Build a mapping of note titles/filenames to their Jekyll URLs.
    This enables conversion of [[wikilinks]] to proper Jekyll links.

    Returns:
        Dictionary mapping note title (without extension) to Jekyll URL
        Example: {'2PC': '/notes/2pc/', 'NPRG077': '/courses/nprg077-...'}
    """
    link_mapping = {}

    for source_rel_path, jekyll_collection in SOURCE_MAPPINGS.items():
        source_dir = OBSIDIAN_ROOT / source_rel_path

        if not source_dir.exists():
            continue

        # Get all markdown files
        md_files = get_directory_md_files(source_dir)

        for md_file in md_files:
            # Get the note title (filename without extension)
            note_title = md_file.stem

            # Generate the Jekyll slug
            try:
                jekyll_slug = slugify(ObsidianPath(md_file))
                # Remove .md extension for URL
                jekyll_slug_no_ext = Path(jekyll_slug).stem

                # Build Jekyll URL based on collection
                # Remove leading underscore from collection name for URL
                collection_name = jekyll_collection.lstrip('_')
                jekyll_url = f"/{collection_name}/{jekyll_slug_no_ext}.html"

                # Map both the original title and lowercase version
                link_mapping[note_title] = jekyll_url
                link_mapping[note_title.lower()] = jekyll_url
            except PublishTransformError:
                # Skip files that can't be slugified
                continue

    return link_mapping

def ensure_front_matter(filepath: JekyllPath, original_filename: str = None):
    """
    Ensure a markdown file has front matter with proper title.
    If file has front matter but no title, add title from original filename.
    If file has no front matter, create minimal front matter with title.

    Args:
        filepath: Path to the Jekyll markdown file
        original_filename: Original filename from Obsidian (before slugification) to preserve proper title with diacritics
    """
    content = filepath.read_text(encoding='utf-8')

    # Extract title from original filename if provided
    if original_filename:
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

            # Add title to existing front matter
            new_front_matter = f"---\n{front_matter}\ntitle: \"{title}\"\n---\n{body}"
            filepath.write_text(new_front_matter, encoding='utf-8')
        return

    # No front matter exists, add minimal front matter
    front_matter = f"---\nlayout: note\ntitle: \"{title}\"\n---\n\n"
    new_content = front_matter + content
    filepath.write_text(new_content, encoding='utf-8')

def transfer_publish_file(source_filepath: ObsidianPath, target_directory: JekyllPath, link_mapping: dict = None):
    jekyll_filename = slugify(source_filepath)

    # Store original filename to preserve title with diacritics
    original_filename = Path(source_filepath).name

    try:
        dst = target_directory / jekyll_filename
        copy_file(source_filepath, dst)

        # Ensure the file has front matter with proper title (required for Jekyll collections)
        ensure_front_matter(dst, original_filename)

        transform_references(dst, link_mapping=link_mapping)
    except PublishTransformError as e:
        #Remove the already copied file from the Jekyll directory
        dst.unlink()
        raise e

def main():
    print("Starting the transfer process...")

    # Phase 1: Build link mapping for all notes
    print("\nPhase 1: Building link mapping...")
    link_mapping = build_link_mapping()
    print(f"✓ Built mapping for {len(link_mapping)} notes")

    # Phase 2: Process each source directory mapping
    print("\nPhase 2: Transferring and transforming files...")
    for source_rel_path, jekyll_collection in SOURCE_MAPPINGS.items():
        source_dir = OBSIDIAN_ROOT / source_rel_path
        jekyll_dir = JEKYLL_ROOT / jekyll_collection

        # Skip if source directory doesn't exist
        if not source_dir.exists():
            print(f"Skipping {source_rel_path} (directory not found)")
            continue

        if not jekyll_dir.exists():
            print(f"Warning: Jekyll directory {jekyll_collection} does not exist, skipping")
            continue

        print(f"\nProcessing: {source_rel_path} -> {jekyll_collection}")

        # Remove contents of Jekyll subdirectory
        remove_contents_of(JekyllPath(jekyll_dir))

        # Get all markdown files from source
        source_files = get_directory_md_files(source_dir)
        print(f"Found {len(source_files)} markdown files")

        # Transfer and transform files
        published = 0
        for source_file in source_files:
            try:
                transfer_publish_file(ObsidianPath(source_file), JekyllPath(jekyll_dir), link_mapping)
                published += 1
                print(f"Transferred {source_file.name}. [{published}/{len(source_files)}]")
            except PublishTransformError as e:
                print(f"Transfer failed for {e.filepath}")
                print(f"Reason: {e.reason}")

    print("\n✓ Transfer process completed!")

if __name__ == "__main__":
    main()
