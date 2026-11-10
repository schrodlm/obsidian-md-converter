"""How an image reference is rewritten for the site, and its file copied there."""

import re
import urllib.parse
from pathlib import Path

from obsidian_md_converter.errors import ConversionError
from obsidian_md_converter.paths import ObsidianPath, OutputPath


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
