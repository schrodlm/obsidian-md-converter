"""How an image reference is rewritten for the site, and its file copied there."""

import urllib.parse
from pathlib import Path

from obsidian_md_converter.errors import ConversionError
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.scanning import Embed, MarkdownImage


def rewrite_embed(embed: Embed, src_dir: ObsidianPath, dest_dir: OutputPath) -> str:
    """
    The Markdown that replaces a scanned image embed; the image is copied to the output image directory.

    Examples:
        ![[image.png]]                 -> ![Image](/assets/img/image.png)
        ![[image.png|200]]             -> ![Image](/assets/img/image.png){:width="200"}
        ![[image.png|alt text|200x100]] -> ![alt text](/assets/img/image.png){:width="200" height="100"}
    """
    relative_img_path = Path(embed.target)
    ensure_image_available(src_dir / relative_img_path, dest_dir / relative_img_path)

    img_tag = f"![{embed.alt or 'Image'}]({output_image_url(dest_dir / relative_img_path, OutputPath._root)})"
    if embed.width and embed.height:
        return f'{img_tag}{{:width="{embed.width}" height="{embed.height}"}}'
    if embed.width:
        return f'{img_tag}{{:width="{embed.width}"}}'
    return img_tag


def rewrite_markdown_image(image: MarkdownImage, original: str, src_dir: ObsidianPath, dest_dir: OutputPath) -> str:
    """
    The Markdown that replaces a scanned Markdown image: a local image is copied and pointed at by its site URL.

    Examples:
        ![Figure 1](splay%20results.png) -> ![Figure 1](/assets/img/splay%20results.png)
        ![logo](https://example.com/logo.png) -> unchanged
    """
    name = local_image_name(image)
    if name is None:
        return original
    img_path = Path(name)
    ensure_image_available(src_dir / img_path, dest_dir / img_path)
    return f"![{image.alt}]({output_image_url(dest_dir / img_path, OutputPath._root)})"


def local_image_name(image: Embed | MarkdownImage) -> str | None:
    """
    The vault image a reference shows, named relative to the vault's image folder, or None for an image on the web.

    Examples:
        ![[splay results.png]] -> 'splay results.png'
        ![Figure 1](splay%20results.png) -> 'splay results.png'
        ![logo](https://example.com/logo.png) -> None
    """
    if isinstance(image, Embed):
        return image.target
    if image.url.startswith(('http://', 'https://')):
        return None
    return urllib.parse.unquote(image.url)


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
