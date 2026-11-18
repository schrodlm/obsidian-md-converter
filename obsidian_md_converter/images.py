"""How an image reference is rewritten for the site, and its file copied there.

An image missing from the vault is written as its alt text, so the page keeps what the image described.
"""

import urllib.parse
from pathlib import Path

from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.processors.base import PageImage, Processor
from obsidian_md_converter.scanning import Embed, MarkdownImage


def rewrite_embed(embed: Embed, src_dir: ObsidianPath, dest_dir: OutputPath, copy_dir: Path,
                  processor: Processor) -> str:
    """
    The Markdown that replaces a scanned image embed: the image is copied into `copy_dir` and handed to the processor,
    pointed at in `dest_dir`, with 'Image' as its alt text when the embed gives none. An image missing from the vault
    becomes its alt text.
    """
    relative_img_path = Path(embed.target)
    if not copy_image(src_dir / relative_img_path, copy_dir / relative_img_path):
        return embed.alt or ''

    url = output_image_url(dest_dir / relative_img_path, OutputPath._root)
    return processor.image_markdown(PageImage(embed.alt or 'Image', url, embed.width, embed.height))


def rewrite_markdown_image(image: MarkdownImage, original: str, src_dir: ObsidianPath, dest_dir: OutputPath,
                           copy_dir: Path, processor: Processor) -> str:
    """
    The Markdown that replaces a scanned Markdown image: a local image is copied into `copy_dir` and handed to the
    processor, pointed at in `dest_dir`; an image on the web is left as written. An image missing from the vault
    becomes its alt text.
    """
    name = local_image_name(image)
    if name is None:
        return original
    img_path = Path(name)
    if not copy_image(src_dir / img_path, copy_dir / img_path):
        return image.alt
    return processor.image_markdown(PageImage(image.alt, output_image_url(dest_dir / img_path, OutputPath._root)))


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
    """
    The site-absolute URL of a copied image, percent-encoded so names with spaces stay one link target.

    Example: <output root>/assets/img/splay results.png -> '/assets/img/splay%20results.png'
    """
    return '/' + urllib.parse.quote(str(Path(output_path).relative_to(root)))


def copy_file(src: Path, dst: Path):
    # TODO: Src should always be in obsidian root while dst should always be in output path
    dst.write_bytes(src.read_bytes())


def copy_image(obsidian_img_path: Path, output_img_path: Path) -> bool:
    """Copy a vault image to the site; False when the vault does not have it."""
    if not obsidian_img_path.exists():
        return False
    if not output_img_path.parent.exists():
        output_img_path.parent.mkdir(parents=True)
    copy_file(obsidian_img_path, output_img_path)
    return True
