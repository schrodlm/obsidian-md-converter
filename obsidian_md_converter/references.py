"""Rewriting one page: each Obsidian link, embed and Markdown image is handed to the links or images module."""

import re
from pathlib import Path

from obsidian_md_converter.images import ensure_image_available, rewrite_embed, rewrite_markdown_image, transform_image_ref, transform_md_match
from obsidian_md_converter.links import NoteLinks, rewrite_wikilink, transform_md_ref
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter import scanning
from obsidian_md_converter.scanning import Embed, WikiLink


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


def rewrite_references(filepath: OutputPath, src_dir: ObsidianPath, dest_dir: OutputPath, links: NoteLinks):
    """
    Rewrite the links, embeds and Markdown images of the page at `filepath` for the site.

    Only the structures the scanner recognises are replaced; every other character, including
    anything inside code, passes through unchanged. Images are copied to the output image directory.
    """
    content = filepath.read_text(encoding='utf-8')
    note = scanning.scan(content)
    pieces = [content[:len(content) - len(note.body)]]
    position = 0
    for structure in note.replaceable:
        pieces.append(note.body[position:structure.span.start])
        original = note.body[structure.span.start:structure.span.end]
        if isinstance(structure, WikiLink):
            pieces.append(rewrite_wikilink(structure, links))
        elif isinstance(structure, Embed):
            pieces.append(rewrite_embed(structure, src_dir, dest_dir))
        else:
            pieces.append(rewrite_markdown_image(structure, original, src_dir, dest_dir))
        position = structure.span.end
    pieces.append(note.body[position:])
    filepath.write_text("".join(pieces), encoding='utf-8')
