"""Rewriting one page: each Obsidian link, embed and Markdown image is handed to the links or images module."""

from obsidian_md_converter.images import rewrite_embed, rewrite_markdown_image
from obsidian_md_converter.links import NoteLinks, rewrite_wikilink
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter import scanning
from obsidian_md_converter.scanning import Embed, WikiLink


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
