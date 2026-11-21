from pathlib import Path

from obsidian_md_converter.processors.anchors import gfm_anchor_ids
from obsidian_md_converter.processors.base import LinkTarget, PageImage, PageLink, Processor


class MarkdownProcessor(Processor):
    """
    How a run's pages are written as normalized Markdown, for any Markdown site (docs/normalized-markdown.md).

    It follows no generator's conventions: links point at the linked page's file, and the site decides its URLs.
    """

    def page_path(self, neutral_path: Path, front_matter: dict) -> Path:
        """The neutral path as it is: a dated note keeps its date in its front matter, not in its name."""
        return neutral_path

    def default_start(self, site_folder: str) -> str:
        """The site folder's exact name. Links point at files, so no address of this processor uses it."""
        return '/' + site_folder

    def link_address(self, target: LinkTarget) -> str:
        """
        The linked page's file, relative to the linking page, so it works wherever the converted folder is placed.

        Example: from 'courses/nswi035.md' to 'notes/tmux.md' -> '../notes/tmux.md'
        """
        return target.file.as_posix()

    def anchor_ids(self, heading_texts: list[str]) -> list[str]:
        """GitHub's heading ids, which the specification asks every site to give its headings too."""
        return gfm_anchor_ids(heading_texts)

    def front_matter_defaults(self) -> dict[str, str]:
        """None: a page gets only its title, and no generator-specific field."""
        return {}

    def link_markdown(self, link: PageLink) -> str:
        """An ordinary Markdown link. Example: PageLink('tmux', '../notes/tmux.md') -> '[tmux](../notes/tmux.md)'"""
        return f"[{link.text}]({link.url})"

    def image_markdown(self, image: PageImage) -> str:
        """
        An ordinary Markdown image, with no size.

        Example: PageImage('Flow', '/assets/img/a.png', '200') -> '![Flow](/assets/img/a.png)'
        """
        return f"![{image.alt}]({image.url})"
