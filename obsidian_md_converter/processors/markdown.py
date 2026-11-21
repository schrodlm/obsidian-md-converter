from pathlib import Path

from obsidian_md_converter.processors.anchors import gfm_anchor_ids
from obsidian_md_converter.processors.base import LinkTarget, PageImage, PageLink, Processor


class MarkdownProcessor(Processor):
    """
    How a run's pages are written as normalized Markdown, for any Markdown site (docs/normalized-markdown.md).

    It follows no generator's conventions. Links point at the linked page's file, and the site decides its URLs; with
    links='urls' they point at the page's URL instead, '/<site folder>/<page>/', for a site that serves its pages there
    and shouldn't have to turn file links into URLs itself.
    """

    def __init__(self, links: str = 'files'):
        # 'files' or 'urls': what a link to another page points at.
        self.links = links

    def page_path(self, neutral_path: Path, front_matter: dict) -> Path:
        """The neutral path as it is: a dated note keeps its date in its front matter, not in its name."""
        return neutral_path

    def default_start(self, site_folder: str) -> str:
        """The site folder's exact name, where its pages' URLs start. Example: 'courses' -> '/courses'"""
        return '/' + site_folder

    def link_address(self, target: LinkTarget) -> str:
        """
        The linked page's file, relative to the linking page, so it works wherever the converted folder is placed;
        with links='urls', its URL: the site folder's start, then its path, ending in '/'.

        Examples, from 'courses/nswi035.md' to 'notes/tmux.md':
            links='files' -> '../notes/tmux.md'
            links='urls'  -> '/notes/tmux/'
        """
        if self.links == 'urls':
            return f"{target.start}/{target.neutral_path.with_suffix('').as_posix()}/"
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
