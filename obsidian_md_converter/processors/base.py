from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LinkTarget:
    """
    The page a link points at, resolved by the core: its site folder's start, its path there, and its file relative
    to the linking page.

    Example: from 'courses/nswi035.md' to 'notes/tmux.md' -> LinkTarget('/notes', 'tmux.md', '../notes/tmux.md')
    """
    start: str
    neutral_path: Path
    file: Path


@dataclass(frozen=True)
class PageLink:
    """
    A link as a page shows it, resolved by the core: its text and the URL it points at, with the heading's anchor.

    Example: [[tmux#Prefix key|the prefix]] -> PageLink('the prefix', '/notes/tmux.html#prefix-key')
    """
    text: str
    url: str


@dataclass(frozen=True)
class PageImage:
    """
    An image as a page shows it, resolved by the core: its alt text, its URL on the site and the size the note gave.

    Example: ![[a.png|Flow|200x100]] -> PageImage('Flow', '/assets/img/a.png', '200', '100')
    """
    alt: str
    url: str
    width: str | None = None
    height: str | None = None


class Processor(ABC):
    """
    How one kind of site, such as Jekyll, expects a run's pages to be written.

    The core reads the vault, resolves every note, heading and image, checks them and writes the pages. Wherever
    sites differ, it asks the processor, which answers from resolved values with plain values:
      - a page's name and address on the site: page_path, default_start, link_address, anchor_ids
      - the front matter fields the site needs: front_matter_defaults
      - the Markdown text of a structure: link_markdown, image_markdown

    It never parses, looks anything up or touches the disk; what the core hands it is all it knows.
    """

    @abstractmethod
    def page_path(self, neutral_path: Path, front_matter: dict) -> Path:
        """
        The page's path below its mapping's site folder, given its neutral path and the note's front matter values.

        Raises:
            ValueError: if the front matter lacks what the site needs to name the page.
        """

    @abstractmethod
    def default_start(self, site_folder: str) -> str:
        """How the addresses of a site folder's pages start, as the site's generator does it by default."""

    @abstractmethod
    def link_address(self, target: LinkTarget) -> str:
        """What a link to a page points at: its URL, or its file."""

    @abstractmethod
    def anchor_ids(self, heading_texts: list[str]) -> list[str]:
        """The ids the site gives a page's headings, in the same order, given each heading's rendered text."""

    @abstractmethod
    def front_matter_defaults(self) -> dict[str, str]:
        """Fields a page gets when its note's front matter lacks them, besides its title, as plain values."""

    @abstractmethod
    def link_markdown(self, link: PageLink) -> str:
        """The Markdown text that shows a link on a page."""

    @abstractmethod
    def image_markdown(self, image: PageImage) -> str:
        """The Markdown text that shows an image on a page."""
