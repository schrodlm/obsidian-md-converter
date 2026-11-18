from abc import ABC, abstractmethod
from pathlib import Path


class Processor(ABC):
    """
    How one kind of site, such as Jekyll, expects a run's pages to be written.

    The core reads the vault, resolves every note, heading and image, checks them and writes the pages. Wherever
    sites differ, it asks the processor, which answers from resolved values with plain values:
      - a page's name and address on the site: page_path, address, anchor_ids
      - the front matter fields the site needs: front_matter_defaults

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
    def address(self, neutral_path: Path, site_folder: str) -> str:
        """The URL that links point at, given the page's neutral path and the name of its mapping's site folder."""

    @abstractmethod
    def anchor_ids(self, heading_texts: list[str]) -> list[str]:
        """The ids the site gives a page's headings, in the same order, given each heading's rendered text."""

    @abstractmethod
    def front_matter_defaults(self) -> dict[str, str]:
        """Fields a page gets when its note's front matter lacks them, besides its title, as plain values."""
