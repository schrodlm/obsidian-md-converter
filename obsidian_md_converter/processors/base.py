from abc import ABC, abstractmethod
from pathlib import Path


class Processor(ABC):
    """
    How a run's pages are expressed for one kind of site, such as Jekyll.

    Its methods only format what the core has already resolved: they never parse, look anything up or touch the disk.
    """

    @abstractmethod
    def page_path(self, neutral_path: Path, front_matter: dict) -> Path:
        """
        The page's path below its mapping's site folder, given its neutral path and the note's front matter values.

        Raises:
            ValueError: if the front matter lacks what the site needs to name the page.
        """
