"""The staging folder: a run's new site folders, written beside the site and moved into it only once all of them are written."""

import shutil
from pathlib import Path


class Staging:
    """
    The site folders a run replaces, staged beside the site so that no site generator ever sees them half written.

    Example: for the site '/blog', the page '/blog/_notes/hello.md' is written to
    '/.blog.staging/new/_notes/hello.md', and `swap()` moves '_notes' into '/blog' as a whole.
    Used as a context manager, the staging folder is deleted when the run ends, whether it succeeded or not.
    """

    def __init__(self, site_root: Path, folders: list[Path]):
        self.site_root = Path(site_root)
        # Beside the site rather than inside it, and so on the same disk: each swap is a single rename.
        self.root = self.site_root.parent / f".{self.site_root.name}.staging"
        self.folders = [Path(folder).relative_to(self.site_root) for folder in folders]

    def __enter__(self) -> 'Staging':
        # A run that died before cleaning up leaves its staging folder behind.
        shutil.rmtree(self.root, ignore_errors=True)
        for folder in self.folders:
            (self.root / 'new' / folder).mkdir(parents=True)
        return self

    def __exit__(self, *exc_info):
        shutil.rmtree(self.root, ignore_errors=True)

    def path_for(self, site_path: Path) -> Path:
        """Where a file of the site is written until the swap."""
        return self.root / 'new' / Path(site_path).relative_to(self.site_root)

    def swap(self):
        """Replace each staged folder of the site with its new version; the old ones are deleted on exit."""
        for folder in self.folders:
            site_folder = self.site_root / folder
            if site_folder.exists():
                old = self.root / 'old' / folder
                old.parent.mkdir(parents=True, exist_ok=True)
                site_folder.rename(old)
            site_folder.parent.mkdir(parents=True, exist_ok=True)
            (self.root / 'new' / folder).rename(site_folder)
