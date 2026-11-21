"""The `convert` fixture: write a few notes into a fresh vault, run the real conversion, and read what the site received."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from obsidian_md_converter.pipeline import run as run_conversion
from obsidian_md_converter.config import Config

VAULTS = Path(__file__).parent / "vaults"

# Mirrors a real vault's setup: these folders are published, front matter is Jekyll's.
DEFAULT_MAPPINGS = {
    "notes": "_notes",
    "publish/posts": "_posts",
    "publish/projects": "_projects",
    "logs": "_logs",
}


@dataclass
class ConvertedSite:
    root: Path
    vault: Path
    errors: list[str]
    warnings: list[str]

    def page(self, relative_path: str) -> str:
        return (self.root / relative_path).read_text(encoding="utf-8")

    def has(self, relative_path: str) -> bool:
        return (self.root / relative_path).exists()


def write_files(root: Path, files: dict[str, str | bytes]):
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")


@pytest.fixture
def convert(tmp_path):
    """Return a function that converts `notes` (vault path -> content) and returns what the site received.

    Optional knobs, for the tests that need them:
        vault: a fixture vault from tests/vaults/ to start from
        validate: run in --validate mode
        force: create missing output folders (--force)
        fix: resolve tabs in front matter in the vault's notes (--fix)
        mappings: vault folder -> site folder; only these folders are created, and only when a note is in them.
            A tuple of vault folders publishes them all into one site folder.
        urls: vault folder -> the url its mapping starts its addresses with
        processor: the config's processor section; Jekyll with the 'note' layout by default
        existing_site: files already in the site before the run (site path -> content)
    """

    def run(
        notes: dict[str, str | bytes],
        vault: str | None = None,
        validate: bool = False,
        force: bool = True,
        fix: bool = False,
        mappings: dict[str | tuple[str, ...], str] | None = None,
        existing_site: dict[str, str | bytes] | None = None,
        urls: dict[str, str] | None = None,
        processor: dict | None = None,
    ) -> ConvertedSite:
        vault_root = tmp_path / "vault"
        site_root = tmp_path / "site"
        if vault is not None:
            shutil.copytree(VAULTS / vault, vault_root)
        if mappings is None:
            mappings = DEFAULT_MAPPINGS
            for source in mappings:
                (vault_root / source).mkdir(parents=True, exist_ok=True)
        (vault_root / "assets" / "images").mkdir(parents=True, exist_ok=True)
        (site_root / "assets" / "img").mkdir(parents=True, exist_ok=True)

        write_files(vault_root, notes)
        write_files(site_root, existing_site or {})

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump({
            "paths": {"obsidian_root": str(vault_root), "output_root": str(site_root)},
            "source_mappings": [
                ({"sources": list(source)} if isinstance(source, tuple) else {"source": source})
                | {"destination": destination}
                | ({"url": urls[source]} if source in (urls or {}) else {})
                for source, destination in mappings.items()
            ],
            "images": {"source_dir": "assets/images", "destination_dir": "assets/img"},
            "processor": processor or {"type": "jekyll", "default_layout": "note"},
        }))
        config = Config(config_path=config_file, validate_only=validate or None, force_create=force or None, fix_source=fix or None)
        problems, _ = run_conversion(config)
        return ConvertedSite(
            root=site_root,
            vault=vault_root,
            errors=[str(problem) for problem in problems if problem.is_error],
            warnings=[str(problem) for problem in problems if not problem.is_error],
        )

    return run
