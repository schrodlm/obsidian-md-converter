"""The `convert` fixture: write a few notes into a fresh vault, run the real conversion, and read what the site received."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from obsidian_md_converter.pipeline import process_mappings
from obsidian_md_converter.links import build_link_mapping
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
        mappings: vault folder -> site folder; only these folders are created, and only when a note is in them
        existing_site: files already in the site before the run (site path -> content)
    """

    def run(
        notes: dict[str, str | bytes],
        vault: str | None = None,
        validate: bool = False,
        force: bool = True,
        mappings: dict[str, str] | None = None,
        existing_site: dict[str, str | bytes] | None = None,
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
            "source_mappings": [{"source": source, "destination": destination} for source, destination in mappings.items()],
            "images": {"source_dir": "assets/images", "destination_dir": "assets/img"},
            "front_matter": {
                "enabled": True,
                "type": "jekyll",
                "preserve_original_title": True,
                "jekyll": {"auto_add": True, "default_layout": "note"},
            },
        }))
        config = Config(config_path=config_file, validate_only=validate or None, force_create=force or None)
        errors, warnings, _ = process_mappings(build_link_mapping(config), config)
        return ConvertedSite(root=site_root, errors=errors, warnings=warnings)

    return run
