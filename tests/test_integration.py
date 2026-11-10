"""Integration tests — exercise the real pipeline end-to-end (no mocks)."""

import pytest
import shutil
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.cli import process_file, build_link_mapping, process_mappings, slugify
from obsidian_md_converter.errors import ConversionError


@pytest.fixture
def vault(tmp_path):
    """Set up a minimal obsidian vault + output tree and return a Config."""
    obsidian = tmp_path / "vault"
    output = tmp_path / "output"

    # Vault structure
    (obsidian / "notes").mkdir(parents=True)
    (obsidian / "assets" / "images").mkdir(parents=True)
    (output / "assets" / "img").mkdir(parents=True)
    (output / "_notes").mkdir(parents=True)

    # A test image (fake PNG header)
    (obsidian / "assets" / "images" / "logo.png").write_bytes(b"\x89PNG fake")

    # TODO: Add standard markdown image test once validate_images_in_content
    # resolves ![](path) against source file parent instead of obsidian_image_dir
    (obsidian / "notes" / "Hello World.md").write_text(
        "---\ntitle: \"Hello World\"\n---\n"
        "Some content with a [[Second Note]] link.\n"
    )

    # A second note so we can test link mapping
    (obsidian / "notes" / "Second Note.md").write_text(
        "---\ntitle: \"Second Note\"\n---\n"
        "References [[Hello World]].\n"
    )

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"""\
paths:
  obsidian_root: "{obsidian}"
  output_root: "{output}"
images:
  source_dir: "assets/images"
  destination_dir: "assets/img"
source_mappings:
  - source: "notes"
    destination: "_notes"
links:
  url_format: "/{{collection}}/{{slug}}.html"
front_matter:
  enabled: true
  type: "base"
  preserve_original_title: true
""")

    config = Config(config_path=cfg_file)
    return config, obsidian, output


class TestUnusableNames:
    """A note whose name has nothing to build a URL from fails, naming the file, instead of becoming a hidden '.md'."""

    def test_punctuation_only_name_fails_and_writes_nothing(self, vault):
        config, obsidian, output = vault
        (obsidian / "notes" / "???.md").write_text("Body\n")

        errors, warnings, total_files = process_mappings(build_link_mapping(config), config)

        assert any("???.md" in e and "rename it" in e for e in errors)
        assert not (output / "_notes" / ".md").exists()

    def test_names_in_other_scripts_are_transliterated(self, vault):
        config, obsidian, output = vault
        (obsidian / "notes" / "Привет мир.md").write_text("Body\n")

        errors, warnings, total_files = process_mappings(build_link_mapping(config), config)

        assert errors == []
        assert (output / "_notes" / "privet-mir.md").exists()
