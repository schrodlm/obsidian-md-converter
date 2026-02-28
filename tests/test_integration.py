"""Integration tests — exercise the real pipeline end-to-end (no mocks)."""

import pytest
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.cli import process_file, build_link_mapping


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
    transform_links: true
    copy_images: true
links:
  url_format: "/{{collection}}/{{slug}}.html"
  fallback_mode: "text"
front_matter:
  enabled: true
  type: "base"
  preserve_original_title: true
""")

    config = Config(config_path=cfg_file)
    return config, obsidian, output


class TestProcessFilePipeline:
    """End-to-end: config → process_file → output file with correct content."""

    def test_produces_slugified_output(self, vault):
        config, obsidian, output = vault
        source = ObsidianPath(obsidian / "notes" / "Hello World.md")
        target = OutputPath(output / "_notes")

        warnings = process_file(source, target, link_mapping={}, config=config)

        out_file = target.path / "hello-world.md"
        assert out_file.exists(), f"Expected {out_file} to be created"
        assert warnings == []

    def test_preserves_original_title(self, vault):
        config, obsidian, output = vault
        source = ObsidianPath(obsidian / "notes" / "Hello World.md")
        target = OutputPath(output / "_notes")

        process_file(source, target, link_mapping={}, config=config)

        content = (target.path / "hello-world.md").read_text()
        # Original title preserved (not slugified "hello-world")
        assert 'title: "Hello World"' in content

    def test_front_matter_present_in_output(self, vault):
        config, obsidian, output = vault
        source = ObsidianPath(obsidian / "notes" / "Hello World.md")
        target = OutputPath(output / "_notes")

        process_file(source, target, link_mapping={}, config=config)

        content = (target.path / "hello-world.md").read_text()
        assert content.startswith("---\n")
        assert "---\n" in content[4:]  # closing delimiter


class TestBuildLinkMapping:
    """End-to-end: config with source_mappings → link mapping dict."""

    def test_builds_mapping_for_notes(self, vault):
        config, obsidian, output = vault
        mapping = build_link_mapping(config)

        # Both notes should be mapped (original + lowercase)
        assert "Hello World" in mapping
        assert "Second Note" in mapping
        # URL format applied
        assert mapping["Hello World"].endswith(".html")
        assert "/notes/" in mapping["Hello World"]