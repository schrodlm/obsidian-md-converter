"""Integration tests — exercise the real pipeline end-to-end (no mocks)."""

import pytest
import shutil
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.cli import process_file, build_link_mapping, process_mappings
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

    def test_invalid_front_matter_fails_the_note_and_writes_nothing(self, vault):
        config, obsidian, output = vault
        broken = obsidian / "notes" / "Broken.md"
        broken.write_text("---\ntags:\n      - a\n  - b\n---\nBody\n")
        target = OutputPath(output / "_notes")

        with pytest.raises(ConversionError, match="Invalid front matter YAML"):
            process_file(ObsidianPath(broken), target, link_mapping={}, config=config)

        assert not (target.path / "broken.md").exists()


class TestBuildLinkMapping:
    """End-to-end: config with source_mappings → link mapping dict."""

    def test_builds_mapping_for_notes(self, vault):
        config, obsidian, output = vault
        mapping = build_link_mapping(config)

        # Both notes should be mapped (original + lowercase)
        assert "Hello World" in mapping
        assert "Second Note" in mapping
        # URL format applied
        assert mapping["Hello World"].url.endswith(".html")
        assert "/notes/" in mapping["Hello World"].url

class TestProcessMappings:
    """A run whose mapped folders are missing fails instead of converting nothing."""

    def test_missing_source_directory_is_an_error(self, vault):
        config, obsidian, output = vault
        shutil.rmtree(obsidian / "notes")

        errors, warnings, total_files = process_mappings(link_mapping={}, config=config)

        assert any("Source directory not found" in e for e in errors)
        assert total_files == 0

    def test_missing_output_directory_without_force_is_an_error(self, vault):
        config, obsidian, output = vault
        shutil.rmtree(output / "_notes")

        errors, warnings, total_files = process_mappings(link_mapping={}, config=config)

        assert any("Output directory not found" in e for e in errors)

    def test_images_no_note_references_are_removed_from_the_output(self, vault):
        config, obsidian, output = vault
        (obsidian / "notes" / "Hello World.md").write_text("Logo: ![[logo.png]]\n")
        stale = output / "assets" / "img" / "stale.png"
        stale.write_bytes(b"\x89PNG stale")

        errors, warnings, total_files = process_mappings(link_mapping={}, config=config)

        assert errors == []
        assert not stale.exists()
        assert (output / "assets" / "img" / "logo.png").exists()

    def test_validation_leaves_output_images_alone(self, vault):
        config, obsidian, output = vault
        stale = output / "assets" / "img" / "stale.png"
        stale.write_bytes(b"\x89PNG stale")
        config.validate_only = True

        process_mappings(link_mapping={}, config=config)

        assert stale.exists()


class TestLinks:
    """Wikilinks become links to the published page, to the heading's anchor when they name one."""

    def convert(self, vault, body):
        config, obsidian, output = vault
        (obsidian / "notes" / "Second Note.md").write_text(
            "---\ntitle: \"Second Note\"\n---\n## Totální uspořádání (total ordering)\ntext\n"
        )
        source = obsidian / "notes" / "Hello World.md"
        source.write_text(body)
        target = OutputPath(output / "_notes")

        warnings = process_file(ObsidianPath(source), target, build_link_mapping(config), config)

        content = (target.path / "hello-world.md").read_text()
        return content, warnings

    def test_link_to_a_note_points_at_its_page(self, vault):
        content, warnings = self.convert(vault, "See [[Second Note|the second]].\n")
        assert "[the second](/notes/second-note.html)" in content
        assert warnings == []

    def test_link_to_a_heading_points_at_its_anchor(self, vault):
        content, warnings = self.convert(vault, "See [[Second Note#Totální uspořádání (total ordering)|ordering]].\n")
        assert "[ordering](/notes/second-note.html#totální-uspořádání-total-ordering)" in content
        assert warnings == []

    def test_heading_link_without_display_text_shows_note_and_heading(self, vault):
        content, _ = self.convert(vault, "See [[second note#totální uspořádání (total ordering)]].\n")
        assert "[second note > totální uspořádání (total ordering)](/notes/second-note.html#totální-uspořádání-total-ordering)" in content

    def test_link_to_a_missing_heading_points_at_the_page_and_warns(self, vault):
        content, warnings = self.convert(vault, "See [[Second Note#Renamed|ordering]].\n")
        assert "[ordering](/notes/second-note.html)" in content
        assert len(warnings) == 1
        assert "Renamed" in warnings[0]

    def test_link_to_a_heading_in_the_same_note_is_a_bare_anchor(self, vault):
        content, warnings = self.convert(vault, "## Prefix key\nSee [[#Prefix key]].\n")
        assert "[Prefix key](#prefix-key)" in content
        assert warnings == []

    def test_link_to_a_missing_note_is_plain_text_without_warning(self, vault):
        content, warnings = self.convert(vault, "See [[Not Written Yet|later]] and [[Not Written Yet#Part]].\n")
        assert "See later and Not Written Yet > Part." in content
        assert warnings == []
