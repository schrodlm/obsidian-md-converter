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


class TestSubfolders:
    """Notes in subfolders of a mapping are published below its destination, keeping their folder path."""

    def test_nested_note_is_written_into_its_subfolder_and_linked_there(self, vault):
        config, obsidian, output = vault
        nested = obsidian / "notes" / "Moje Projekty" / "rentczecher"
        nested.mkdir(parents=True)
        (nested / "First Entry.md").write_text("Entry body\n")
        (obsidian / "notes" / "Hello World.md").write_text("See [[First Entry]].\n")

        link_mapping = build_link_mapping(config)
        errors, warnings, total_files = process_mappings(link_mapping, config)

        assert errors == []
        assert (output / "_notes" / "moje-projekty" / "rentczecher" / "first-entry.md").exists()
        hello = (output / "_notes" / "hello-world.md").read_text()
        assert "[First Entry](/notes/moje-projekty/rentczecher/first-entry.html)" in hello


class TestOutputNames:
    """The neutral output name is the same for every generator; Jekyll's post convention is applied on write."""

    def jekyll_vault(self, vault):
        config, obsidian, output = vault
        config.front_matter_config = {'type': 'jekyll', 'jekyll': {'auto_add': True, 'default_layout': 'note'}}
        return config, obsidian, output

    def test_neutral_name_ignores_post_front_matter(self, vault):
        config, obsidian, output = vault
        post = obsidian / "notes" / "Turing Machine.md"
        post.write_text("---\nlayout: post\ndate: 2024-12-20\n---\nBody\n")
        assert slugify(ObsidianPath(post)) == "turing-machine.md"

    def test_jekyll_post_is_written_under_its_dated_name(self, vault):
        config, obsidian, output = self.jekyll_vault(vault)
        post = obsidian / "notes" / "Turing Machine.md"
        post.write_text("---\nlayout: post\ndate: 2024-12-20\n---\nBody\n")
        target = OutputPath(output / "_notes")

        process_file(ObsidianPath(post), target, link_mapping={}, config=config)

        assert (target.path / "2024-12-20-turing-machine.md").exists()

    def test_validation_fails_a_jekyll_post_without_a_date(self, vault):
        config, obsidian, output = self.jekyll_vault(vault)
        config.validate_only = True
        post = obsidian / "notes" / "Undated.md"
        post.write_text("---\nlayout: post\n---\nBody\n")

        with pytest.raises(ConversionError, match="date"):
            process_file(ObsidianPath(post), None, link_mapping={}, config=config)


class TestOutputClashes:
    """Two notes that would be written to the same file fail the run, and neither is written."""

    def add_clashing_notes(self, obsidian):
        (obsidian / "notes" / "Délivery (protocols).md").write_text("Real note\n")
        (obsidian / "notes" / "delivery protocols.md").write_text("Leftover\n")

    def test_clashing_notes_are_an_error_and_neither_is_written(self, vault):
        config, obsidian, output = vault
        self.add_clashing_notes(obsidian)

        errors, warnings, total_files = process_mappings(link_mapping={}, config=config)

        clash_errors = [e for e in errors if "would both be written to delivery-protocols.md" in e]
        assert len(clash_errors) == 1
        assert not (output / "_notes" / "delivery-protocols.md").exists()
        assert (output / "_notes" / "hello-world.md").exists()

    def test_validation_reports_clashes(self, vault):
        config, obsidian, output = vault
        self.add_clashing_notes(obsidian)
        config.validate_only = True

        errors, warnings, total_files = process_mappings(link_mapping={}, config=config)

        assert any("would both be written to" in e for e in errors)

    def test_same_name_in_different_subfolders_does_not_clash(self, vault):
        config, obsidian, output = vault
        for project in ("alpha", "beta"):
            (obsidian / "notes" / project).mkdir()
            (obsidian / "notes" / project / "Log.md").write_text(f"{project}\n")

        errors, warnings, total_files = process_mappings(link_mapping={}, config=config)

        assert errors == []
        assert (output / "_notes" / "alpha" / "log.md").exists()
        assert (output / "_notes" / "beta" / "log.md").exists()


class TestEmptyNotes:
    """A note with nothing below its front matter is a placeholder: not published, and links to it are plain text."""

    def test_empty_note_is_not_published_and_links_to_it_are_text(self, vault):
        config, obsidian, output = vault
        (obsidian / "notes" / "Load Balancer.md").write_text("\n  \n")
        (obsidian / "notes" / "Header Only.md").write_text("---\ntags: [x]\n---\n\n")
        (obsidian / "notes" / "Hello World.md").write_text("Behind a [[Load Balancer]].\n")

        errors, warnings, total_files = process_mappings(build_link_mapping(config), config)

        assert errors == []
        assert not (output / "_notes" / "load-balancer.md").exists()
        assert not (output / "_notes" / "header-only.md").exists()
        assert "Behind a Load Balancer." in (output / "_notes" / "hello-world.md").read_text()

    def test_empty_note_does_not_clash_with_the_real_one(self, vault):
        config, obsidian, output = vault
        (obsidian / "notes" / "Delivery (protocols).md").write_text("Real note\n")
        (obsidian / "notes" / "Delivery protocols.md").write_text("")

        errors, warnings, total_files = process_mappings(link_mapping={}, config=config)

        assert errors == []
        assert "Real note" in (output / "_notes" / "delivery-protocols.md").read_text()


class TestDrafts:
    """A note marked `draft: true` is not published, and links to it are plain text."""

    def test_draft_is_not_published_and_links_to_it_are_text(self, vault):
        config, obsidian, output = vault
        (obsidian / "notes" / "Work In Progress.md").write_text("---\ndraft: true\n---\nNot ready\n")
        (obsidian / "notes" / "Hello World.md").write_text("See [[Work In Progress]].\n")

        errors, warnings, total_files = process_mappings(build_link_mapping(config), config)

        assert errors == []
        assert not (output / "_notes" / "work-in-progress.md").exists()
        assert "See Work In Progress." in (output / "_notes" / "hello-world.md").read_text()

    def test_draft_false_and_quoted_true_are_published(self, vault):
        config, obsidian, output = vault
        (obsidian / "notes" / "Finished.md").write_text("---\ndraft: false\n---\nDone\n")
        (obsidian / "notes" / "Quoted.md").write_text('---\ndraft: "true"\n---\nText, not a boolean\n')

        errors, warnings, total_files = process_mappings(link_mapping={}, config=config)

        assert errors == []
        assert (output / "_notes" / "finished.md").exists()
        assert (output / "_notes" / "quoted.md").exists()


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
