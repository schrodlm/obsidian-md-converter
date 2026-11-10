"""Whole conversions of small vaults, grouped by feature."""


class TestFixtureVaults:
    """A fixture vault from tests/vaults/ converts like a real one."""

    def test_course_notes_link_into_long_sections(self, convert):
        """Links into the sections of a long note land on their anchors, and its image is copied."""
        site = convert({}, vault="algorithms-course")
        exam = site.page("_notes/exam-prep.md")
        assert "(/notes/data-structures.html#amortized-analysis-amortizovaná-analýza)" in exam
        assert "(/notes/data-structures.html#range-trees-intervalové-stromy)" in exam
        assert site.has("assets/img/splay results.png")


class TestPageOutput:
    """A published note becomes a page under its slugified name, its front matter keeping the original title."""

    def test_note_is_written_under_its_slugified_name(self, convert):
        site = convert({"notes/Hello World.md": "Body\n"})
        assert site.has("_notes/hello-world.md")
        assert site.errors == []
        assert site.warnings == []

    def test_original_title_is_kept_in_front_matter(self, convert):
        site = convert({"notes/ABA problém.md": "Body\n"})
        page = site.page("_notes/aba-problem.md")
        assert page.startswith("---\n")
        assert 'title: "ABA problém"' in page

    def test_invalid_front_matter_fails_the_note_and_writes_nothing(self, convert):
        site = convert({"notes/Broken.md": "---\ntags:\n      - a\n  - b\n---\nBody\n"})
        assert any("Broken.md" in e and "Invalid front matter YAML" in e for e in site.errors)
        assert not site.has("_notes/broken.md")


class TestRunFailures:
    """A run whose mapped folders are missing fails instead of converting nothing."""

    def test_missing_source_folder_is_an_error(self, convert):
        site = convert({"notes/A.md": "Body\n"}, mappings={"notes": "_notes", "gone": "_gone"})
        assert any("Source directory not found" in e for e in site.errors)

    def test_missing_output_folder_without_force_is_an_error(self, convert):
        site = convert({"notes/A.md": "Body\n"}, mappings={"notes": "_notes"}, force=False)
        assert any("Output directory not found" in e for e in site.errors)


class TestImages:
    """Every image reference points at the copied image by its site-absolute, percent-encoded URL, and the site keeps only referenced images."""

    IMAGE = {"assets/images/splay results.png": b"\x89PNG fake"}

    def test_markdown_image_with_encoded_name_is_copied_and_linked_absolutely(self, convert):
        site = convert(self.IMAGE | {"notes/Hello.md": "![Figure 1](splay%20results.png)\n"})
        assert "![Figure 1](/assets/img/splay%20results.png)" in site.page("_notes/hello.md")
        assert site.has("assets/img/splay results.png")

    def test_wikilink_image_name_with_spaces_is_percent_encoded(self, convert):
        site = convert(self.IMAGE | {"notes/Hello.md": "![[splay results.png|300]]\n"})
        assert '![Image](/assets/img/splay%20results.png){:width="300"}' in site.page("_notes/hello.md")

    def test_external_image_is_unchanged(self, convert):
        site = convert({"notes/Hello.md": "![logo](https://example.com/logo.png)\n"})
        assert "![logo](https://example.com/logo.png)" in site.page("_notes/hello.md")

    def test_images_no_note_references_are_removed_from_the_site(self, convert):
        site = convert(self.IMAGE | {"notes/Hello.md": "![[splay results.png]]\n"},
                       existing_site={"assets/img/stale.png": b"\x89PNG stale"})
        assert site.errors == []
        assert not site.has("assets/img/stale.png")
        assert site.has("assets/img/splay results.png")

    def test_validation_leaves_site_images_alone(self, convert):
        site = convert({"notes/Hello.md": "Body\n"}, validate=True, existing_site={"assets/img/stale.png": b"\x89PNG stale"})
        assert site.has("assets/img/stale.png")
