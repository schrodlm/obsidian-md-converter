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
