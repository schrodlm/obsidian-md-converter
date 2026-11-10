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
