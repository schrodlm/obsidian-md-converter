import datetime
from pathlib import Path

from obsidian_md_converter.front_matter import JekyllFrontMatter
from obsidian_md_converter.processors import JekyllProcessor


class TestJekyllPostSlug:
    def test_iso_date_prefixes_post_slug(self):
        fm = JekyllFrontMatter('---\nlayout: post\ndate: 2024-12-20\n---\n')
        assert isinstance(fm.get("date"), datetime.date)
        assert JekyllProcessor().page_path(Path("turing-machine.md"), fm.metadata) == Path("2024-12-20-turing-machine.md")

    def test_dotted_date_prefixes_post_slug(self):
        fm = JekyllFrontMatter('---\nlayout: post\ndate: 20.9.2025\n---\n')
        assert JekyllProcessor().page_path(Path("an-unexamined-habit.md"), fm.metadata) == Path("2025-09-20-an-unexamined-habit.md")

    def test_non_post_slug_is_unchanged(self):
        fm = JekyllFrontMatter('---\nlayout: note\ndate: 2024-12-20\n---\n')
        assert JekyllProcessor().page_path(Path("note.md"), fm.metadata) == Path("note.md")
