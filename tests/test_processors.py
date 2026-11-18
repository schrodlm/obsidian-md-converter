import datetime
from pathlib import Path

from obsidian_md_converter.front_matter import FrontMatter
from obsidian_md_converter.processors import JekyllProcessor


class TestJekyllPostSlug:
    def test_iso_date_prefixes_post_slug(self):
        fm = FrontMatter('---\nlayout: post\ndate: 2024-12-20\n---\n')
        assert isinstance(fm.get("date"), datetime.date)
        assert JekyllProcessor("/{collection}/{slug}.html").page_path(Path("turing-machine.md"), fm.metadata) == Path("2024-12-20-turing-machine.md")

    def test_dotted_date_prefixes_post_slug(self):
        fm = FrontMatter('---\nlayout: post\ndate: 20.9.2025\n---\n')
        assert JekyllProcessor("/{collection}/{slug}.html").page_path(Path("an-unexamined-habit.md"), fm.metadata) == Path("2025-09-20-an-unexamined-habit.md")

    def test_non_post_slug_is_unchanged(self):
        fm = FrontMatter('---\nlayout: note\ndate: 2024-12-20\n---\n')
        assert JekyllProcessor("/{collection}/{slug}.html").page_path(Path("note.md"), fm.metadata) == Path("note.md")


class TestJekyllAddress:
    JEKYLL = JekyllProcessor("/{collection}/{slug}.html")

    def test_collection_folder_loses_its_underscore(self):
        assert self.JEKYLL.address(Path("tmux.md"), "_notes") == "/notes/tmux.html"

    def test_subfolders_stay_in_the_address(self):
        assert self.JEKYLL.address(Path("mff/nprg077.md"), "_courses") == "/courses/mff/nprg077.html"

    def test_url_format_decides_the_shape(self):
        assert JekyllProcessor("/{collection}/{slug}/").address(Path("tmux.md"), "_notes") == "/notes/tmux/"


class TestJekyllAnchorIds:
    JEKYLL = JekyllProcessor("/{collection}/{slug}.html")

    def test_text_is_lowercased_and_spaces_become_hyphens(self):
        assert self.JEKYLL.anchor_ids(["Prefix key"]) == ["prefix-key"]

    def test_punctuation_is_dropped(self):
        assert self.JEKYLL.anchor_ids(["2-3 trees & B+"]) == ["2-3-trees--b"]

    def test_repeats_are_numbered_across_the_page(self):
        assert self.JEKYLL.anchor_ids(["Prefix key", "Other", "Prefix key", "Prefix key"]) == [
            "prefix-key", "other", "prefix-key-1", "prefix-key-2",
        ]

    def test_letters_with_diacritics_are_kept(self):
        assert self.JEKYLL.anchor_ids(["Totální uspořádání"]) == ["totální-uspořádání"]
