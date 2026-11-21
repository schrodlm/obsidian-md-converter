import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from obsidian_md_converter.errors import ConfigError
from obsidian_md_converter.front_matter import FrontMatter
from obsidian_md_converter.processors import JekyllProcessor, processor_for
from obsidian_md_converter.processors.base import LinkTarget, PageImage, PageLink


class TestJekyllPostSlug:
    def test_iso_date_prefixes_post_slug(self):
        fm = FrontMatter('---\nlayout: post\ndate: 2024-12-20\n---\n')
        assert isinstance(fm.get("date"), datetime.date)
        assert JekyllProcessor().page_path(Path("turing-machine.md"), fm.metadata) == Path("2024-12-20-turing-machine.md")

    def test_dotted_date_prefixes_post_slug(self):
        fm = FrontMatter('---\nlayout: post\ndate: 20.9.2025\n---\n')
        assert JekyllProcessor().page_path(Path("an-unexamined-habit.md"), fm.metadata) == Path("2025-09-20-an-unexamined-habit.md")

    def test_non_post_slug_is_unchanged(self):
        fm = FrontMatter('---\nlayout: note\ndate: 2024-12-20\n---\n')
        assert JekyllProcessor().page_path(Path("note.md"), fm.metadata) == Path("note.md")


class TestJekyllAddress:
    JEKYLL = JekyllProcessor()

    def test_collection_folder_loses_its_underscore(self):
        assert self.JEKYLL.link_address(LinkTarget(self.JEKYLL.default_start("_notes"), Path("tmux.md"), Path())) == "/notes/tmux.html"

    def test_default_start_drops_the_collection_underscore(self):
        assert self.JEKYLL.default_start("_courses") == "/courses"

    def test_pretty_permalink_ends_the_address_in_a_slash(self):
        assert JekyllProcessor(permalink="pretty").link_address(LinkTarget("/courses", Path("mff/nprg077.md"), Path())) == "/courses/mff/nprg077/"

    def test_subfolders_stay_in_the_address(self):
        assert self.JEKYLL.link_address(LinkTarget("/courses", Path("mff/nprg077.md"), Path())) == "/courses/mff/nprg077.html"


class TestJekyllAnchorIds:
    JEKYLL = JekyllProcessor()

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


class TestJekyllFrontMatterDefaults:
    def test_default_layout_is_the_only_default(self):
        assert JekyllProcessor(default_layout="note").front_matter_defaults() == {"layout": "note"}

    def test_no_default_layout_gives_no_defaults(self):
        assert JekyllProcessor().front_matter_defaults() == {}


class TestJekyllLinkMarkdown:
    JEKYLL = JekyllProcessor()

    def test_link_is_ordinary_markdown(self):
        link = PageLink("the prefix", "/notes/tmux.html#prefix-key")
        assert self.JEKYLL.link_markdown(link) == "[the prefix](/notes/tmux.html#prefix-key)"

    def test_link_within_the_page_keeps_its_anchor_only_url(self):
        assert self.JEKYLL.link_markdown(PageLink("Prefix key", "#prefix-key")) == "[Prefix key](#prefix-key)"


class TestJekyllImageMarkdown:
    JEKYLL = JekyllProcessor()

    def test_image_without_a_size_is_plain_markdown(self):
        assert self.JEKYLL.image_markdown(PageImage("Fig", "/assets/img/b%20c.png")) == "![Fig](/assets/img/b%20c.png)"

    def test_width_is_a_kramdown_attribute(self):
        image = PageImage("Image", "/assets/img/a.png", width="200")
        assert self.JEKYLL.image_markdown(image) == '![Image](/assets/img/a.png){:width="200"}'

    def test_width_and_height_are_kramdown_attributes(self):
        image = PageImage("Flow", "/assets/img/a.png", width="200", height="100")
        assert self.JEKYLL.image_markdown(image) == '![Flow](/assets/img/a.png){:width="200" height="100"}'

    def test_height_without_a_width_is_not_written(self):
        image = PageImage("Flow", "/assets/img/a.png", height="100")
        assert self.JEKYLL.image_markdown(image) == "![Flow](/assets/img/a.png)"


class TestProcessorFor:
    @staticmethod
    def config(processor: dict) -> SimpleNamespace:
        """The Config field processor_for reads."""
        return SimpleNamespace(processor_config=processor)

    def test_jekyll_gets_its_default_layout(self):
        processor = processor_for(self.config({"type": "jekyll", "default_layout": "note"}))
        assert isinstance(processor, JekyllProcessor)
        assert processor.front_matter_defaults() == {"layout": "note"}

    def test_no_processor_section_is_jekyll_without_a_default_layout(self):
        processor = processor_for(self.config({}))
        assert isinstance(processor, JekyllProcessor)
        assert processor.front_matter_defaults() == {}

    def test_unknown_type_raises(self):
        with pytest.raises(ConfigError, match="Unknown processor type 'astro'"):
            processor_for(self.config({"type": "astro"}))

    def test_pretty_permalink_reaches_the_processor(self):
        processor = processor_for(self.config({"type": "jekyll", "permalink": "pretty"}))
        assert processor.link_address(LinkTarget("/notes", Path("tmux.md"), Path())) == "/notes/tmux/"

    def test_unknown_permalink_raises(self):
        with pytest.raises(ConfigError, match="Unknown jekyll permalink 'date'"):
            processor_for(self.config({"type": "jekyll", "permalink": "date"}))

    def test_unknown_option_raises(self):
        with pytest.raises(ConfigError, match="default_layuot"):
            processor_for(self.config({"type": "jekyll", "default_layuot": "note"}))
