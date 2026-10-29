import datetime

from obsidian_md_converter.front_matter import FrontMatter, JekyllFrontMatter


class TestYamlParsing:
    def test_valid_front_matter_parses_into_typed_values(self):
        fm = FrontMatter('---\ntitle: "Notes"\nworth_it: true\ntags:\n  - mff\n---\nBody\n')
        assert fm.yaml_error is None
        assert fm.metadata == {"title": "Notes", "worth_it": True, "tags": ["mff"]}

    def test_unevenly_indented_list_is_reported_with_its_line(self):
        fm = FrontMatter('---\ntitle: x\ntags:\n      - a\n  - b\n---\nBody\n')
        assert fm.yaml_error is not None
        assert "(line 5)" in fm.yaml_error

    def test_unquoted_url_in_flow_list_is_reported(self):
        fm = FrontMatter('---\nreferences: [https://example.com/watch?v=1, https://example.com/b]\n---\n')
        assert fm.yaml_error is not None

    def test_front_matter_that_is_not_a_mapping_is_reported(self):
        fm = FrontMatter('---\n- just\n- a list\n---\nBody\n')
        assert fm.yaml_error is not None

    def test_empty_front_matter_is_valid(self):
        fm = FrontMatter('---\n\n---\nBody\n')
        assert fm.yaml_error is None
        assert fm.metadata == {}

    def test_note_without_front_matter_is_valid(self):
        fm = FrontMatter('Just a body\n')
        assert fm.yaml_error is None

    def test_fixing_tabs_clears_the_yaml_error(self):
        fm = FrontMatter('---\ntags:\n\t- a\n---\nBody\n')
        assert fm.yaml_error is not None
        assert fm.fix_tabs()
        assert fm.yaml_error is None
        assert fm.metadata == {"tags": ["a"]}


class TestJekyllPostSlug:
    def test_iso_date_prefixes_post_slug(self):
        fm = JekyllFrontMatter('---\nlayout: post\ndate: 2024-12-20\n---\n')
        assert isinstance(fm.get("date"), datetime.date)
        assert fm.apply_slug_transform("turing-machine.md") == "2024-12-20-turing-machine.md"

    def test_dotted_date_prefixes_post_slug(self):
        fm = JekyllFrontMatter('---\nlayout: post\ndate: 20.9.2025\n---\n')
        assert fm.apply_slug_transform("an-unexamined-habit.md") == "2025-09-20-an-unexamined-habit.md"

    def test_non_post_slug_is_unchanged(self):
        fm = JekyllFrontMatter('---\nlayout: note\ndate: 2024-12-20\n---\n')
        assert fm.apply_slug_transform("note.md") == "note.md"
