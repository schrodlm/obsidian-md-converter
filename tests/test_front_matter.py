import yaml

from obsidian_md_converter.front_matter import FrontMatter, JekyllFrontMatter
from obsidian_md_converter.front_matter.base import yaml_fields


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


class TestYamlFields:
    def test_each_field_is_one_double_quoted_line(self):
        assert yaml_fields({'layout': 'note', 'title': 'CAP theorem'}) == 'layout: "note"\ntitle: "CAP theorem"'

    def test_quotes_and_backslashes_are_escaped(self):
        title = 'Say "hi" to C:\\temp'
        assert yaml.safe_load(yaml_fields({'title': title})) == {'title': title}

    def test_non_ascii_is_written_as_is(self):
        assert yaml_fields({'title': 'ABA problém'}) == 'title: "ABA problém"'

    def test_title_with_a_quote_keeps_the_front_matter_valid(self):
        fm = FrontMatter('---\ntags: [mff]\n---\nBody\n')
        fm.ensure('The "CAP" theorem')
        assert FrontMatter(fm.serialize()).metadata == {'tags': ['mff'], 'title': 'The "CAP" theorem'}


class TestJekyllEnsure:
    AUTO_ADD = {'jekyll': {'auto_add': True, 'default_layout': 'note'}}

    def test_front_matter_without_layout_gets_the_default_layout(self):
        fm = JekyllFrontMatter('---\ntags:\n  - mff\n---\nBody\n', self.AUTO_ADD)
        fm.ensure("CAP theorem")
        assert fm.get("layout") == "note"
        assert fm.serialize() == '---\ntags:\n  - mff\ntitle: "CAP theorem"\nlayout: "note"\n---\nBody\n'

    def test_existing_layout_is_kept(self):
        fm = JekyllFrontMatter('---\nlayout: course\n---\nBody\n', self.AUTO_ADD)
        fm.ensure("NPRG077")
        assert fm.get("layout") == "course"

    def test_note_without_front_matter_gets_layout_and_title(self):
        fm = JekyllFrontMatter('Body\n', self.AUTO_ADD)
        fm.ensure("2PC")
        assert fm.serialize() == '---\ntitle: "2PC"\nlayout: "note"\n---\n\nBody\n'

    def test_layout_is_not_added_without_auto_add(self):
        fm = JekyllFrontMatter('---\ntags: []\n---\nBody\n', {'jekyll': {'auto_add': False}})
        fm.ensure("x")
        assert fm.get("layout") is None

    def test_note_without_front_matter_gets_a_title_without_auto_add(self):
        fm = JekyllFrontMatter('Body\n', {'jekyll': {'auto_add': False}})
        fm.ensure("2PC")
        assert fm.serialize() == '---\ntitle: "2PC"\n---\n\nBody\n'
