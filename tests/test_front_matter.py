import yaml

from obsidian_md_converter.front_matter import FrontMatter
from obsidian_md_converter.front_matter.base import yaml_fields
from obsidian_md_converter.processors import JekyllProcessor


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
        fm.add({'title': 'The "CAP" theorem'})
        assert FrontMatter(fm.serialize()).metadata == {'tags': ['mff'], 'title': 'The "CAP" theorem'}


class TestAdd:
    def test_only_missing_fields_are_added_at_the_end(self):
        fm = FrontMatter('---\nlayout: course\n---\nBody\n')
        fm.add({'title': 'NPRG077', 'layout': 'note'})
        assert fm.serialize() == '---\nlayout: course\ntitle: "NPRG077"\n---\nBody\n'

    def test_nothing_missing_leaves_the_content_unchanged(self):
        content = '---\ntitle: Kept\n---\nBody\n'
        fm = FrontMatter(content)
        fm.add({'title': 'Other'})
        assert fm.serialize() == content

    def test_note_without_front_matter_gets_one_holding_the_fields(self):
        fm = FrontMatter('Body\n')
        fm.add({'title': '2PC'})
        assert fm.serialize() == '---\ntitle: "2PC"\n---\n\nBody\n'
        assert fm.metadata == {'title': '2PC'}

    def test_adding_nothing_to_a_note_without_front_matter_creates_none(self):
        fm = FrontMatter('Body\n')
        fm.add({})
        assert fm.serialize() == 'Body\n'


class TestJekyllDefaults:
    WITH_LAYOUT = JekyllProcessor("/{collection}/{slug}.html", default_layout="note")
    WITHOUT_LAYOUT = JekyllProcessor("/{collection}/{slug}.html")

    def test_front_matter_without_layout_gets_the_default_layout(self):
        fm = FrontMatter('---\ntags:\n  - mff\n---\nBody\n')
        fm.add({"title": "CAP theorem"} | self.WITH_LAYOUT.front_matter_defaults())
        assert fm.get("layout") == "note"
        assert fm.serialize() == '---\ntags:\n  - mff\ntitle: "CAP theorem"\nlayout: "note"\n---\nBody\n'

    def test_existing_layout_is_kept(self):
        fm = FrontMatter('---\nlayout: course\n---\nBody\n')
        fm.add({"title": "NPRG077"} | self.WITH_LAYOUT.front_matter_defaults())
        assert fm.get("layout") == "course"

    def test_note_without_front_matter_gets_layout_and_title(self):
        fm = FrontMatter('Body\n')
        fm.add({"title": "2PC"} | self.WITH_LAYOUT.front_matter_defaults())
        assert fm.serialize() == '---\ntitle: "2PC"\nlayout: "note"\n---\n\nBody\n'

    def test_layout_is_not_added_without_auto_add(self):
        fm = FrontMatter('---\ntags: []\n---\nBody\n')
        fm.add({"title": "x"} | self.WITHOUT_LAYOUT.front_matter_defaults())
        assert fm.get("layout") is None

    def test_note_without_front_matter_gets_a_title_without_auto_add(self):
        fm = FrontMatter('Body\n')
        fm.add({"title": "2PC"} | self.WITHOUT_LAYOUT.front_matter_defaults())
        assert fm.serialize() == '---\ntitle: "2PC"\n---\n\nBody\n'
