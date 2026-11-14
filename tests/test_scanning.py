from obsidian_md_converter.scanning import scan, scan_body, split_front_matter


class TestFrontMatter:
    def test_front_matter_at_the_top_is_split_from_the_body(self):
        raw, body = split_front_matter("---\ntitle: x\n---\nBody\n")
        assert raw == "title: x"
        assert body == "Body\n"

    def test_dashes_later_in_the_note_are_not_front_matter(self):
        raw, body = split_front_matter("Intro\n---\ntitle: x\n---\n")
        assert raw is None
        assert body == "Intro\n---\ntitle: x\n---\n"

    def test_body_lines_count_from_the_top_of_the_note(self):
        note = scan("---\ntitle: x\n---\n## Heading\n")
        assert note.body_line == 4
        assert note.headings[0].span.line == 4


class TestCode:
    def test_nothing_inside_a_fenced_block_is_a_structure(self):
        note = scan("```\nx = arr[[0]]\nsee [[Target]] and ![[a.png]]\n# not a heading\n```\n")
        assert note.links == []
        assert note.embeds == []
        assert note.headings == []
        assert len(note.code) == 1

    def test_nothing_inside_inline_code_is_a_structure(self):
        note = scan("Use `[[Target]]` or ``a ` [[Other]]`` literally.\n")
        assert note.links == []
        assert len(note.code) == 2

    def test_inline_code_never_crosses_a_blank_line(self):
        note = scan("A stray ` backtick.\n\nA real [[Link]] and `code`.\n")
        assert [link.note for link in note.links] == ["Link"]

    def test_an_unclosed_fence_runs_to_the_end(self):
        note = scan("Before [[A]].\n```\nafter [[B]]\n")
        assert [link.note for link in note.links] == ["A"]

    def test_indented_lines_are_not_code(self):
        note = scan("1. Step\n    2. [[splay|splayujeme]] the node\n")
        assert [link.note for link in note.links] == ["splay"]


class TestLinks:
    def test_link_parts_are_read_from_the_brackets(self):
        link = scan("[[logical contracts#Preconditions | precondition]]\n").links[0]
        assert (link.note, link.heading, link.display) == ("logical contracts", "Preconditions", "precondition")

    def test_same_note_and_nested_heading_links(self):
        same_note, nested = scan("[[#Example]] and [[note#H1#H2]]\n").links
        assert (same_note.note, same_note.heading) == ("", "Example")
        assert nested.heading == "H2"

    def test_link_without_display_text_has_none(self):
        assert scan("[[2PC]]\n").links[0].display is None


class TestEmbedsAndImages:
    def test_embed_size_and_alt_are_read(self):
        width_only, both, alt_and_size = scan("![[a.png|300]] ![[b.png|200x100]] ![[c.png|Caption|500]]\n").embeds
        assert (width_only.target, width_only.width, width_only.height) == ("a.png", "300", None)
        assert (both.width, both.height) == ("200", "100")
        assert (alt_and_size.alt, alt_and_size.width) == ("Caption", "500")

    def test_markdown_image_and_link_are_told_apart(self):
        note = scan("![Fig 1](a%20b.png) and [web](https://example.com)\n")
        assert [(image.alt, image.url) for image in note.markdown_images] == [("Fig 1", "a%20b.png")]
        assert [(link.text, link.url) for link in note.markdown_links] == [("web", "https://example.com")]

    def test_replaceable_structures_are_in_text_order(self):
        note = scan("![x](a.png) [[B]] ![[c.png]]\n")
        assert [structure.span.start for structure in note.replaceable] == [0, 12, 18]


class TestPositions:
    def test_line_and_column_are_counted_in_characters(self):
        link = scan_body("český text\nmore [[ABA problém]]\n").links[0]
        assert (link.span.line, link.span.column) == (2, 6)
        assert "český text\nmore [[ABA problém]]\n"[link.span.start:link.span.end] == "[[ABA problém]]"


class TestHeadings:
    def test_heading_level_and_text(self):
        heading = scan("### Linearizability (Atomická):\n").headings[0]
        assert (heading.level, heading.text) == (3, "Linearizability (Atomická):")

    def test_tag_is_not_a_heading(self):
        assert scan("#tag\n").headings == []
