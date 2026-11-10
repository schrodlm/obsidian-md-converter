"""Whole conversions of small vaults, grouped by feature."""

from obsidian_md_converter.cli import slugify
from obsidian_md_converter.paths import ObsidianPath


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


class TestLinks:
    """Wikilinks become links to the published page, to the heading's anchor when they name one."""

    TARGET = {"notes/Second Note.md": "## Totální uspořádání (total ordering)\ntext\n"}

    def test_link_to_a_note_points_at_its_page(self, convert):
        site = convert(self.TARGET | {"notes/Hello.md": "See [[Second Note|the second]].\n"})
        assert "[the second](/notes/second-note.html)" in site.page("_notes/hello.md")
        assert site.warnings == []

    def test_link_matches_the_note_name_case_insensitively(self, convert):
        site = convert(self.TARGET | {"notes/Hello.md": "See [[second note]].\n"})
        assert "[second note](/notes/second-note.html)" in site.page("_notes/hello.md")

    def test_link_to_a_heading_points_at_its_anchor(self, convert):
        site = convert(self.TARGET | {"notes/Hello.md": "See [[Second Note#Totální uspořádání (total ordering)|ordering]].\n"})
        assert "[ordering](/notes/second-note.html#totální-uspořádání-total-ordering)" in site.page("_notes/hello.md")
        assert site.warnings == []

    def test_heading_link_without_display_text_shows_note_and_heading(self, convert):
        site = convert(self.TARGET | {"notes/Hello.md": "See [[second note#totální uspořádání (total ordering)]].\n"})
        assert ("[second note > totální uspořádání (total ordering)](/notes/second-note.html#totální-uspořádání-total-ordering)"
                in site.page("_notes/hello.md"))

    def test_link_to_a_missing_heading_points_at_the_page_and_warns(self, convert):
        site = convert(self.TARGET | {"notes/Hello.md": "See [[Second Note#Renamed|ordering]].\n"})
        assert "[ordering](/notes/second-note.html)" in site.page("_notes/hello.md")
        assert len(site.warnings) == 1
        assert "Renamed" in site.warnings[0]

    def test_link_to_a_heading_in_the_same_note_is_a_bare_anchor(self, convert):
        site = convert({"notes/Hello.md": "## Prefix key\nSee [[#Prefix key]].\n"})
        assert "[Prefix key](#prefix-key)" in site.page("_notes/hello.md")
        assert site.warnings == []

    def test_link_to_a_missing_note_is_plain_text_without_warning(self, convert):
        site = convert({"notes/Hello.md": "See [[Not Written Yet|later]] and [[Not Written Yet#Part]].\n"})
        assert "See later and Not Written Yet > Part." in site.page("_notes/hello.md")
        assert site.warnings == []


class TestObsidianSyntaxPassesThrough:
    """Obsidian syntax the site renders itself reaches the output exactly as written."""

    def test_callouts_keep_their_marker_title_and_body(self, convert):
        site = convert({
            "notes/Second Note.md": "Body\n",
            "notes/Hello.md": "> [!note] Pozor\n> - not self-balancing, see [[Second Note]]\n\n> [!important]\n> untitled\n",
        })
        page = site.page("_notes/hello.md")
        assert "> [!note] Pozor\n> - not self-balancing, see [Second Note](/notes/second-note.html)\n" in page
        assert "> [!important]\n> untitled\n" in page


class TestSubfolders:
    """Notes in subfolders of a mapping are published below its destination, keeping their slugified folder path."""

    def test_nested_note_is_written_into_its_subfolder_and_linked_there(self, convert):
        site = convert({
            "notes/Moje Projekty/rentczecher/First Entry.md": "Entry body\n",
            "notes/Hello.md": "See [[First Entry]].\n",
        })
        assert site.errors == []
        assert site.has("_notes/moje-projekty/rentczecher/first-entry.md")
        assert "[First Entry](/notes/moje-projekty/rentczecher/first-entry.html)" in site.page("_notes/hello.md")


class TestOutputNames:
    """The neutral output name is the same for every generator; Jekyll's post convention is applied when the file is written."""

    POST = "---\nlayout: post\ndate: 2024-12-20\n---\nBody\n"

    def test_neutral_name_ignores_post_front_matter(self, tmp_path):
        ObsidianPath.configure_root(tmp_path)
        post = tmp_path / "Turing Machine.md"
        post.write_text(self.POST)
        assert slugify(ObsidianPath(post)) == "turing-machine.md"

    def test_jekyll_post_is_written_under_its_dated_name(self, convert):
        site = convert({"publish/posts/Turing Machine.md": self.POST})
        assert site.has("_posts/2024-12-20-turing-machine.md")

    def test_validation_fails_a_jekyll_post_without_a_date(self, convert):
        site = convert({"publish/posts/Undated.md": "---\nlayout: post\n---\nBody\n"}, validate=True)
        assert any("Undated.md" in e and "date" in e for e in site.errors)


class TestOutputClashes:
    """Two notes that would be written to the same file fail the run, and neither is written."""

    CLASHING = {"notes/Délivery (protocols).md": "Real note\n", "notes/delivery protocols.md": "Leftover\n"}

    def test_clashing_notes_are_an_error_and_neither_is_written(self, convert):
        site = convert(self.CLASHING | {"notes/Hello World.md": "Body\n"})
        assert len([e for e in site.errors if "would both be written to delivery-protocols.md" in e]) == 1
        assert not site.has("_notes/delivery-protocols.md")
        assert site.has("_notes/hello-world.md")

    def test_validation_reports_clashes(self, convert):
        site = convert(self.CLASHING, validate=True)
        assert any("would both be written to" in e for e in site.errors)

    def test_same_name_in_different_subfolders_does_not_clash(self, convert):
        site = convert({"notes/alpha/Log.md": "alpha\n", "notes/beta/Log.md": "beta\n"})
        assert site.errors == []
        assert site.has("_notes/alpha/log.md")
        assert site.has("_notes/beta/log.md")
