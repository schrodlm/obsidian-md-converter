"""Whole conversions of small vaults, grouped by feature."""

import pytest

from obsidian_md_converter.naming import slugify
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

    def test_post_is_titled_with_its_note_name_not_its_dated_name(self, convert):
        site = convert({"publish/posts/Turing Machine.md": "---\nlayout: post\ndate: 2024-12-20\n---\nBody\n"})
        assert 'title: "Turing Machine"' in site.page("_posts/2024-12-20-turing-machine.md")

    def test_invalid_front_matter_fails_the_note_and_writes_nothing(self, convert):
        site = convert({"notes/Broken.md": "---\ntags:\n      - a\n  - b\n---\nBody\n"})
        assert any("Broken.md" in e and "Invalid front matter YAML" in e for e in site.errors)
        assert not site.has("_notes/broken.md")


class TestValidation:
    """Every check runs before anything is written: an error leaves the site as it was, and --validate reports what a publish would."""

    BROKEN = {"notes/Broken.md": "---\ntags:\n      - a\n  - b\n---\nBody\n"}

    def test_an_error_writes_nothing_and_leaves_the_site_as_it_was(self, convert):
        site = convert(self.BROKEN | {"notes/Hello.md": "Body\n"},
                       existing_site={"_notes/old.md": "Old page\n", "assets/img/old.png": b"\x89PNG old"})
        assert len(site.errors) == 1
        assert not site.has("_notes/hello.md")
        assert site.page("_notes/old.md") == "Old page\n"
        assert site.has("assets/img/old.png")

    def test_validation_reports_a_link_to_a_missing_heading(self, convert):
        site = convert({"notes/Second Note.md": "## Present\n", "notes/Hello.md": "See [[Second Note#Renamed]].\n"}, validate=True)
        assert len(site.warnings) == 1
        assert "Renamed" in site.warnings[0]

    def test_a_problem_names_its_note_line_severity_and_check(self, convert):
        site = convert({"notes/Second Note.md": "## Present\n", "notes/Hello.md": "Intro\n\nSee [[Second Note#Renamed]].\n"})
        assert site.warnings[0].startswith("notes/Hello.md:3: warning: ")
        assert site.warnings[0].endswith(" [missing-heading]")
        assert site.errors == []


class TestStaging:
    """Pages are written beside the site and replace its folders only once all of them are written."""

    def test_a_failure_while_writing_leaves_the_site_as_it_was(self, convert, monkeypatch, tmp_path):
        from obsidian_md_converter import pipeline
        write_page = pipeline.write_page

        def fail_on_second_page(note, *args):
            if note.source.stem == "B":
                raise OSError("No space left on device")
            write_page(note, *args)

        monkeypatch.setattr(pipeline, "write_page", fail_on_second_page)
        old_site = {"_notes/old.md": "Old page\n", "assets/img/old.png": b"\x89PNG old"}
        with pytest.raises(OSError):
            convert({"notes/A.md": "Body\n", "notes/B.md": "Body\n"}, existing_site=old_site)

        site = tmp_path / "site"
        assert sorted(path.relative_to(site).as_posix() for path in site.rglob("*") if path.is_file()) == [
            "_notes/old.md", "assets/img/old.png",
        ]
        assert [path.name for path in site.parent.iterdir() if "staging" in path.name] == []

    def test_a_run_leaves_no_staging_folder_beside_the_site(self, convert):
        site = convert({"notes/A.md": "Body\n"})
        assert [path.name for path in site.root.parent.iterdir() if "staging" in path.name] == []

    def test_a_staging_folder_left_by_a_crashed_run_is_replaced(self, convert, tmp_path):
        leftover = tmp_path / ".site.staging" / "new" / "_notes" / "stale.md"
        leftover.parent.mkdir(parents=True)
        leftover.write_text("Stale\n", encoding="utf-8")
        site = convert({"notes/A.md": "Body\n"})
        assert site.has("_notes/a.md")
        assert not site.has("_notes/stale.md")


class TestTabsInFrontMatter:
    """Tabs in front matter are a warning; the page is written with spaces, and --fix replaces them in the vault too."""

    TABS = {"notes/Hello.md": "---\ntags:\n\t- a\n---\nBody\n"}

    def test_tabs_warn_and_the_page_is_written_with_spaces(self, convert):
        site = convert(self.TABS)
        assert site.warnings == ["notes/Hello.md: warning: Tabs in front matter YAML (use --fix to resolve) [tabs-in-front-matter]"]
        assert "tags:\n  - a\n" in site.page("_notes/hello.md")
        assert "\t" in (site.vault / "notes/Hello.md").read_text(encoding="utf-8")

    def test_fix_replaces_the_tabs_in_the_vault(self, convert):
        site = convert(self.TABS, fix=True)
        assert site.warnings == ["notes/Hello.md: warning: Fixed tabs in front matter YAML [tabs-in-front-matter]"]
        assert (site.vault / "notes/Hello.md").read_text(encoding="utf-8") == "---\ntags:\n  - a\n---\nBody\n"
        assert "tags:\n  - a\n" in site.page("_notes/hello.md")


class TestMissingFolders:
    """A vault folder that is gone publishes nothing and the run goes on; a site folder that is missing stops it without --force."""

    def test_missing_source_folder_warns_and_empties_its_site_folder(self, convert):
        site = convert({"notes/A.md": "Body\n"}, mappings={"notes": "_notes", "gone": "_gone"},
                       existing_site={"_gone/old.md": "Old page\n"})
        assert site.errors == []
        assert site.warnings == [f"warning: Source directory not found: {site.vault / 'gone'}; nothing is published from it [missing-folder]"]
        assert site.has("_notes/a.md")
        assert site.has("_gone")
        assert not site.has("_gone/old.md")

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

    def test_markdown_image_without_alt_text_is_rewritten(self, convert):
        site = convert(self.IMAGE | {"notes/Hello.md": "![](splay%20results.png)\n"})
        assert "![](/assets/img/splay%20results.png)" in site.page("_notes/hello.md")

    def test_external_image_is_unchanged(self, convert):
        site = convert({"notes/Hello.md": "![logo](https://example.com/logo.png)\n"})
        assert "![logo](https://example.com/logo.png)" in site.page("_notes/hello.md")

    def test_images_no_note_references_are_removed_from_the_site(self, convert):
        site = convert(self.IMAGE | {"notes/Hello.md": "![[splay results.png]]\n"},
                       existing_site={"assets/img/stale.png": b"\x89PNG stale"})
        assert site.errors == []
        assert not site.has("assets/img/stale.png")
        assert site.has("assets/img/splay results.png")

    def test_missing_image_warns_on_its_line(self, convert):
        site = convert(self.IMAGE | {"notes/Hello.md": "![[splay results.png]]\n\n![Figure](gone%20image.png)\n"})
        assert site.errors == []
        assert site.warnings == ["notes/Hello.md:3: warning: Missing image: gone image.png; writing its alt text instead [missing-image]"]

    def test_missing_image_is_written_as_its_alt_text(self, convert):
        site = convert({"notes/Hello.md": "A ![[flow.png|Data flow|300]] B ![[gone.png]] C ![Figure](gone.png) D\n"})
        assert site.page("_notes/hello.md").endswith("A Data flow B  C Figure D\n")
        assert not site.has("assets/img/flow.png")

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

    def test_mapping_url_starts_the_address_instead_of_the_default(self, convert):
        site = convert(self.TARGET | {"notes/Hello.md": "See [[Second Note]].\n"}, urls={"notes": "/study"})
        assert "[Second Note](/study/second-note.html)" in site.page("_notes/hello.md")

    def test_root_mapping_url_puts_addresses_at_the_site_root(self, convert):
        site = convert(self.TARGET | {"notes/Hello.md": "See [[Second Note]].\n"}, urls={"notes": "/"})
        assert "[Second Note](/second-note.html)" in site.page("_notes/hello.md")

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

    def test_links_inside_code_are_left_as_written(self, convert):
        site = convert(self.TARGET | {
            "notes/Hello.md": "Use `[[Second Note]]` or:\n\n```\nx = arr[[0]]\nsee [[Second Note]]\n```\nand [[Second Note]].\n",
        })
        page = site.page("_notes/hello.md")
        assert "Use `[[Second Note]]` or:" in page
        assert "x = arr[[0]]\nsee [[Second Note]]\n" in page
        assert "and [Second Note](/notes/second-note.html)." in page

    def test_links_in_front_matter_are_left_as_written(self, convert):
        site = convert(self.TARGET | {"notes/Hello.md": "---\nrelated: \"[[Second Note]]\"\n---\nBody\n"})
        assert 'related: "[[Second Note]]"' in site.page("_notes/hello.md")

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

    def test_clashing_notes_are_an_error_and_nothing_is_written(self, convert):
        site = convert(self.CLASHING | {"notes/Hello World.md": "Body\n"})
        assert len([e for e in site.errors if "would both be written to delivery-protocols.md" in e]) == 1
        assert not site.has("_notes/delivery-protocols.md")
        assert not site.has("_notes/hello-world.md")

    def test_validation_reports_clashes(self, convert):
        site = convert(self.CLASHING, validate=True)
        assert any("would both be written to" in e for e in site.errors)

    def test_same_name_in_different_subfolders_does_not_clash(self, convert):
        site = convert({"notes/alpha/Log.md": "alpha\n", "notes/beta/Log.md": "beta\n"})
        assert site.errors == []
        assert site.has("_notes/alpha/log.md")
        assert site.has("_notes/beta/log.md")


class TestEmptyNotes:
    """A note with nothing below its front matter is a placeholder: not published, and links to it are plain text."""

    def test_empty_note_is_not_published_and_links_to_it_are_text(self, convert):
        site = convert({
            "notes/Load Balancer.md": "\n  \n",
            "notes/Header Only.md": "---\ntags: [x]\n---\n\n",
            "notes/Hello.md": "Behind a [[Load Balancer]].\n",
        })
        assert site.errors == []
        assert not site.has("_notes/load-balancer.md")
        assert not site.has("_notes/header-only.md")
        assert "Behind a Load Balancer." in site.page("_notes/hello.md")

    def test_empty_note_does_not_clash_with_the_real_one(self, convert):
        site = convert({"notes/Delivery (protocols).md": "Real note\n", "notes/Delivery protocols.md": ""})
        assert site.errors == []
        assert "Real note" in site.page("_notes/delivery-protocols.md")


class TestDrafts:
    """A note marked `draft: true` is not published, and links to it are plain text."""

    def test_draft_is_not_published_and_links_to_it_are_text(self, convert):
        site = convert({
            "notes/Work In Progress.md": "---\ndraft: true\n---\nNot ready\n",
            "notes/Hello.md": "See [[Work In Progress]].\n",
        })
        assert site.errors == []
        assert not site.has("_notes/work-in-progress.md")
        assert "See Work In Progress." in site.page("_notes/hello.md")

    def test_draft_false_and_quoted_true_are_published(self, convert):
        site = convert({
            "notes/Finished.md": "---\ndraft: false\n---\nDone\n",
            "notes/Quoted.md": '---\ndraft: "true"\n---\nText, not a boolean\n',
        })
        assert site.errors == []
        assert site.has("_notes/finished.md")
        assert site.has("_notes/quoted.md")


class TestUnusableNames:
    """A note whose name has nothing to build a URL from fails, naming the file; names in other scripts are transliterated."""

    def test_punctuation_only_name_fails_and_writes_nothing(self, convert):
        site = convert({"notes/???.md": "Body\n"})
        assert any("???.md" in e and "rename it" in e for e in site.errors)
        assert not site.has("_notes/.md")

    def test_names_in_other_scripts_are_transliterated(self, convert):
        site = convert({"notes/Привет мир.md": "Body\n"})
        assert site.errors == []
        assert site.has("_notes/privet-mir.md")


class TestSeveralVaultFolders:
    """Several vault folders can feed one site folder; each note keeps its path below its own vault folder."""

    COURSES = {("uni/master/courses", "uni/bachelor/courses"): "_courses"}

    def test_notes_of_both_folders_land_in_one_site_folder(self, convert):
        site = convert({
            "uni/master/courses/NSWI035.md": "See [[BI-OSY]].\n",
            "uni/bachelor/courses/BI-OSY.md": "Body\n",
        }, mappings=self.COURSES)
        assert site.has("_courses/nswi035.md")
        assert site.has("_courses/bi-osy.md")
        assert "[BI-OSY](/courses/bi-osy.html)" in site.page("_courses/nswi035.md")

    def test_same_page_from_both_folders_is_a_clash(self, convert):
        site = convert({
            "uni/master/courses/Operating systems.md": "Master\n",
            "uni/bachelor/courses/Operating systems.md": "Bachelor\n",
        }, mappings=self.COURSES)
        assert any("would both be written to operating-systems.md" in error for error in site.errors)
        assert not site.has("_courses/operating-systems.md")

    def test_note_in_a_nested_mapped_folder_is_published_once_from_it(self, convert):
        site = convert({
            "uni/master/MFF.md": "See [[NSWI035]].\n",
            "uni/master/courses/NSWI035.md": "Body\n",
        }, mappings={"uni": "_uni", "uni/master/courses": "_courses"})
        assert site.has("_uni/master/mff.md")
        assert site.has("_courses/nswi035.md")
        assert not site.has("_uni/master/courses/nswi035.md")
        assert "[NSWI035](/courses/nswi035.html)" in site.page("_uni/master/mff.md")
        assert site.errors == []

    def test_missing_folder_is_reported_while_the_other_still_publishes(self, convert):
        site = convert({"uni/master/courses/NSWI035.md": "Body\n"}, mappings=self.COURSES)
        assert any("uni/bachelor/courses" in warning for warning in site.warnings)
        assert site.has("_courses/nswi035.md")


class TestNormalizedMarkdown:
    """With the markdown processor, links point at files relative to the linking page (docs/normalized-markdown.md)."""

    MARKDOWN = {"type": "markdown"}
    MAPPINGS = {"notes": "notes", "uni/master/courses": "courses", "publish/posts": "posts"}
    NOTES = {
        "notes/tmux.md": "## Prefix key\ntext\n",
        "notes/Hello.md": "See [[tmux]] and [[#Below]].\n\n## Below\n",
        "uni/master/courses/NSWI035.md": "See [[tmux#Prefix key|the prefix]] and ![[a.png|Flow|200]].\n",
        "publish/posts/Turing Machine.md": "---\nlayout: post\ndate: 2024-12-20\n---\nBody\n",
    }

    def convert(self, convert):
        return convert(self.NOTES | {"assets/images/a.png": b"png"}, mappings=self.MAPPINGS, processor=self.MARKDOWN)

    def test_link_into_another_folder_climbs_out_of_its_own(self, convert):
        page = self.convert(convert).page("courses/nswi035.md")
        assert "[the prefix](../notes/tmux.md#prefix-key)" in page

    def test_link_within_a_folder_names_the_file(self, convert):
        assert "[tmux](tmux.md)" in self.convert(convert).page("notes/hello.md")

    def test_link_to_an_own_heading_keeps_only_the_anchor(self, convert):
        assert "[Below](#below)" in self.convert(convert).page("notes/hello.md")

    def test_image_has_an_absolute_path_and_no_size(self, convert):
        assert "![Flow](/assets/img/a.png)." in self.convert(convert).page("courses/nswi035.md")

    def test_post_keeps_its_plain_name_and_gets_no_layout(self, convert):
        site = self.convert(convert)
        page = site.page("posts/turing-machine.md")
        assert 'title: "Turing Machine"' in page
        assert page.count("layout:") == 1


class TestObsidianOnlySyntax:
    """Obsidian syntax with no form in plain Markdown has a defined outcome, for every processor."""

    def test_comment_is_removed_from_the_page(self, convert):
        site = convert({"notes/Hello.md": "Keep %%hide [[Gone]]%% this.\n"})
        assert "Keep  this.\n" in site.page("_notes/hello.md")
        assert site.warnings == []

