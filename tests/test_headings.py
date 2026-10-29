from obsidian_md_converter.headings import HeadingAnchors, strip_heading


class TestStripHeading:
    def test_punctuation_becomes_space_and_runs_collapse(self):
        assert strip_heading("Linearizability (Linearizovatelnost / Atomická):") == "Linearizability Linearizovatelnost Atomická"

    def test_wikilink_markup_in_a_heading_is_stripped(self):
        assert strip_heading("[[koordinace|Synchronizace a koordinace]]") == "koordinace Synchronizace a koordinace"


class TestGfmAnchors:
    """Anchors match what kramdown-parser-gfm produces under Jekyll's defaults."""

    def test_diacritics_survive_and_parentheses_drop(self):
        anchors = HeadingAnchors("## Totální uspořádání (total ordering)\n")
        assert anchors.anchor_for("Totální uspořádání (total ordering)") == "totální-uspořádání-total-ordering"

    def test_spaces_are_not_collapsed(self):
        anchors = HeadingAnchors("#### Linearizability (Linearizovatelnost / Atomická):\n")
        assert anchors.anchor_for("Linearizability (Linearizovatelnost / Atomická)") == "linearizability-linearizovatelnost--atomická"

    def test_code_and_symbols_are_dropped(self):
        anchors = HeadingAnchors("## 2-3 trees & `B+` trees\n")
        assert anchors.anchor_for("2-3 trees & `B+` trees") == "2-3-trees--b-trees"

    def test_repeated_heading_gets_a_numbered_anchor(self):
        anchors = HeadingAnchors("## Prefix key\ntext\n## Other\n## Prefix key\n")
        assert anchors.anchor_for("Prefix key") == "prefix-key"
        assert anchors.anchor_for("Other") == "other"

    def test_numbering_counts_different_headings_that_share_an_anchor(self):
        anchors = HeadingAnchors("## A b\n## A-b\n")
        assert anchors.anchor_for("A b") == "a-b"
        assert anchors.anchor_for("A-b") == "a-b-1"

    def test_heading_with_a_link_takes_its_display_text(self):
        anchors = HeadingAnchors("## [[koordinace v distribuovaném systému|Synchronizace a koordinace]]\n")
        link_heading = "koordinace v distribuovaném systému Synchronizace a koordinace"
        assert anchors.anchor_for(link_heading) == "synchronizace-a-koordinace"

    def test_heading_with_a_markdown_link_takes_its_text(self):
        anchors = HeadingAnchors("## [Stavovost](/notes/stavovost.html)\n")
        assert anchors.anchor_for("[Stavovost](/notes/stavovost.html)") == "stavovost"

    def test_explicit_id_wins(self):
        anchors = HeadingAnchors("## Custom {#my-id}\n")
        assert anchors.anchor_for("Custom") == "my-id"


class TestHeadingLookup:
    def test_link_matches_heading_case_insensitively(self):
        anchors = HeadingAnchors("### [[FLP impossibility]]\n")
        assert anchors.anchor_for("FLP Impossibility") == "flp-impossibility"

    def test_missing_heading_has_no_anchor(self):
        anchors = HeadingAnchors("## Present\n")
        assert anchors.anchor_for("Absent") is None

    def test_headings_inside_code_fences_are_ignored(self):
        anchors = HeadingAnchors("```bash\n# not a heading\n```\n")
        assert anchors.anchor_for("not a heading") is None

    def test_hash_without_space_is_a_tag_not_a_heading(self):
        anchors = HeadingAnchors("#tag\n")
        assert anchors.anchor_for("tag") is None
