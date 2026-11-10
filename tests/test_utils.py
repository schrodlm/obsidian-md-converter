from obsidian_md_converter.utils import slugify_text


class TestSlugifyText:
    def test_accents_are_dropped_and_letters_kept(self):
        assert slugify_text("ABA problém (2)") == "aba-problem-2"

    def test_separators_collapse_into_single_hyphens(self):
        assert slugify_text("  2PC -- two_phase  commit ") == "2pc-two-phase-commit"

    def test_edge_hyphens_are_trimmed(self):
        assert slugify_text("? x ?") == "x"
        assert slugify_text("- intro -") == "intro"
