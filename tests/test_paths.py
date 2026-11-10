import pytest

from obsidian_md_converter.paths import ObsidianPath, OutputPath


@pytest.fixture(autouse=True)
def configure_roots(tmp_path):
    ObsidianPath.configure_root(tmp_path / "vault")
    OutputPath.configure_root(tmp_path / "vault")


class TestEquality:
    def test_paths_to_the_same_place_are_equal_and_hash_alike(self):
        assert ObsidianPath("notes") == ObsidianPath("notes")
        assert len({ObsidianPath("notes"), ObsidianPath("notes")}) == 1

    def test_paths_to_different_places_differ(self):
        assert ObsidianPath("notes") != ObsidianPath("posts")

    def test_vault_and_output_paths_never_equal(self):
        assert ObsidianPath("notes") != OutputPath("notes")
