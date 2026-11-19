import pytest
from pathlib import Path

from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.errors import ConfigError


@pytest.fixture(autouse=True)
def configure_roots(tmp_path):
    """Configure validated path roots so ObsidianPath/OutputPath can be created."""
    vault = tmp_path / "vault"
    output = tmp_path / "output"
    vault.mkdir()
    output.mkdir()
    ObsidianPath.configure_root(vault)
    OutputPath.configure_root(output)
    return vault, output


# ---------------------------------------------------------------------------
# Mapping construction validation
# ---------------------------------------------------------------------------

class TestMappingConstruction:
    def test_requires_source(self):
        with pytest.raises(ValueError, match="source"):
            Mapping(sources=(), destination="dst")

    def test_requires_destination(self):
        with pytest.raises(ValueError, match="destination"):
            Mapping(sources=("src",), destination=None)

    def test_requires_both(self):
        with pytest.raises(ValueError):
            Mapping(sources=(), destination=None)

    def test_empty_string_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            Mapping(sources=("",), destination="dst")

    def test_empty_string_destination_raises(self):
        with pytest.raises(ValueError, match="destination"):
            Mapping(sources=("src",), destination="")

    def test_valid_construction(self):
        m = Mapping(sources=("src",), destination="dst")
        assert m.sources == ("src",)
        assert m.destination == "dst"


# ---------------------------------------------------------------------------
# Mapping.from_yaml
# ---------------------------------------------------------------------------

class TestMappingFromYaml:
    def test_minimal_entry(self, configure_roots):
        vault, output = configure_roots
        entry = {"source": str(vault / "notes"), "destination": str(output / "out")}
        m = Mapping.from_yaml(entry)
        assert [source.path for source in m.sources] == [Path(vault / "notes")]
        assert m.destination.path == Path(output / "out")

    def test_missing_source_raises(self):
        entry = {"destination": "/some/path"}
        with pytest.raises(ConfigError, match="source"):
            Mapping.from_yaml(entry)

    def test_missing_destination_raises(self):
        entry = {"source": "/some/path"}
        with pytest.raises(ConfigError, match="destination"):
            Mapping.from_yaml(entry)

    def test_empty_dict_raises(self):
        with pytest.raises(ConfigError):
            Mapping.from_yaml({})

    def test_unknown_keys_ignored(self, configure_roots):
        """Keys a mapping does not know, such as options an older config still carries, are ignored."""
        vault, output = configure_roots
        entry = {
            "source": str(vault / "notes"),
            "destination": str(output / "out"),
            "transform_links": True,
            "copy_images": True,
        }
        m = Mapping.from_yaml(entry)
        assert m == Mapping((ObsidianPath(vault / "notes"),), OutputPath(output / "out"))

    def test_sources_list_publishes_several_folders(self, configure_roots):
        vault, output = configure_roots
        entry = {"sources": [str(vault / "master"), str(vault / "bachelor")], "destination": str(output / "_courses")}
        assert [source.path for source in Mapping.from_yaml(entry).sources] == [vault / "master", vault / "bachelor"]

    @pytest.mark.parametrize("entry", [
        {"source": "notes", "sources": ["notes"], "destination": "_notes"},
        {"destination": "_notes"},
    ])
    def test_source_and_sources_together_or_neither_raises(self, entry):
        with pytest.raises(ConfigError, match="either 'source"):
            Mapping.from_yaml(entry)

    @pytest.mark.parametrize("sources", [[], "notes"])
    def test_sources_that_is_not_a_non_empty_list_raises(self, sources):
        with pytest.raises(ConfigError, match="must be a list"):
            Mapping.from_yaml({"sources": sources, "destination": "_notes"})

    @pytest.mark.parametrize("url, start", [(None, None), ("/study", "/study"), ("/study/", "/study"), ("/", "")])
    def test_url_is_kept_without_its_trailing_slash(self, configure_roots, url, start):
        vault, output = configure_roots
        entry = {"source": str(vault / "notes"), "destination": str(output / "out")}
        if url is not None:
            entry["url"] = url
        assert Mapping.from_yaml(entry).url == start

    @pytest.mark.parametrize("url", ["study", "", 42])
    def test_url_not_starting_with_a_slash_raises(self, configure_roots, url):
        vault, output = configure_roots
        entry = {"source": str(vault / "notes"), "destination": str(output / "out"), "url": url}
        with pytest.raises(ConfigError, match="must start with '/'"):
            Mapping.from_yaml(entry)


# ---------------------------------------------------------------------------
# Mapping equality and hashing (frozen dataclass)
# ---------------------------------------------------------------------------

def make_mapping(source: str, destination: str) -> Mapping:
    return Mapping((ObsidianPath(source),), OutputPath(destination))


class TestMappingEquality:
    def test_same_paths_are_equal(self):
        assert make_mapping("src", "dst") == make_mapping("src", "dst")

    def test_different_paths_not_equal(self):
        assert make_mapping("src1", "dst") != make_mapping("src2", "dst")

    def test_hashable_for_sets(self):
        assert len({make_mapping("src", "dst"), make_mapping("src", "dst")}) == 1

    def test_different_mappings_in_set(self):
        assert len({make_mapping("src1", "dst1"), make_mapping("src2", "dst2")}) == 2

    def test_frozen_prevents_mutation(self):
        m = make_mapping("src", "dst")
        with pytest.raises(AttributeError):
            m.sources = ("other",)


class TestOutputLocation:
    """A source file keeps its slugified subdirectory below the mapping's destination."""

    def make_mapping(self):
        return Mapping((ObsidianPath("logs"),), OutputPath("_logs"))

    def test_subdirectory_is_relative_to_the_folder_holding_the_file(self):
        mapping = Mapping((ObsidianPath("master"), ObsidianPath("bachelor")), OutputPath("_courses"))
        assert mapping.subdirectory_for(ObsidianPath("bachelor/os/BI-OSY.md").path) == Path("os")

    def test_file_directly_in_source_has_no_subdirectory(self):
        mapping = self.make_mapping()
        assert mapping.subdirectory_for(ObsidianPath("logs/entry.md").path) == Path(".")
        assert mapping.output_directory_for(ObsidianPath("logs/entry.md").path).path == OutputPath("_logs").path

    def test_nested_file_keeps_its_slugified_subdirectory(self):
        mapping = self.make_mapping()
        source = ObsidianPath("logs/Moje Projekty/rentczecher/2026-10-16 Entry.md").path
        assert mapping.subdirectory_for(source) == Path("moje-projekty/rentczecher")
        assert mapping.output_directory_for(source).path == OutputPath("_logs/moje-projekty/rentczecher").path
