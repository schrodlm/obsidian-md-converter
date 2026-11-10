import pytest
from pathlib import Path

from obsidian_md_converter.mapping import Mapping, MappingFlags
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
# MappingFlags
# ---------------------------------------------------------------------------

class TestMappingFlags:
    def test_defaults_are_false(self):
        mf = MappingFlags()
        assert mf.transform_links is False

    def test_explicit_values(self):
        mf = MappingFlags(transform_links=True)
        assert mf.transform_links is True


# ---------------------------------------------------------------------------
# Mapping construction validation
# ---------------------------------------------------------------------------

class TestMappingConstruction:
    def test_requires_source(self):
        with pytest.raises(ValueError, match="source"):
            Mapping(source=None, destination="dst", flags=MappingFlags())

    def test_requires_destination(self):
        with pytest.raises(ValueError, match="destination"):
            Mapping(source="src", destination=None, flags=MappingFlags())

    def test_requires_both(self):
        with pytest.raises(ValueError):
            Mapping(source=None, destination=None, flags=MappingFlags())

    def test_empty_string_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            Mapping(source="", destination="dst", flags=MappingFlags())

    def test_empty_string_destination_raises(self):
        with pytest.raises(ValueError, match="destination"):
            Mapping(source="src", destination="", flags=MappingFlags())

    def test_valid_construction(self):
        m = Mapping(source="src", destination="dst", flags=MappingFlags())
        assert m.source == "src"
        assert m.destination == "dst"


# ---------------------------------------------------------------------------
# Mapping.from_yaml
# ---------------------------------------------------------------------------

class TestMappingFromYaml:
    def test_minimal_entry(self, configure_roots):
        vault, output = configure_roots
        entry = {"source": str(vault / "notes"), "destination": str(output / "out")}
        m = Mapping.from_yaml(entry)
        assert m.source.path == Path(vault / "notes")
        assert m.destination.path == Path(output / "out")
        assert m.flags.transform_links is False

    def test_with_flags(self, configure_roots):
        vault, output = configure_roots
        entry = {
            "source": str(vault / "notes"),
            "destination": str(output / "out"),
            "transform_links": True,
        }
        m = Mapping.from_yaml(entry)
        assert m.flags.transform_links is True

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

    def test_unknown_flags_ignored(self, configure_roots):
        vault, output = configure_roots
        entry = {
            "source": str(vault / "notes"),
            "destination": str(output / "out"),
            "unknown_flag": True,
        }
        m = Mapping.from_yaml(entry)
        assert not hasattr(m.flags, "unknown_flag")

    def test_falsy_flag_values_not_set(self, configure_roots):
        vault, output = configure_roots
        entry = {
            "source": str(vault / "notes"),
            "destination": str(output / "out"),
            "transform_links": 0,
        }
        m = Mapping.from_yaml(entry)
        assert m.flags.transform_links is False


# ---------------------------------------------------------------------------
# Mapping equality and hashing (frozen dataclass)
# ---------------------------------------------------------------------------

def make_mapping(source: str, destination: str, transform_links: bool = False) -> Mapping:
    return Mapping(ObsidianPath(source), OutputPath(destination), MappingFlags(transform_links=transform_links))


class TestMappingEquality:
    def test_same_paths_are_equal(self):
        assert make_mapping("src", "dst") == make_mapping("src", "dst")

    def test_flags_excluded_from_equality(self):
        """Flags have compare=False, so two Mappings with same paths but different flags are equal."""
        assert make_mapping("src", "dst") == make_mapping("src", "dst", transform_links=True)

    def test_different_paths_not_equal(self):
        assert make_mapping("src1", "dst") != make_mapping("src2", "dst")

    def test_hashable_for_sets(self):
        # Same paths -> same hash (flags excluded from hash)
        assert len({make_mapping("src", "dst"), make_mapping("src", "dst", transform_links=True)}) == 1

    def test_different_mappings_in_set(self):
        assert len({make_mapping("src1", "dst1"), make_mapping("src2", "dst2")}) == 2

    def test_frozen_prevents_mutation(self):
        m = make_mapping("src", "dst")
        with pytest.raises(AttributeError):
            m.source = "other"


class TestOutputLocation:
    """A source file keeps its slugified subdirectory below the mapping's destination."""

    def make_mapping(self):
        return Mapping(ObsidianPath("logs"), OutputPath("_logs"), MappingFlags())

    def test_file_directly_in_source_has_no_subdirectory(self):
        mapping = self.make_mapping()
        assert mapping.subdirectory_for(ObsidianPath("logs/entry.md").path) == Path(".")
        assert mapping.output_directory_for(ObsidianPath("logs/entry.md").path).path == OutputPath("_logs").path

    def test_nested_file_keeps_its_slugified_subdirectory(self):
        mapping = self.make_mapping()
        source = ObsidianPath("logs/Moje Projekty/rentczecher/2026-10-16 Entry.md").path
        assert mapping.subdirectory_for(source) == Path("moje-projekty/rentczecher")
        assert mapping.output_directory_for(source).path == OutputPath("_logs/moje-projekty/rentczecher").path
