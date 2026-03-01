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
        assert mf.copy_images is False

    def test_explicit_values(self):
        mf = MappingFlags(transform_links=True, copy_images=True)
        assert mf.transform_links is True
        assert mf.copy_images is True


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
        assert m.flags.copy_images is False

    def test_with_flags(self, configure_roots):
        vault, output = configure_roots
        entry = {
            "source": str(vault / "notes"),
            "destination": str(output / "out"),
            "transform_links": True,
            "copy_images": True,
        }
        m = Mapping.from_yaml(entry)
        assert m.flags.transform_links is True
        assert m.flags.copy_images is True

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
            "transform_links": False,
            "copy_images": 0,
        }
        m = Mapping.from_yaml(entry)
        assert m.flags.transform_links is False
        assert m.flags.copy_images is False


# ---------------------------------------------------------------------------
# Mapping.from_string
# ---------------------------------------------------------------------------

class TestMappingFromString:
    def test_basic_src_dst(self):
        m = Mapping.from_string("src:dst")
        assert m.source == "src"
        assert m.destination == "dst"
        assert m.flags.transform_links is False
        assert m.flags.copy_images is False

    def test_with_single_flag(self):
        m = Mapping.from_string("src:dst:transform_links")
        assert m.flags.transform_links is True
        assert m.flags.copy_images is False

    def test_with_multiple_flags(self):
        m = Mapping.from_string("src:dst:transform_links:copy_images")
        assert m.flags.transform_links is True
        assert m.flags.copy_images is True

    def test_missing_destination_raises(self):
        with pytest.raises(ConfigError, match="src:dst"):
            Mapping.from_string("only_source")

    def test_empty_string_raises(self):
        with pytest.raises(ConfigError):
            Mapping.from_string("")

    def test_invalid_flag_raises(self):
        with pytest.raises(ConfigError, match="No such flag"):
            Mapping.from_string("src:dst:nonexistent_flag")


# ---------------------------------------------------------------------------
# Mapping equality and hashing (frozen dataclass)
# ---------------------------------------------------------------------------

class TestMappingEquality:
    def test_same_paths_are_equal(self):
        m1 = Mapping.from_string("src:dst")
        m2 = Mapping.from_string("src:dst")
        assert m1 == m2

    def test_flags_excluded_from_equality(self):
        """Flags have compare=False, so two Mappings with same paths but different flags are equal."""
        m1 = Mapping.from_string("src:dst")
        m2 = Mapping.from_string("src:dst:transform_links")
        assert m1 == m2

    def test_different_paths_not_equal(self):
        m1 = Mapping.from_string("src1:dst")
        m2 = Mapping.from_string("src2:dst")
        assert m1 != m2

    def test_hashable_for_sets(self):
        m1 = Mapping.from_string("src:dst")
        m2 = Mapping.from_string("src:dst:transform_links")
        # Same paths -> same hash (flags excluded from hash)
        s = {m1, m2}
        assert len(s) == 1

    def test_different_mappings_in_set(self):
        m1 = Mapping.from_string("src1:dst1")
        m2 = Mapping.from_string("src2:dst2")
        s = {m1, m2}
        assert len(s) == 2

    def test_frozen_prevents_mutation(self):
        m = Mapping.from_string("src:dst")
        with pytest.raises(AttributeError):
            m.source = "other"