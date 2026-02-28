import pytest
import shutil
import tempfile
from pathlib import Path

from obsidian_md_converter.config import Config, DEFAULT_CONFIG_PATH
from obsidian_md_converter.errors import ConfigError


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary obsidian and output directories."""
    obsidian = tmp_path / "vault"
    output = tmp_path / "output"
    obsidian.mkdir()
    output.mkdir()
    # Image dirs
    (obsidian / "assets" / "images").mkdir(parents=True)
    (output / "assets" / "img").mkdir(parents=True)
    return obsidian, output


@pytest.fixture
def minimal_yaml(tmp_path, tmp_dirs):
    """Write a minimal valid config.yaml and return its path."""
    obsidian, output = tmp_dirs
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""\
paths:
  obsidian_root: "{obsidian}"
  output_root: "{output}"
images:
  source_dir: "assets/images"
  destination_dir: "assets/img"
""")
    return cfg


@pytest.fixture
def full_yaml(tmp_path, tmp_dirs):
    """Write a full config.yaml with all sections."""
    obsidian, output = tmp_dirs
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""\
paths:
  obsidian_root: "{obsidian}"
  output_root: "{output}"
images:
  source_dir: "assets/images"
  destination_dir: "assets/img"
source_mappings:
  - source: "notes"
    destination: "_notes"
    transform_links: true
    copy_images: false
links:
  url_format: "/{{collection}}/{{slug}}/"
  fallback_mode: "warn"
front_matter:
  enabled: true
  type: "jekyll"
  preserve_original_title: true
  jekyll:
    auto_add: true
    default_layout: "note"
runtime:
  validate_only: true
  force_create: true
  fix_source: true
""")
    return cfg


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_loads_from_explicit_path(self, minimal_yaml, tmp_dirs):
        obsidian, output = tmp_dirs
        config = Config(config_path=minimal_yaml)
        assert config.obsidian_root == obsidian
        assert config.output_root == output

    def test_loads_from_default_path(self, minimal_yaml, tmp_dirs, monkeypatch):
        """When no config_path is given, falls back to DEFAULT_CONFIG_PATH."""
        obsidian, output = tmp_dirs
        monkeypatch.setattr(
            "obsidian_md_converter.config.DEFAULT_CONFIG_PATH",
            minimal_yaml,
        )
        config = Config()
        assert config.obsidian_root == obsidian
        assert config.output_root == output

    def test_missing_config_file_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            Config(config_path=missing)

    def test_empty_config_file_raises(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ConfigError, match="not a valid YAML mapping"):
            Config(config_path=empty)

    def test_missing_required_field_raises(self, tmp_path, tmp_dirs):
        obsidian, _ = tmp_dirs
        cfg = tmp_path / "config.yaml"
        # Missing output_root
        cfg.write_text(f"""\
paths:
  obsidian_root: "{obsidian}"
images:
  source_dir: "assets/images"
  destination_dir: "assets/img"
""")
        with pytest.raises(ConfigError, match="output_root"):
            Config(config_path=cfg)


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------

class TestCLIOverrides:
    def test_cli_overrides_optional_yaml_values(self, full_yaml):
        """CLI override wins over YAML for optional fields."""
        config = Config(
            config_path=full_yaml,
            fallback_mode="keep",
            url_format="/custom/{slug}/",
        )
        # YAML has "warn" and "/{collection}/{slug}/" but CLI should win
        assert config.fallback_mode == "keep"
        assert config.url_format == "/custom/{slug}/"

    def test_cli_overrides_yaml_paths(self, minimal_yaml, tmp_path):
        alt_obsidian = tmp_path / "alt_vault"
        alt_output = tmp_path / "alt_output"
        alt_obsidian.mkdir()
        alt_output.mkdir()
        (alt_obsidian / "img").mkdir()
        (alt_output / "img").mkdir()

        config = Config(
            config_path=minimal_yaml,
            obsidian_root=str(alt_obsidian),
            output_root=str(alt_output),
            obsidian_image_dir="img",
            output_image_dir="img",
        )
        assert config.obsidian_root == alt_obsidian.resolve()
        assert config.output_root == alt_output.resolve()

    def test_cli_overrides_runtime_flags(self, minimal_yaml):
        config = Config(
            config_path=minimal_yaml,
            validate_only=True,
            force_create=True,
            fix_source=True,
        )
        assert config.validate_only is True
        assert config.force_create is True
        assert config.fix_source is True

    def test_none_overrides_fall_through_to_yaml(self, full_yaml):
        config = Config(
            config_path=full_yaml,
            validate_only=None,
            force_create=None,
        )
        # YAML has these set to True
        assert config.validate_only is True
        assert config.force_create is True

    def test_cli_only_no_yaml(self, tmp_path, monkeypatch):
        """Config can be created purely from CLI overrides (no YAML file)."""
        obsidian = tmp_path / "vault"
        output = tmp_path / "output"
        obsidian.mkdir()
        output.mkdir()
        (obsidian / "img").mkdir()
        (output / "img").mkdir()

        # Ensure default config doesn't exist
        monkeypatch.setattr(
            "obsidian_md_converter.config.DEFAULT_CONFIG_PATH",
            tmp_path / "nonexistent.yaml",
        )

        config = Config(
            obsidian_root=str(obsidian),
            output_root=str(output),
            obsidian_image_dir="img",
            output_image_dir="img",
        )
        assert config.obsidian_root == obsidian.resolve()
        assert config.output_root == output.resolve()

    def test_unknown_overrides_silently_ignored(self, minimal_yaml):
        """Extra kwargs that don't match any config field are ignored."""
        config = Config(
            config_path=minimal_yaml,
            bogus_key="foo",
            another_unknown=42,
        )
        assert not hasattr(config, "bogus_key")
        assert not hasattr(config, "another_unknown")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestPathResolution:
    def test_absolute_paths_stay_absolute(self, minimal_yaml, tmp_path):
        alt = tmp_path / "absolute_vault"
        alt_out = tmp_path / "absolute_output"
        alt.mkdir()
        alt_out.mkdir()
        (alt / "img").mkdir()
        (alt_out / "img").mkdir()

        config = Config(
            config_path=minimal_yaml,
            obsidian_root=str(alt),
            output_root=str(alt_out),
            obsidian_image_dir="img",
            output_image_dir="img",
        )
        assert config.obsidian_root == alt.resolve()

    def test_relative_paths_resolve_against_config_dir(self, tmp_path):
        vault = tmp_path / "vault"
        output = tmp_path / "output"
        vault.mkdir()
        output.mkdir()
        (vault / "assets" / "images").mkdir(parents=True)
        (output / "assets" / "img").mkdir(parents=True)

        cfg = tmp_path / "config.yaml"
        cfg.write_text("""\
paths:
  obsidian_root: "vault"
  output_root: "output"
images:
  source_dir: "assets/images"
  destination_dir: "assets/img"
""")
        config = Config(config_path=cfg)
        assert config.obsidian_root == (tmp_path / "vault").resolve()
        assert config.output_root == (tmp_path / "output").resolve()

    def test_tilde_expansion(self, minimal_yaml, tmp_path):
        """Ensure ~ in paths gets expanded (doesn't crash)."""
        # We can't easily test ~ resolves to home, but we can ensure
        # _resolve_path doesn't crash on it
        config = Config(config_path=minimal_yaml)
        resolved = config._resolve_config_path("~/some/path")
        assert "~" not in str(resolved)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_values(self, minimal_yaml):
        config = Config(config_path=minimal_yaml)
        assert config.url_format == "/{collection}/{slug}.html"
        assert config.fallback_mode == "text"
        assert config.front_matter_enabled is True
        assert config.preserve_original_title is False
        assert config.validate_only is False
        assert config.force_create is False
        assert config.fix_source is False
        assert config.source_mappings == set()

    def test_yaml_overrides_defaults(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.fallback_mode == "warn"
        assert config.preserve_original_title is True
        assert config.validate_only is True
        assert config.force_create is True
        assert config.fix_source is True
        assert len(config.source_mappings) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_same_root_raises(self, tmp_path):
        same = tmp_path / "same_dir"
        same.mkdir()
        (same / "assets" / "images").mkdir(parents=True)

        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"""\
paths:
  obsidian_root: "{same}"
  output_root: "{same}"
images:
  source_dir: "assets/images"
  destination_dir: "assets/images"
""")
        with pytest.raises(ConfigError, match="must not be the same"):
            Config(config_path=cfg)

    def test_invalid_fallback_mode_raises(self, tmp_path, tmp_dirs):
        obsidian, output = tmp_dirs
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"""\
paths:
  obsidian_root: "{obsidian}"
  output_root: "{output}"
images:
  source_dir: "assets/images"
  destination_dir: "assets/img"
links:
  fallback_mode: "invalid_mode"
""")
        with pytest.raises(ConfigError, match="Invalid fallback_mode"):
            Config(config_path=cfg)

    def test_preserve_title_without_front_matter_raises(self, tmp_path, tmp_dirs):
        obsidian, output = tmp_dirs
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"""\
paths:
  obsidian_root: "{obsidian}"
  output_root: "{output}"
images:
  source_dir: "assets/images"
  destination_dir: "assets/img"
front_matter:
  enabled: false
  preserve_original_title: true
""")
        with pytest.raises(ConfigError, match="preserve_original_title"):
            Config(config_path=cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_resolve_returns_first_non_none(self, minimal_yaml):
        config = Config(config_path=minimal_yaml)
        assert config._resolve(None, None, "found", default="default") == "found"
        assert config._resolve(None, None, default="default") == "default"
        assert config._resolve("first", "second", default="default") == "first"
        assert config._resolve(False, "second", default="default") is False

    def test_resolve_required_raises_on_all_none(self, minimal_yaml):
        config = Config(config_path=minimal_yaml)
        with pytest.raises(ConfigError, match="my_field"):
            config._resolve_required(None, None, name="my_field")

    def test_resolve_required_returns_first_non_none(self, minimal_yaml):
        config = Config(config_path=minimal_yaml)
        assert config._resolve_required(None, "val", name="f") == "val"
        assert config._resolve_required("first", None, name="f") == "first"