from pathlib import Path
import yaml

from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.errors import ConfigError
from obsidian_md_converter.utils import nested_get

# Default config location (next to the package)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.resolve() / "config.yaml"

VALID_FALLBACK_MODES = {"text", "keep", "warn"}


class Config:
    def __init__(self, config_path: Path | None = None, **overrides):
        # Load YAML (optional — might not exist if everything is passed via CLI)
        if config_path:
            self._raw_config = self._load_yaml(config_path)
        elif DEFAULT_CONFIG_PATH.exists():
            self._raw_config = self._load_yaml(DEFAULT_CONFIG_PATH)
        else:
            self._raw_config = {}
            self._config_dir = Path.cwd()

        self._overrides = overrides
        self._parse()
        self._validate()

    def _load_yaml(self, config_path: Path) -> dict:
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self._config_dir = config_path.parent

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ConfigError("Config file is empty or not a valid YAML mapping")

        return config

    def _parse(self):
        # --- Paths (required — from YAML or CLI) ---
        self.obsidian_root = self._resolve_config_path(self._resolve_required(
            self._overrides.get('obsidian_root'),
            nested_get(self._raw_config,'paths', 'obsidian_root'),
            name='obsidian_root',
        ))
        self.output_root = self._resolve_config_path(self._resolve_required(
            self._overrides.get('output_root'),
            nested_get(self._raw_config,'paths', 'output_root'),
            name='output_root',
        ))

        # Set roots for validated path classes
        ObsidianPath.configure_root(self.obsidian_root)
        OutputPath.configure_root(self.output_root)

        # --- Images (required — relative to roots) ---
        self.obsidian_image_dir = ObsidianPath(
            self.obsidian_root / self._resolve_required(
                self._overrides.get('obsidian_image_dir'),
                nested_get(self._raw_config,'images', 'source_dir'),
                name='obsidian_image_dir',
            )
        )
        self.output_image_dir = OutputPath(
            self.output_root / self._resolve_required(
                self._overrides.get('output_image_dir'),
                nested_get(self._raw_config,'images', 'destination_dir'),
                name='output_image_dir',
            )
        )

        # --- Source mappings (optional — can be provided via CLI instead) ---
        #TODO: Introduce Mapping class
        self.source_mappings = self._resolve(
            self._overrides.get('source_mappings'),
            nested_get(self._raw_config,'source_mappings'),
            default=[],
        )

        # --- Links (optional with defaults) ---
        #TODO: Introduce link class
        self.url_format = self._resolve(
            self._overrides.get('url_format'),
            nested_get(self._raw_config,'links', 'url_format'),
            default="/{collection}/{slug}.html",
        )
        self.fallback_mode = self._resolve(
            self._overrides.get('fallback_mode'),
            nested_get(self._raw_config,'links', 'fallback_mode'),
            default="text",
        )

        # --- Front matter ---
        #TODO: Introduce FrontMatterConfig class
        # Passed directly to create_front_matter() factory — Config doesn't
        # need to know about Jekyll-specific or other type-specific details.
        self.front_matter_enabled = self._resolve(
            self._overrides.get('front_matter_enabled'),
            nested_get(self._raw_config,'front_matter', 'enabled'),
            default=True,
        )
        self.front_matter_config = self._resolve(
            self._overrides.get('front_matter_config'),
            nested_get(self._raw_config,'front_matter'),
            default={},
        )
        self.preserve_original_title = self._resolve(
            self._overrides.get('preserve_original_title'),
            nested_get(self._raw_config,'front_matter', 'preserve_original_title'),
            default=False,
        )

        # --- Runtime flags (YAML default, CLI wins) ---
        self.validate_only = self._resolve(
            self._overrides.get('validate_only'),
            nested_get(self._raw_config,'runtime', 'validate_only'),
            default=False,
        )
        self.force_create = self._resolve(
            self._overrides.get('force_create'),
            nested_get(self._raw_config,'runtime', 'force_create'),
            default=False,
        )
        self.fix_source = self._resolve(
            self._overrides.get('fix_source'),
            nested_get(self._raw_config,'runtime', 'fix_source'),
            default=False,
        )

    def _validate(self):
        """Check that config values make sense together."""
        if self.obsidian_root == self.output_root:
            raise ConfigError(
                "'obsidian_root' and 'output_root' must not be the same directory. "
                "This would overwrite your Obsidian vault."
            )

        if self.fallback_mode not in VALID_FALLBACK_MODES:
            raise ConfigError(
                f"Invalid fallback_mode: '{self.fallback_mode}'. "
                f"Must be one of: {', '.join(sorted(VALID_FALLBACK_MODES))}"
            )

        if not self.front_matter_enabled and self.preserve_original_title:
            raise ConfigError(
                "'preserve_original_title' is enabled but front matter is disabled. "
                "Cannot preserve titles without front matter processing."
            )

    # --- Helpers ---

    def _resolve_config_path(self, raw: str) -> Path:
        """Resolve a path — expand ~, absolute stays absolute, relative resolves against config dir."""
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self._config_dir / p).resolve()

    def _resolve(self, *values, default=None):
        """Return first non-None value, or default."""
        for v in values:
            if v is not None:
                return v
        return default

    def _resolve_required(self, *values, name: str):
        """Return first non-None value, raise ConfigError if all are None."""
        for v in values:
            if v is not None:
                return v
        raise ConfigError(
            f"Missing required config: '{name}'. "
            "Set in config.yaml or pass via CLI."
        )