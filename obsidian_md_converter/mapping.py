from dataclasses import dataclass
from pathlib import Path

from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.errors import ConfigError
from obsidian_md_converter.naming import slugify_text


# A vault directory published into an output directory
@dataclass(frozen=True)
class Mapping():
    source: ObsidianPath
    destination: OutputPath
    # How its pages' addresses start, such as '/study', when not the processor's default for the destination.
    url: str | None = None

    def __post_init__(self):
        if not self.source:
            raise ValueError("Mapping requires a source.")
        if not self.destination:
            raise ValueError("Mapping requires a destination.")

    def subdirectory_for(self, source_file: Path) -> Path:
        """The slugified subdirectory a source file keeps below the destination ('.' for a file directly in the source)."""
        relative_parent = Path(source_file).parent.relative_to(Path(self.source))
        return Path(*(slugify_text(part) for part in relative_parent.parts))

    def output_directory_for(self, source_file: Path) -> OutputPath:
        """The directory a source file is written into."""
        return self.destination / self.subdirectory_for(source_file)

    @classmethod
    def from_yaml(cls, entry : dict) -> "Mapping":
        required_keys = ['source', 'destination']
        
        for key in required_keys:
            if key not in entry:
                raise ConfigError(f"Required key {key} missing.")
        
        source = ObsidianPath(entry.get('source'))
        destination = OutputPath(entry.get('destination'))

        url = entry.get('url')
        if url is not None and not (isinstance(url, str) and url.startswith('/')):
            raise ConfigError(f"Mapping url must start with '/', such as '/study': {url!r}")
        # '/study/' and '/study' start the same addresses; the root '/' starts them at ''.
        return Mapping(source, destination, None if url is None else url.rstrip('/'))
