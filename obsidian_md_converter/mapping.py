from dataclasses import dataclass, field
from pathlib import Path

from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.errors import ConfigError
from obsidian_md_converter.utils import slugify_text


@dataclass
class MappingFlags():
    transform_links: bool = False
    copy_images: bool = False

# Class to represent mapping of different subdirectories of root to the output dirs and 
# provide a way to define specific settings for them
@dataclass(frozen=True)
class Mapping():
    source: ObsidianPath
    destination: OutputPath
    flags: MappingFlags = field(compare=False, hash=False)

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

        mf = MappingFlags()
        for attr, _ in mf.__dict__.items():
            if entry.get(attr):
                setattr(mf,attr, True)

        return Mapping(source,destination,mf)

    @classmethod
    def from_string(cls, entry: str) -> "Mapping":
        parts = entry.split(':')
        if len(parts) < 2:
            raise ConfigError(f"Mapping must have at least src:dst, got \"{entry}\"")
        src, dst = parts[0], parts[1]
        flags = parts[2:]

        mf = MappingFlags()
        for flag in flags:
            if hasattr(mf, flag):
                setattr(mf, flag, True)
            else:
                raise ConfigError(f"No such flag \"{flag}\" exists.")

        return Mapping(src, dst, mf)

            