from dataclasses import dataclass
from pathlib import Path

from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.errors import ConfigError
from obsidian_md_converter.naming import slugify_text


# One or more vault directories published into one output directory
@dataclass(frozen=True)
class Mapping():
    sources: tuple[ObsidianPath, ...]
    destination: OutputPath
    # How its pages' addresses start, such as '/study', when not the processor's default for the destination.
    url: str | None = None

    def __post_init__(self):
        if not self.sources or not all(self.sources):
            raise ValueError("Mapping requires a source.")
        if not self.destination:
            raise ValueError("Mapping requires a destination.")

    def source_of(self, source_file: Path) -> ObsidianPath:
        """The vault folder of this mapping that holds a source file."""
        return next(source for source in self.sources if Path(source_file).is_relative_to(Path(source)))

    def subdirectory_for(self, source_file: Path) -> Path:
        """The slugified subdirectory a source file keeps below the destination ('.' for a file directly in its source)."""
        relative_parent = Path(source_file).parent.relative_to(Path(self.source_of(source_file)))
        return Path(*(slugify_text(part) for part in relative_parent.parts))

    def output_directory_for(self, source_file: Path) -> OutputPath:
        """The directory a source file is written into."""
        return self.destination / self.subdirectory_for(source_file)

    @classmethod
    def from_yaml(cls, entry : dict) -> "Mapping":
        """
        A mapping from its config entry: one vault folder as 'source', or several as 'sources', and its site folder.

        Examples:
            {source: notes, destination: _notes}
            {sources: [uni/master/courses, uni/bachelor/courses], destination: _courses, url: /courses}
        """
        if 'destination' not in entry:
            raise ConfigError("Required key destination missing.")
        if ('source' in entry) == ('sources' in entry):
            raise ConfigError("A mapping needs either 'source: <folder>' or 'sources: [<folder>, ...]', not both.")
        names = [entry['source']] if 'source' in entry else entry['sources']
        if not isinstance(names, list) or not names:
            raise ConfigError(f"Mapping sources must be a list of vault folders: {names!r}")

        sources = tuple(ObsidianPath(name) for name in names)
        destination = OutputPath(entry.get('destination'))

        url = entry.get('url')
        if url is not None and not (isinstance(url, str) and url.startswith('/')):
            raise ConfigError(f"Mapping url must start with '/', such as '/study': {url!r}")
        # '/study/' and '/study' start the same addresses; the root '/' starts them at ''.
        return Mapping(sources, destination, None if url is None else url.rstrip('/'))
