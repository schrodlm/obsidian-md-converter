"""Which notes a mapping publishes, and whether any two of them would be written to the same file."""

from pathlib import Path

from obsidian_md_converter.errors import ConversionError
from obsidian_md_converter.front_matter import FrontMatter
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.naming import slugify
from obsidian_md_converter.paths import ObsidianPath


def get_directory_md_files(directory: Path) -> list[Path]:
    """The markdown files a mapping publishes: every one under its source directory, subdirectories included."""
    return [path for path in directory.rglob("*.md") if is_published(path)]


def is_published(path: Path) -> bool:
    """A note is published unless it is a draft (`draft: true`) or a placeholder with nothing below its front matter."""
    fm = FrontMatter(path.read_text(encoding='utf-8'))
    return fm.get('draft') is not True and bool(fm.body.strip())


def find_output_clashes(source_files: list[Path], mapping: Mapping) -> dict[Path, list[Path]]:
    """Group the source files that share a neutral output path, keeping only the groups of two or more."""
    sources_by_output: dict[Path, list[Path]] = {}
    for source_file in source_files:
        try:
            output = mapping.subdirectory_for(source_file) / slugify(ObsidianPath(source_file))
        except ConversionError:
            # Its conversion reports the unusable name.
            continue
        sources_by_output.setdefault(output, []).append(source_file)
    return {output: sources for output, sources in sources_by_output.items() if len(sources) > 1}
