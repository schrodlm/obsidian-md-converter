"""A run: index every published note, check them all, and only then write the site, replacing its folders at once."""

from pathlib import Path

from obsidian_md_converter.checks import checks_for, fix, validate
from obsidian_md_converter.config import Config
from obsidian_md_converter.index import IndexedNote, NoteIndex, build_index, read_front_matter
from obsidian_md_converter.links import NoteLinks
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.problems import Problem
from obsidian_md_converter.processors import Processor, processor_for
from obsidian_md_converter.references import rewrite_references
from obsidian_md_converter.staging import Staging


def run(config: Config) -> tuple[list[Problem], int]:
    """
    Convert the vault into the site, or with --validate only check it.

    Any error stops the run before the site is touched, and the site's folders are replaced only
    once every page is written.

    Returns:
        Tuple of (problems, number of published notes)
    """
    processor = processor_for(config)
    print("\nIndexing notes...")
    index = build_index(config, processor)
    print(f"Indexed {len(index)} notes")

    problems = validate(index, checks_for(config, processor))
    if config.fix_source:
        problems = fix(problems)
    if config.validate_only or any(problem.is_error for problem in problems):
        return problems, len(index)

    with Staging(config.output_root, replaced_folders(index, config)) as staging:
        write_site(index, processor, config, staging)
        staging.swap()
    return problems, len(index)


def replaced_folders(index: NoteIndex, config: Config) -> list[Path]:
    """
    The site folders a run replaces: the image folder, and each mapping's site folder.

    The site mirrors the vault, so a mapping whose vault folder is gone replaces its site folder with an empty one.
    A mapping with neither folder has nothing to replace.
    """
    folders = [config.output_image_dir.path]
    for mapping_folders in index.folders:
        if mapping_folders.source_exists or mapping_folders.destination_exists:
            folders.append(mapping_folders.mapping.destination.path)
    return folders


def write_site(index: NoteIndex, processor: Processor, config: Config, staging: Staging):
    """Write a page for every indexed note, and copy the images they show, into the staging folder."""
    for mapping in config.source_mappings:
        mapping : Mapping
        notes = index.notes_in(mapping)
        if not notes:
            continue
        print(f"\nProcessing: {mapping.source} -> {mapping.destination}")
        for number, note in enumerate(notes, start=1):
            write_page(note, index, processor, config, staging)
            print(f"Transferred {note.source.name}. [{number}/{len(notes)}]")


def write_page(note: IndexedNote, index: NoteIndex, processor: Processor, config: Config, staging: Staging):
    """Write the page of one checked note, named and linked from what the index holds about it."""
    dst = staging.path_for(note.mapping.destination.path / note.page_path(processor))
    dst.parent.mkdir(parents=True, exist_ok=True)

    fm = read_front_matter(note.source.read_text(encoding='utf-8'), config.front_matter_config)
    # The title is the note's name, which the page's file name loses: "ABA problém" for aba-problem.md
    fm.add({'title': note.source.stem} | fm.defaults())

    dst.write_text(fm.serialize(), encoding='utf-8')

    links = NoteLinks(index=index, own_anchors=note.anchors)
    rewrite_references(dst, config.obsidian_image_dir, config.output_image_dir,
                       staging.path_for(config.output_image_dir.path), links)
