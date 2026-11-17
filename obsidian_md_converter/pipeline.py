"""A run: index every published note, check them all, and only then write the site."""

import os
from pathlib import Path

from obsidian_md_converter.checks import checks_for, fix, validate
from obsidian_md_converter.config import Config
from obsidian_md_converter.front_matter import FrontMatter, front_matter_class
from obsidian_md_converter.index import IndexedNote, NoteIndex, build_index, read_front_matter
from obsidian_md_converter.links import NoteLinks
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.paths import OutputPath
from obsidian_md_converter.problems import Problem
from obsidian_md_converter.references import rewrite_references


# DANGEROUS method, use with care!
# Usable only in this directory
def remove_contents_of(directory: OutputPath):
    if not directory.is_relative_to(OutputPath._root):
        raise RuntimeError("Trying to remove contents outside of this project! Aborted.")

    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            (Path(root) / name).unlink()
        for name in dirs:
            (Path(root) / name).rmdir()


def run(config: Config) -> tuple[list[Problem], int]:
    """
    Convert the vault into the site, or with --validate only check it.

    Any error stops the run before the site is touched.

    Returns:
        Tuple of (problems, number of published notes)
    """
    print("\nIndexing notes...")
    index = build_index(config)
    print(f"Indexed {len(index)} notes")

    problems = validate(index, checks_for(config))
    if config.fix_source:
        problems = fix(problems)
    if config.validate_only or any(problem.is_error for problem in problems):
        return problems, len(index)

    write_site(index, config)
    return problems, len(index)


def write_site(index: NoteIndex, config: Config):
    """Replace the site's published folders and images with a page for every indexed note."""
    # Images are rebuilt each run like pages, so an image no published note references leaves the site.
    if config.output_image_dir.exists():
        print(f"\nClearing images: {config.output_image_dir}")
        remove_contents_of(config.output_image_dir)

    fm_class = front_matter_class(config.front_matter_config)
    for mapping in config.source_mappings:
        mapping : Mapping
        if not mapping.source.exists():
            # The site mirrors the vault: a folder the vault no longer has publishes nothing.
            if mapping.destination.exists():
                print(f"\nClearing: {mapping.destination} (its source {mapping.source} is gone)")
                remove_contents_of(mapping.destination)
            continue
        if not mapping.destination.exists():
            print(f"Creating output directory: {mapping.destination}")
            mapping.destination.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing: {mapping.source} -> {mapping.destination}")
        remove_contents_of(mapping.destination)

        notes = index.notes_in(mapping)
        for number, note in enumerate(notes, start=1):
            write_page(note, index, fm_class, config)
            print(f"Transferred {note.source.name}. [{number}/{len(notes)}]")


def write_page(note: IndexedNote, index: NoteIndex, fm_class: type[FrontMatter], config: Config):
    """Write the page of one checked note, named and linked from what the index holds about it."""
    dst = note.mapping.destination / note.page_path(fm_class)
    dst.parent.mkdir(parents=True, exist_ok=True)

    fm = read_front_matter(note.source.read_text(encoding='utf-8'), config.front_matter_config)
    #TODO: Move logic to to fm.try_apply()
    if config.front_matter_enabled:
        # Determine title from original filename (before slugification)
        if config.preserve_original_title:
            title = note.source.stem
        else:
            title = dst.stem.replace('-', ' ').title()

        fm.ensure(title)

    dst.write_text(fm.serialize(), encoding='utf-8')

    links = NoteLinks(index=index, own_anchors=note.anchors)
    rewrite_references(dst, config.obsidian_image_dir, config.output_image_dir, config.output_image_dir, links)
