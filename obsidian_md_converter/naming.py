"""What a note's output file and URL are called: its neutral, URL-safe slug."""

import re
from pathlib import Path

from anyascii import anyascii

from obsidian_md_converter.errors import ConversionError
from obsidian_md_converter.paths import ObsidianPath


def slugify_text(text: str) -> str:
    """
    Transform text into a URL-safe slug: lowercase ASCII letters and digits joined by single hyphens.

    Example:
        "ABA problém (2)" -> "aba-problem-2"
    """
    # Transliterate every script to ASCII ("problém" -> "problem", "Привет" -> "Privet", "Straße" -> "Strasse")
    text = anyascii(text.strip())

    text = re.sub(r'[^\w\s-]', '', text)  # Remove remaining non-word chars
    text = re.sub(r'[\s_-]+', '-', text)  # Convert spaces/underscores to hyphens

    return text.strip('-').lower()


def slugify(filepath: ObsidianPath) -> str:
    """
    Transform a filename into its neutral, URL-safe output name.

    The name is the same for every site generator; a generator's own naming
    convention (such as Jekyll's dated post filenames) is applied when the file is written.

    Args:
        filepath: ObsidianPath object pointing to the file to convert

    Returns:
        A sanitized filename with:
        - Only lowercase letters, numbers, and hyphens
        - No special characters or punctuation
        - Spaces converted to hyphens
        - Multiple hyphens collapsed
        - Leading/trailing hyphens removed

    Example:
        "Hello World!.md" -> "hello-world.md"

    Raises:
        ConversionError: if the name has no letters or digits to build the slug from.
    """
    stem_slug = slugify_text(Path(filepath).stem)
    if not stem_slug:
        raise ConversionError(filepath, "its name has no letters or digits to build a URL from; rename it")
    return stem_slug + Path(filepath).suffix.strip().lower()
