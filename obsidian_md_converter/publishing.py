"""Which notes are published."""

from obsidian_md_converter.front_matter import FrontMatter


def is_published(content: str) -> bool:
    """A note is published unless it is a draft (`draft: true`) or a placeholder with nothing below its front matter."""
    fm = FrontMatter(content)
    return fm.get('draft') is not True and bool(fm.body.strip())
