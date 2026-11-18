import datetime
import re
from pathlib import Path

from obsidian_md_converter.processors.base import Processor
from obsidian_md_converter.utils import parse_date

# What a GFM heading id drops: anything but a letter, a digit, '_', '-', a space or a tab.
_GFM_NON_WORD_RE = re.compile(r'[^\w\- \t]')


class JekyllProcessor(Processor):
    """How a run's pages are expressed for a Jekyll site."""

    def __init__(self, url_format: str, default_layout: str | None = None):
        # Such as '/{collection}/{slug}.html', filled in with the collection's name and the page's slug.
        self.url_format = url_format
        # The layout a page gets when its front matter names none, such as 'note'; None adds no layout.
        self.default_layout = default_layout

    def page_path(self, neutral_path: Path, front_matter: dict) -> Path:
        """
        The neutral path, except that a post gets the dated name Jekyll's '_posts' collection requires.

        Example: 'turing-machine.md' with 'layout: post' and 'date: 2024-12-20' -> '2024-12-20-turing-machine.md'
        """
        if front_matter.get("layout") != "post":
            return neutral_path

        date_value = front_matter.get("date")
        if not date_value:
            raise ValueError("Missing 'date' field for post layout")

        # YAML reads an ISO date as a date object; other formats arrive as text.
        if isinstance(date_value, datetime.date):
            parsed_date = date_value
        else:
            parsed_date = parse_date(str(date_value))
        if parsed_date is None:
            raise ValueError("Invalid date format in front matter for post layout")

        date_prefix = f"{parsed_date.year:04d}-{parsed_date.month:02d}-{parsed_date.day:02d}"
        return neutral_path.with_name(f"{date_prefix}-{neutral_path.name}")

    def address(self, neutral_path: Path, site_folder: str) -> str:
        """
        The URL format filled in: a collection's site folder starts with '_', which its URLs leave out.

        Example: 'tmux.md' in '_notes' -> '/notes/tmux.html'
        """
        collection = site_folder.lstrip('_')
        return self.url_format.format(collection=collection, slug=neutral_path.with_suffix('').as_posix())

    def anchor_ids(self, heading_texts: list[str]) -> list[str]:
        """
        The ids kramdown gives a page's headings: Jekyll converts Markdown with kramdown, which reads it as
        GitHub Flavored Markdown (GFM) by default.

        An id is the heading's text lowercased, with punctuation dropped and each space or tab turned into '-'.
        A repeated id gets '-1', '-2' and so on, counted across the page, which is why it takes every heading at once.

        Example: ['Prefix key', '2-3 trees & B+', 'Prefix key'] -> ['prefix-key', '2-3-trees--b', 'prefix-key-1']
        """
        counts: dict[str, int] = {}
        ids = []
        for text in heading_texts:
            base = _GFM_NON_WORD_RE.sub('', text.lower()).replace(' ', '-').replace('\t', '-')
            count = counts.get(base, -1) + 1
            counts[base] = count
            ids.append(f"{base}-{count}" if count else base)
        return ids

    def front_matter_defaults(self) -> dict[str, str]:
        """
        The default layout, which names the template Jekyll renders a page in: 'note' means '_layouts/note.html'.

        Example: default_layout 'note' -> {'layout': 'note'}; no default_layout -> {}
        """
        return {'layout': self.default_layout} if self.default_layout else {}
