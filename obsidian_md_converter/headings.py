import re

from obsidian_md_converter import scanning

_ATX_HEADING_RE = re.compile(r'^#{1,6}[ \t]+(?P<text>.*?)[ \t]*#*[ \t]*$')
_FENCE_RE = re.compile(r'^[ \t]*(```|~~~)')
_EXPLICIT_ID_RE = re.compile(r'[ \t]*\{#(?P<id>[^}\s]+)\}$')

_WIKILINK_RE = re.compile(r'!?\[\[(?P<target>[^\]\[|]+)(?:\|(?P<display>[^\]\[]+))?\]\]')
_MARKDOWN_LINK_RE = re.compile(r'!?\[(?P<text>[^\]]*)\]\([^)]*\)')

# Characters Obsidian replaces with a space when matching a link to a heading.
_OBSIDIAN_HEADING_PUNCTUATION_RE = re.compile(r'[!"#$%&()*+,.:;<=>?@^`{|}~\\/\[\]\r\n]')
_GFM_NON_WORD_RE = re.compile(r'[^\w\- \t]')


def strip_heading(text: str) -> str:
    """Normalize heading text the way Obsidian does before comparing it with a link's heading."""
    stripped = _OBSIDIAN_HEADING_PUNCTUATION_RE.sub(' ', text)
    return re.sub(r'\s+', ' ', stripped).strip()


def heading_key(text: str) -> str:
    """The key under which a heading and a link to it meet: Obsidian matches headings case-insensitively."""
    return strip_heading(text).lower()


def rendered_heading_text(source: str) -> str:
    """The heading's visible text once its links are rendered: a link shows only its display text."""
    def wikilink_text(match: re.Match) -> str:
        display = match.group('display')
        return (display if display is not None else match.group('target')).strip()

    text = _WIKILINK_RE.sub(wikilink_text, source)
    return _MARKDOWN_LINK_RE.sub(lambda m: m.group('text'), text)


class HeadingAnchors:
    """The anchors a Jekyll page gives its headings, looked up by the heading text an Obsidian link names.

    Jekyll's default kramdown reads Markdown as GFM, whose header ids are the lowercased rendered
    text with non-word characters removed and each space turned into '-', numbered on repeats.
    """

    def __init__(self, body: str):
        self._anchors: dict[str, str] = {}
        id_counts: dict[str, int] = {}

        for source in (heading.text for heading in scanning.scan_body(body).headings):
            explicit = _EXPLICIT_ID_RE.search(source)
            if explicit:
                source = source[:explicit.start()]
                anchor = explicit.group('id')
            else:
                anchor = self._gfm_id(rendered_heading_text(source), id_counts)
            self._anchors.setdefault(heading_key(source), anchor)

    @staticmethod
    def _heading_sources(body: str) -> list[str]:
        sources = []
        in_fence = False
        for line in body.splitlines():
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = _ATX_HEADING_RE.match(line)
            if match:
                sources.append(match.group('text'))
        return sources

    @staticmethod
    def _gfm_id(text: str, id_counts: dict[str, int]) -> str:
        base = _GFM_NON_WORD_RE.sub('', text.lower()).replace(' ', '-').replace('\t', '-')
        count = id_counts.get(base, -1) + 1
        id_counts[base] = count
        return f"{base}-{count}" if count else base

    def anchor_for(self, heading: str) -> str | None:
        """The page anchor for the heading a link names, or None when the note has no such heading."""
        return self._anchors.get(heading_key(heading))
