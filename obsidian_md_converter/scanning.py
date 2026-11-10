"""Reading a note: find the structures the converter acts on, each with its place in the text."""

import re
from dataclasses import dataclass

_FRONT_MATTER_RE = re.compile(r'^---\s*\n(?P<raw>.*?)\n---\s*\n', re.DOTALL)
_FENCE_RE = re.compile(r'^[ \t]*(```|~~~)')
_HEADING_RE = re.compile(r'^#{1,6}[ \t]+(?P<text>.*?)[ \t]*#*[ \t]*$')
_CALLOUT_RE = re.compile(r'^>[ \t]*\[!(?P<kind>[^\]]+)\][+-]?[ \t]*(?P<title>.*)$')
# Inline code may wrap lines but, like a paragraph, never crosses a blank line.
_INLINE_CODE_RE = re.compile(r'(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)*?(?<!`)(?P=ticks)(?!`)', re.DOTALL)
_WIKILINK_RE = re.compile(r'(?P<bang>!?)\[\[(?P<inner>[^\]\[]+)\]\]')
_MARKDOWN_LINK_RE = re.compile(r'(?P<bang>!?)\[(?P<text>[^\]]*)\]\((?P<url>[^)]+)\)')
_SIZE_RE = re.compile(r'^(?P<width>\d+)(?:x(?P<height>\d+))?$')


@dataclass(frozen=True)
class Span:
    """Where a structure sits in the note: character offsets, and the 1-based line and column of its start."""
    start: int
    end: int
    line: int
    column: int


@dataclass(frozen=True)
class Heading:
    span: Span
    level: int
    text: str


@dataclass(frozen=True)
class WikiLink:
    """A link to a note, `[[note#heading|text]]`; `raw` is everything between the brackets."""
    span: Span
    raw: str
    note: str
    heading: str
    display: str | None


@dataclass(frozen=True)
class Embed:
    """An embed, `![[name|alt|300x200]]`."""
    span: Span
    raw: str
    target: str
    alt: str | None
    width: str | None
    height: str | None


@dataclass(frozen=True)
class MarkdownLink:
    span: Span
    text: str
    url: str


@dataclass(frozen=True)
class MarkdownImage:
    span: Span
    alt: str
    url: str


@dataclass(frozen=True)
class Callout:
    span: Span
    kind: str
    title: str


@dataclass(frozen=True)
class CodeRegion:
    """A fenced code block or an inline code span; nothing inside it is a structure."""
    span: Span


@dataclass(frozen=True)
class ScannedNote:
    """A note split into its front matter and body, with the body's structures in the order they appear.

    Spans are offsets into `body`; their lines count from the first line of the note itself.
    """
    front_matter: str | None
    body: str
    body_line: int
    headings: list[Heading]
    links: list[WikiLink]
    embeds: list[Embed]
    markdown_links: list[MarkdownLink]
    markdown_images: list[MarkdownImage]
    callouts: list[Callout]
    code: list[CodeRegion]

    @property
    def replaceable(self) -> list[WikiLink | Embed | MarkdownImage]:
        """The structures a page may rewrite, in text order."""
        return sorted([*self.links, *self.embeds, *self.markdown_images], key=lambda structure: structure.span.start)


def split_front_matter(content: str) -> tuple[str | None, str]:
    """The raw YAML between the opening and closing `---` at the very top of a note, and the body below it."""
    if not content.startswith('---'):
        return None, content
    match = _FRONT_MATTER_RE.match(content)
    if not match:
        return None, content
    return match.group('raw'), content[match.end():]


def scan(content: str) -> ScannedNote:
    """Read a note once and record every structure the converter acts on."""
    front_matter, body = split_front_matter(content)
    body_line = content[:len(content) - len(body)].count('\n') + 1
    return scan_body(body, front_matter=front_matter, body_line=body_line)


def scan_body(body: str, front_matter: str | None = None, body_line: int = 1) -> ScannedNote:
    """Record the structures of a note's body, already separated from its front matter."""
    locate = _Locator(body, body_line)

    fenced, headings, callouts = _scan_lines(body, locate)
    inline_code = [
        CodeRegion(locate(match.start(), match.end()))
        for match in _INLINE_CODE_RE.finditer(body)
        if not _inside(match.start(), fenced)
    ]
    code = sorted([*fenced, *inline_code], key=lambda region: region.span.start)

    links, embeds = [], []
    for match in _WIKILINK_RE.finditer(body):
        if _inside(match.start(), code):
            continue
        span = locate(match.start(), match.end())
        inner = match.group('inner')
        if match.group('bang'):
            embeds.append(_embed(span, inner))
        else:
            links.append(_wikilink(span, inner))

    markdown_links, markdown_images = [], []
    for match in _MARKDOWN_LINK_RE.finditer(body):
        if _inside(match.start(), code):
            continue
        span = locate(match.start(), match.end())
        if match.group('bang'):
            markdown_images.append(MarkdownImage(span, match.group('text'), match.group('url')))
        elif match.group('text'):
            markdown_links.append(MarkdownLink(span, match.group('text'), match.group('url')))

    return ScannedNote(
        front_matter=front_matter,
        body=body,
        body_line=body_line,
        headings=headings,
        links=links,
        embeds=embeds,
        markdown_links=markdown_links,
        markdown_images=markdown_images,
        callouts=callouts,
        code=code,
    )


def _scan_lines(body: str, locate: "_Locator") -> tuple[list[CodeRegion], list[Heading], list[Callout]]:
    """Fenced code blocks, headings and callouts: the structures that occupy whole lines."""
    fenced, headings, callouts = [], [], []
    fence_start = None
    offset = 0
    for line in body.splitlines(keepends=True):
        text = line.rstrip('\r\n')
        if _FENCE_RE.match(text):
            if fence_start is None:
                fence_start = offset
            else:
                fenced.append(CodeRegion(locate(fence_start, offset + len(text))))
                fence_start = None
        elif fence_start is None:
            heading = _HEADING_RE.match(text)
            if heading:
                level = len(text) - len(text.lstrip('#'))
                headings.append(Heading(locate(offset, offset + len(text)), level, heading.group('text')))
            callout = _CALLOUT_RE.match(text)
            if callout:
                callouts.append(Callout(locate(offset, offset + len(text)), callout.group('kind'), callout.group('title').strip()))
        offset += len(line)
    if fence_start is not None:
        # An unclosed fence runs to the end of the note, as in Obsidian.
        fenced.append(CodeRegion(locate(fence_start, len(body))))
    return fenced, headings, callouts


def _wikilink(span: Span, inner: str) -> WikiLink:
    parts = [part.strip() for part in inner.split('|')]
    note, _, subpath = parts[0].partition('#')
    # A nested subpath (#H1#H2) names its innermost heading last.
    heading = subpath.split('#')[-1].strip()
    display = parts[1] if len(parts) > 1 else None
    return WikiLink(span, inner, note.strip(), heading, display)


def _embed(span: Span, inner: str) -> Embed:
    parts = [part.strip() for part in inner.split('|')]
    alt = width = height = None
    for part in parts[1:]:
        size = _SIZE_RE.match(part)
        if size:
            width, height = size.group('width'), size.group('height')
        else:
            alt = part
    return Embed(span, inner, parts[0], alt, width, height)


def _inside(offset: int, regions: list[CodeRegion]) -> bool:
    return any(region.span.start <= offset < region.span.end for region in regions)


class _Locator:
    """Turns character offsets in the body into spans with lines and columns counted in the whole note."""

    def __init__(self, body: str, first_line: int):
        self._line_starts = [0] + [index + 1 for index, char in enumerate(body) if char == '\n']
        self._first_line = first_line

    def __call__(self, start: int, end: int) -> Span:
        line_index = self._line_index(start)
        return Span(start, end, self._first_line + line_index, start - self._line_starts[line_index] + 1)

    def _line_index(self, offset: int) -> int:
        low, high = 0, len(self._line_starts) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self._line_starts[middle] <= offset:
                low = middle
            else:
                high = middle - 1
        return low
