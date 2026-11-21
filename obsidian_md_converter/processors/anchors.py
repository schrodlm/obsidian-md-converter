"""Heading anchor rules that more than one processor's site uses."""

import re

# What a GFM heading id drops: anything but a letter, a digit, '_', '-', a space or a tab.
_GFM_NON_WORD_RE = re.compile(r'[^\w\- \t]')


def gfm_anchor_ids(heading_texts: list[str]) -> list[str]:
    """
    The ids GitHub Flavored Markdown (GFM) gives a page's headings.

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
