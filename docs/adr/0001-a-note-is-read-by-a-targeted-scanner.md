# 1. A note is read by a targeted scanner, not a Markdown parser

Status: accepted (2026-11-10)

## Context

The converter changes only a few things in a note: its front matter, its links,
embeds and images. Everything else, paragraphs, lists, tables, math and the
text of callouts, must reach the site exactly as written. Until v0.2 each step
matched text patterns over the whole note, so it also rewrote link-like text
inside code: `arr[[0]]` in a code block came out as `arr0`. A full Markdown
parser (markdown-it-py and similar) was considered. Such parsers are built to
render HTML. They build a tree of the whole note but do not report the
character positions of links inside a paragraph, and the converter needs those
positions because it edits the note rather than re-rendering it.

## Decision

A note is read by a scanner that recognises only the structures the converter
acts on: front matter, fenced and inline code, headings, wikilinks, embeds,
Markdown links and images, and callout markers. It records each one with its
line and column in the source. Code is recognised, and nothing inside it is
changed. A page is the source note with the recognised structures replaced and
every other character passed through.

## Consequences

- Text outside the recognised structures cannot be damaged by conversion.
- Problems point at the line in the note the author edits, not at a line in
  the converted copy.
- New Obsidian syntax is supported by teaching the scanner a new structure.
  Anything the scanner does not know passes through unchanged.
