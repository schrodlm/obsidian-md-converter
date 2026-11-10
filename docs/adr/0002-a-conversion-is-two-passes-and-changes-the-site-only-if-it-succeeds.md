# 2. A conversion runs in two passes and changes the site only if it succeeds

Status: accepted (2026-11-10)

## Context

A link in one note needs the address and headings of another, and clash
detection needs every output name, so some knowledge of all notes must exist
before any page is written. Until v0.2 the converter emptied the site's
folders first and then wrote note by note: a failure halfway left the site
half-written, and checks such as links to missing headings ran only during
writing, so `--validate` never saw them. Holding every note and every finished
page in memory was considered. It fits easily, at a few MB for a vault of a
few hundred notes, but memory would grow with the vault, and it still needs a
safe way to replace the site.

## Decision

A run reads each note twice, one note at a time. The first pass scans every
note into a small index of published notes: their names, front matter,
headings and references, with no note text. Every check then runs over the
index, and any error stops the run before anything is written. `--validate`
stops there in every case. The second pass builds each page from its source
note and writes it once into a staging folder, copying each used image once.
Only when the whole run has succeeded does the staging folder replace the
site's converted folders. On any failure it is discarded and the site is
unchanged.

## Consequences

- The site always holds a complete conversion of one vault state, never a
  partial one.
- `--validate`, and so the pre-commit hook, reports every problem a publish
  would report.
- After validation, note content can no longer fail a run.
- Memory holds the index and one note, however large the vault grows.
- Every note is read and scanned twice. That is the cost of not holding notes
  in memory.
