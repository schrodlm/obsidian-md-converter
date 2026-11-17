---
name: architecture-diff
description: Draw a one-screen before/after diagram of how a change moves the converter's architecture, generated from the code. Use before asking the user to review a commit or open a PR, or when they ask to see the architecture or what a change does to it.
---

# Architecture diff

The user reviews direction from a picture, not from the diff. Draw what the change does to the
converter's structure, from the code as it is, so nothing has to be kept up to date by hand.
The diagram is shown, never committed. Decisions that should last go into `docs/adr/`.

## Range

- Default: `master` against the working tree (`git diff master`), so uncommitted work shows.
- A commit: `<sha>~1` against `<sha>`. A range or branch the user names: that range.

## Gather

1. `git diff --stat <base>` and `git log --oneline <base>..HEAD` for what changed.
2. Read each changed module on both sides (`git show <base>:<path>` for before). Follow the run
   from `cli.py` through its entry point, so the flow is the real call order, not the file list.
3. Read `CONTEXT.md` and the ADRs the change touches. Use their words for every box and label.

## Draw

One HTML page that fits on one screen. Match `example.html` next to this file: its sections,
its classes and its theme variables.

1. **The change in words**, first, so the diagrams have context: two or three plain sentences.
   The problem a user ran into (with a concrete example), then what the change does about it.
   Example: "One missing image stopped the whole run, so a single broken embed kept every other
   page from updating. Now a missing image is a warning and the page shows its alt text."
2. **Before**: the run as stages left to right. Only stages the change touches, plus one on
   each side for context.
3. **After**: the same, with new stages outlined (`.new`) and a point that decides whether the
   run goes on outlined as a gate (`.gate`).
4. **On each stage, the one fact that changed**: how often notes are read, whether it touches
   the disk, what happens to a bad note. Mark worse in red (`.bad`), better in green (`.good`).
5. **Shapes**: the data types or class hierarchy the change adds or reshapes, as short field
   lists. Leave out types that did not change.
6. **Where this sits**: the ADR steps as pills. Done, this change, next.

Rules:

- Every function, class and module named on the page exists in the code at the range's end.
  Check each name with grep before rendering.
- Show what a user of the converter would notice (a page now missing, a run now stopping),
  next to the structure that causes it.
- Apart from the opening sentences, no paragraphs. A label or one short line per box. A change
  that moves nothing structural gets the sentences alone, not a diagram.

## Show

Write the page to the scratchpad and render it in the session (in T3 Code: `html_preview`,
then `html_render`). In the reply, don't describe the page; say only what it can't show, such
as what was tested and what to look at first.
