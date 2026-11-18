---
name: architecture-diff
description: Summarise a change to the converter as What and Why, and draw it in the UML view that fits it best, generated from the code. Use before making a planned change, before asking the user to review a commit or open a PR, or when they ask to see the architecture or what a change does to it.
---

# Architecture diff

The user reviews direction from a summary and a picture, not from the diff. Say what a change
does and why, then draw it in the one view that shows it best, from the code as it is, so
nothing has to be kept up to date by hand. The diagrams are shown, never committed. Decisions
that should last go into `docs/adr/`.

## Range

- Default: `master` against the working tree (`git diff master`), so uncommitted work shows.
- A commit: `<sha>~1` against `<sha>`. A range or branch the user names: that range.
- A planned change, not yet written: `master` against the design as discussed. The after
  diagram's title says its new names are proposed.

## Gather

1. `git diff --stat <base>` and `git log --oneline <base>..HEAD` for what changed.
2. Read each changed module on both sides (`git show <base>:<path>` for before). Follow the run
   from `cli.py` through its entry point, so dependencies and calls are the real ones.
3. Read `CONTEXT.md` and the ADRs the change touches. Use their words for packages and notes.

## Summarise

Two short paragraphs, in `CONTEXT.md`'s words, plain enough to read without the code:

- **What**: what changes in the code, in one or two sentences.
- **Why**: the problem it solves, with one concrete example (a note, a link, a page), in one
  or two sentences.

## Pick the view

Choose the view that shows this change best: usually one, at most two. Each view answers one
question; match it to what the change actually moves:

| View | Answers | Fits a change that |
|---|---|---|
| Component diagram | Where is the boundary, and what crosses it? | adds a module, or moves who calls whom |
| Worked example table | What does it do to real input? | changes output, or adds a module whose job shows best on input → output |
| Class diagrams, before and after | How does the code's shape change? | moves code between modules, or changes types, signatures or dependencies |
| Sequence diagram | When does it happen in a run? | changes the order of calls, or adds a pass or a gate |

A change that moves nothing structural gets What and Why alone. Name the chosen view and why
in one line.

Draw the diagrams in PlantUML, in the scratchpad. Copy [`example/`](example/) there and edit
it: it is the processors PR drawn in all four views, and its `style.iuml` holds the shared look.

- **Component diagram**: the module, its callers, the interface between them, and a note on
  what crosses it (in, out, and what it never does).
- **Worked example table**: Markdown, one row per operation: what it is handed, then what each
  implementation returns. Take outputs from the code, its docstrings or its tests; an output
  only an issue proposes cites that issue.
- **Class diagrams**: the modules and classes the change touches, plus one neighbour on each
  side. A module is a class marked `<<module>>`, with its functions as members.
- **Sequence diagram**: `autonumber`, with a point that decides whether the run goes on drawn
  as an `alt`.

The two class diagrams share a **skeleton**, so they compare box by box:

- The same packages, named by the stage they serve (`core: run`, `core: index`,
  `core: validate`, `core: write`), declared in the same order. A package the change adds
  comes last.
- The same classes in the same packages, except those the change adds, removes or moves.
- Pinned placement: every arrow between packages is `-[dashed,norank]->`, and hidden edges at
  the end fix where each package sits, as in the example.

Colour carries the change, with the legend in each `title`:

- Before: red (`#FFE0E0` box, `<color:#C00000>` member) marks what the change takes away or moves.
- After: blue (`#E0ECFF` box, `<color:#0050C0>` member) marks what it adds or changes.
- A member that moves gets a note in the after diagram naming where it came from, such as
  `page_path <- JekyllFrontMatter.apply_slug_transform`.

Every name in the before diagram exists at the range's base, and every name in the after
diagram at its end, with its real signature. Check each with grep before rendering.

## Render

`norank` needs a recent PlantUML; the system `plantuml` is from 2020. Use the pinned jar:

```sh
jar=~/.cache/plantuml/plantuml-1.2024.7.jar
[ -f "$jar" ] || { mkdir -p "${jar%/*}" && curl -sL -o "$jar" \
  https://github.com/plantuml/plantuml/releases/download/v1.2024.7/plantuml-1.2024.7.jar; }
java -jar "$jar" -tpng *.puml
```

PlantUML reads a line starting with `'` as a comment, even inside a note, and drops it silently:
start such a line with something else.

Read every PNG before showing it. Done when no label covers another, and in class diagrams
the packages sit in the same places before and after.

## Show

What, Why, then the chosen view, with diagrams as Markdown images by absolute path. After it,
only what the view can't show, such as what was tested, the open decisions, and what to look
at first.
