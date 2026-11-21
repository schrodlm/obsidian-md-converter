# Normalized Markdown

Version 1.1

Changes since 1.0: links can point at URLs instead of files (`links: urls`).

Normalized Markdown is the generator-neutral form of a page that the converter
writes with `processor: {type: markdown}`. It is written for any Markdown
site and follows no generator's conventions. A site builds on this document
alone, without knowing how the converter works. Only a new major version
may break a site that follows an older one, and each converter release states
which version it writes.

A site's Markdown renderer needs CommonMark, GitHub's tables, task lists and
footnotes, math, and callouts, all described below.

## What the site configures

The specification fixes the format. Each site's config chooses:

- which vault folders are published into which site folder (`source_mappings`);
- where images are copied (`images.destination_dir`);
- whether links point at files, the default, or at URLs (`links: urls`).

By default nothing in the format depends on the site's URLs: links point at
files, and the site decides what URL each file gets.

## Files

- One Markdown file per page, in its mapping's site folder.
- A page keeps its subfolders below the vault folder it comes from.
- Each folder and file name is a slug: lowercase ASCII words joined by
  hyphens, with diacritics removed. A dated note, such as a blog post, is named
  the same way: its date stays in its front matter, not in its name.
- A note marked `draft: true`, or with nothing below its front matter, doesn't
  become a page.

```
vault: uni/master/courses/NSWI035 - Principy distribuovaných systémů.md
site:  courses/nswi035-principy-distribuovanych-systemu.md
```

## Front matter

- Every page has front matter, and it is valid YAML.
- `title` is always present: the note's own `title`, or else the note's name
  (`Teorém CAP` for `teorem-cap.md`).
- Every other field of the note is passed through unchanged, including any
  wikilink in it, such as `related: "[[Other note]]"`.
- The converter adds no generator-specific field, such as `layout`.

```yaml
---
tags: [mff]
title: "Teorém CAP"
---
```

## Body

- CommonMark, with GitHub's tables, task lists and footnotes.
- Math as `$…$` and `$$…$$`.
- Fenced code keeps its language.
- The converter adds no raw HTML; HTML written in a note is passed through.

### Callouts

A callout is a blockquote whose first line names its type, as Obsidian writes
it: any type, an optional fold marker (`+` open, `-` closed) and an optional
title. The site decides how to show each type, including ones it doesn't know.

```markdown
> [!warning]- Watch out
> Collapsed by default.
```

## Links

A link to another published page is an ordinary Markdown link to that page's
file, relative to the linking page, with the heading's anchor when it names one.
Every link points at a file that exists in the output, so the files can be
read and checked as they are; the site turns each file link into its URL.

From `courses/nswi035-principy-distribuovanych-systemu.md`:

```markdown
[tmux > Prefix key](../notes/tmux.md#prefix-key)
[NSWI028](nswi028-embedded-systems.md)
[Prefix key](#prefix-key)
```

Links are relative so they keep working wherever the converted folder is
placed. A site usually keeps its content in a subfolder, such as
`src/content/`:

```
my-site/
└── src/content/
    ├── notes/tmux.md
    └── courses/nswi035.md     ← links to tmux
```

- `../notes/tmux.md` means "up one folder, then `notes/tmux.md`", so it finds
  the file wherever `src/content/` sits.
- `/notes/tmux.md` would mean "`notes/tmux.md` at the very top", where there is
  no such file: GitHub and the usual link plugins would follow it to nothing.

### Links to URLs

With `processor: {type: markdown, links: urls}`, a link points at the page's
URL instead, for a site that serves each page at `/<site folder>/<page>/` and
shouldn't have to turn file links into URLs itself. The URL starts with the
site folder's exact name, or the mapping's `url:`, then the page's path, and
ends in `/`:

```markdown
[tmux > Prefix key](/notes/tmux/#prefix-key)
[NSWI028](/courses/nswi028-embedded-systems/)
```

### Every link

- A link with nowhere to go, to a draft, an empty note or a note that doesn't
  exist, becomes its plain text.
- A link to a heading the page doesn't have points at the page.
- A link's text is the one the note gave, or else the note's name, and
  `Note > Heading` for a heading.

### Heading anchors

A heading's anchor follows GitHub's rule: lowercase, every character
removed except letters, digits, `-`, `_` and spaces, and each space turned
into `-`. Letters keep their accents. A repeated anchor gets `-1`, `-2`.
A site gives its headings ids by the same rule, so the anchors in links find
them.

| Heading | Anchor |
|---|---|
| `Prefix key` | `prefix-key` |
| `Totální uspořádání (total ordering)` | `totální-uspořádání-total-ordering` |
| `2-3 trees & B+` | `2-3-trees--b` |
| `Prefix key`, again | `prefix-key-1` |

## Images

- An image is an ordinary Markdown image, copied to the image folder and
  linked by its path there from the output's root, with no size. The site
  serves the image folder at that same path, as a static folder.
- An image without alt text gets `Image`.
- An image missing from the vault becomes its alt text, or nothing without one.
- An image on the web is left as written.

```markdown
![Flow](/assets/img/a.png)          ← ![[a.png|Flow|200]]
```

## Obsidian-only syntax

Syntax with no form in this format has a defined outcome:

| In the note | On the page |
|---|---|
| `%%comment%%` | removed |
| `==highlight==` | bold: `**highlight**` |
| `![[Other note]]`, a note embed | a link to that note |
| `[[note#^block]]`, a block reference | a link to the note |
| a Dataview block | removed, with a warning when converting |
