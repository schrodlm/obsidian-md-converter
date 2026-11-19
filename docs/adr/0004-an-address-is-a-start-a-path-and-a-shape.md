# 4. A page's address is its site folder's start, its path and its site's shape

Status: accepted (2026-11-18)

## Context

A link points at the address its site serves the linked page at. The site's
generator decides that address, Jekyll from `permalink` in `_config.yml` and
Astro from its route files, so the converter can only predict it. Until now
one template did, `links.url_format: "/{collection}/{slug}.html"`. It spoke
Jekyll's language: its collection is the site folder `_courses` with the `_`
dropped, and `.html` is how Jekyll ends a page. Normalized Markdown (#13)
would have inherited both. Its `{slug}` also carried every subfolder, so the
template could not tell a page's folders from its own name. Serving a folder
note at its folder's address was considered and dropped: it is one vault's
habit, not something every site needs.

Linking to the page's file (`../courses/nswi035.md`) and letting the generator
turn it into a URL was considered too. Jekyll, Hugo, MkDocs and others can do
that, but Astro cannot: its site's own route files decide every URL, so a file
link needs a plugin there that guesses the URL anyway. Predicting is kept: the
rule lives in one place each site's side follows, Jekyll's own default or the
normalized Markdown specification.

## Decision

A page's address has three parts, each decided by what knows it:

- **The start**, one per site folder. The processor derives it from the folder
  the way its generator does by default (`_courses` → `/courses` for Jekyll),
  and the mapping into that folder can override it with `url:`.
- **The path**: the page's subfolders below the vault folder it comes from,
  and its slug (`master/mff`). The core decides it, the same for every site.
- **The shape**, one per site: how its addresses end. Jekyll ends them in
  `.html`, or in `/` with the processor option `permalink: pretty`, named after
  Jekyll's own setting. Normalized Markdown ends them in `/`.

| Vault note | Site folder | Jekyll | Normalized Markdown |
|---|---|---|---|
| `uni/master/MFF.md`, mapped from `uni` | `_uni` | `/uni/master/mff.html` | `/uni/master/mff/` |
| `uni/master/courses/NSWI035 - Principy distribuovaných systémů.md`, mapped from `uni/master/courses` | `_courses` | `/courses/nswi035-principy-distribuovanych-systemu.html` | `/courses/nswi035-principy-distribuovanych-systemu/` |
| the same, with `url: /study` | `_courses` | `/study/nswi035-principy-distribuovanych-systemu.html` | `/study/nswi035-principy-distribuovanych-systemu/` |

`links.url_format` is removed. With no overrides, every Jekyll address stays
what it is today.

## Consequences

- The simplest config states no address at all: each processor's default
  matches its generator's own default.
- A site that changed its generator's URL rules states each change once: a
  start on the site folder, a shape on the processor.
- No note is special. `rentczecher/rentczecher.md` is served like any page,
  at `/projects/rentczecher/rentczecher/` for normalized Markdown.
- A user who customizes their generator's routing still mirrors it in the
  converter's config. A processor may later read its generator's own config
  instead, such as Jekyll's `_config.yml`.
