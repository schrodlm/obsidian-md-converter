# obsidian-md-converter

Publish chosen folders of your Obsidian vault to a static site, with any site
generator you like.

You write notes in Obsidian, with `[[wikilinks]]`, `![[image embeds]]` and
headings you link to. A static site generator such as Astro, Hugo or Jekyll
doesn't understand any of that. This tool turns your notes into Markdown the
site understands, and checks them before anything is published, so a broken
link or a bad front matter never reaches your site.

```markdown
In your vault, notes/2PC.md:              On your site, notes/2pc.md:

See [[tmux#Prefix key|the prefix]].       See [the prefix](tmux.md#prefix-key).
![[flow.png|200]]                         ![Image](/assets/img/flow.png)
```

You keep writing in Obsidian. You choose and style the site yourself.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Choosing what to publish](#choosing-what-to-publish)
- [Choosing the site format](#choosing-the-site-format)
- [What happens to your notes](#what-happens-to-your-notes)
- [Checks](#checks)
- [Running it automatically](#running-it-automatically)
- [Reference](#reference)
- [Development](#development)

## Install

You need Python 3.10 or newer.

```sh
pip install git+https://github.com/schrodlm/obsidian-md-converter
```

This installs the `obsidian-md-converter` command. To pin a release, add it to
the address, for example `...obsidian-md-converter@v0.3.0`.

## Quick start

1. **Write a config file** next to your vault, for example
   `.obsidian-md-converter.yaml`:

   ```yaml
   paths:
     obsidian_root: "."              # the vault; relative paths start at this file
     output_root: "../my-site/src/content"

   source_mappings:
     - source: "notes"              # a folder in your vault
       destination: "notes"         # where its pages go, below output_root

   images:
     source_dir: "assets/images"    # inside the vault
     destination_dir: "assets/img"  # inside output_root

   processor:
     type: markdown                 # plain Markdown for any site
   ```

2. **Check your notes** without writing anything:

   ```sh
   obsidian-md-converter --config .obsidian-md-converter.yaml --validate
   ```

3. **Publish:**

   ```sh
   obsidian-md-converter --config .obsidian-md-converter.yaml --force
   ```

   `--force` creates the site folders the first time. Without it, a missing
   site folder is an error, which protects you from a typo in a path.

Every problem is printed with its note and line, like a compiler:

```
notes/2PC.md:3: warning: Missing image: flow.png; writing its alt text instead [missing-image]
Transfer completed: 0 error(s), 1 warning(s), 212 files processed
```

**Your site only ever changes when the whole run succeeds.** Pages are written
to a staging folder first and swapped in at the end. If anything fails, the
site stays as it was. Your vault is never changed, unless you ask for `--fix`.

## Choosing what to publish

Only the folders you map are published. Each mapping sends one or more vault
folders into one site folder:

```yaml
source_mappings:
  - source: "notes"
    destination: "notes"
  - sources: ["uni/master/courses", "uni/bachelor/courses"]
    destination: "courses"          # both folders' notes end up here
  - source: "uni"
    destination: "uni"
```

- **Subfolders are kept.** `notes/databases/2PC.md` becomes
  `notes/databases/2pc.md`.
- **The most specific folder wins.** Above, `uni` and `uni/master/courses`
  are both mapped. A course note is published once, into `courses`, and the
  walk of `uni` skips it.
- **One mapping per site folder.** List several vault folders under one
  `sources:` instead of writing two mappings with the same `destination`.
- **Some notes are never published:** notes with `draft: true` in their front
  matter, and empty notes (nothing below the front matter, as when you
  follow a link to a note you haven't written yet). Links to them become plain
  text.

## Choosing the site format

The `processor` section says which kind of site the pages are written for.

### `markdown`: plain Markdown for any site

```yaml
processor:
  type: markdown
```

Generic Markdown that follows no generator's conventions. Links point at the
linked page's file, relative to the linking page, so they work wherever you put
the converted folder, and even on GitHub. Your site turns file links into its
URLs. The full format is written down in
[docs/normalized-markdown.md](docs/normalized-markdown.md), so you can build a
site on it without reading this tool's code.

What your site needs:
- a Markdown renderer with GitHub tables, footnotes, math (`$…$`) and
  Obsidian-style callouts (`> [!note]`);
- a way to turn links to `.md` files into URLs. MkDocs and VitePress do this
  already; Hugo, Jekyll and Eleventy need a setting or a plugin;
- the image folder served at the same path, as a static folder;
- heading ids in GitHub's style, which most renderers use by default.

**If your site serves each page at `/<folder>/<page>/`,** as Astro, Hugo and
Eleventy sites usually do, let the converter write URLs instead, and your site
needs no link plugin:

```yaml
processor:
  type: markdown
  links: urls        # [tmux](/notes/tmux/) instead of [tmux](../notes/tmux.md)
```

A mapping's `url:` changes where its URLs start, as for Jekyll below.

### `jekyll`: a Jekyll site

```yaml
processor:
  type: jekyll
  default_layout: note      # optional: the layout a page gets when it names none
  permalink: pretty         # optional: set it when your _config.yml does
```

Pages are written the way Jekyll expects:
- links point at final URLs, such as `/notes/2pc.html` (or `/notes/2pc/` with
  `permalink: pretty`);
- a post (`layout: post` and a `date:`) is named with its date, as in
  `2024-12-20-turing-machine.md`;
- image sizes are written in kramdown's syntax, `{:width="200"}`.

A site folder named `_courses` is linked as `/courses/…`, as Jekyll does by
default. If your site serves it elsewhere, add `url:` to its mapping:

```yaml
  - source: "uni/courses"
    destination: "_courses"
    url: "/study"           # links become /study/<page>.html
```

## What happens to your notes

| In your note | On the page |
|---|---|
| `[[Other note]]` | a link to its page |
| `[[Other note\|my text]]` | the same, showing `my text` |
| `[[Other note#Heading]]` | a link to that heading, shown as `Other note > Heading` |
| `[[#Heading]]` | a link to a heading in the same page |
| a link to a draft, an empty note, or a note that doesn't exist | its plain text |
| `![[flow.png\|Data flow\|200x100]]` | an image, with its alt text and, for Jekyll, its size |
| `![Figure](flow.png)` | the same image, copied from the vault |
| an image missing from the vault | its alt text, with a warning |
| `%%a comment%%` | removed |
| `==a highlight==` | `**a highlight**`, bold |
| `![[Other note]]`, a note shown inline | a link to that note |
| `[[note#^abc123]]`, a link to one paragraph | a link to the note |
| a ` ```dataview ` query | removed, with a warning |
| anything inside code | left exactly as written |

**File names** become lowercase ASCII words joined by hyphens:
`NSWI035 - Principy distribuovaných systémů.md` becomes
`nswi035-principy-distribuovanych-systemu.md`.

**Front matter** is kept as you wrote it. Every page also gets a `title`, your
note's name with its capitals and accents, unless you gave it one, because the
file name loses them. Tabs in front matter, which YAML doesn't allow, are
read as spaces.

## Checks

Every run checks all notes before writing anything. An **error** stops the
run, and your site isn't touched. A **warning** still publishes, with a
sensible fallback.

| Name | Kind | Means |
|---|---|---|
| `missing-folder` | warning | a mapped vault folder doesn't exist |
| `missing-output-folder` | error | a site folder doesn't exist; run with `--force` to create it |
| `unreadable-note` | error | a note isn't valid UTF-8 text |
| `unusable-name` | error | a note's name has no letters or digits to build a file name from |
| `clash` | error | two notes would be written to the same file |
| `tabs-in-front-matter` | warning | front matter indented with tabs; `--fix` rewrites them as spaces in your vault |
| `invalid-front-matter` | error | front matter that isn't valid YAML |
| `unnamable-page` | error | the site can't name the page, such as a Jekyll post without a `date:` |
| `missing-image` | warning | an image the note shows isn't in the vault; the page shows its alt text |
| `missing-heading` | warning | a link names a heading its note doesn't have; it points at the page |
| `dataview-query` | warning | a Dataview query was left out, since only Obsidian can run it |

`--validate` runs every check and writes nothing. That makes it a good
pre-commit hook.

## Running it automatically

**As a pre-commit hook** in your vault, so a broken note can't even be
committed. In the vault's `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/schrodlm/obsidian-md-converter
    rev: v0.3.0
    hooks:
      - id: obsidian-validate
        args: ["--config", ".obsidian-md-converter.yaml"]
```

**In CI**, install the tool and run it with your config whenever the vault
changes, then commit the converted pages to your site's repository.

## Reference

### Command line

```
obsidian-md-converter [--config FILE] [--validate] [--force] [--fix]
                      [--obsidian-root DIR] [--output-root DIR]
                      [--obsidian-image-dir DIR] [--output-image-dir DIR]
```

| Option | Does |
|---|---|
| `--config FILE` | the config file; without it, `config.yaml` next to the package |
| `--validate` | check every note and write nothing |
| `--force` | create missing site folders |
| `--fix` | fix what can be fixed in the vault itself (tabs in front matter) |
| `--obsidian-root`, `--output-root` | override the config's roots; relative to where you run the command |
| `--obsidian-image-dir`, `--output-image-dir` | override the image folders |

The command exits with `1` when there is an error, so scripts and CI notice.

### Config file

| Key | Means |
|---|---|
| `paths.obsidian_root` | the vault; relative paths start at the config file |
| `paths.output_root` | where the site's pages go |
| `source_mappings` | the mappings: `source` or `sources`, `destination`, and an optional `url` where their URLs start |
| `images.source_dir` | the vault's image folder, inside `obsidian_root` |
| `images.destination_dir` | where images are copied, inside `output_root` |
| `processor.type` | `markdown` or `jekyll` |
| `processor.links` | Markdown only: `files` (default) or `urls` |
| `processor.default_layout` | Jekyll only: the layout a page gets when it names none |
| `processor.permalink` | Jekyll only: `pretty` for addresses ending in `/` |
| `runtime.validate_only`, `runtime.force_create`, `runtime.fix_source` | defaults for `--validate`, `--force` and `--fix` |

[config.yaml](config.yaml) is a commented example.

## Development

```sh
pip install -e ".[test]"
pytest
```

The design is written down in the repository:
- [CONTEXT.md](CONTEXT.md): the project's vocabulary, such as note, mapping,
  page and processor.
- [docs/adr/](docs/adr/): the decisions behind the design, one file each.
- [docs/normalized-markdown.md](docs/normalized-markdown.md): the format the
  `markdown` processor writes.
