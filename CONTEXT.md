# obsidian-md-converter

Publishes chosen folders of an Obsidian vault to a static site: it turns
Obsidian's own syntax into what the site understands, and checks the notes
before anything is published.

## Language

### Source

**Vault**:
The Obsidian folder the notes are written in. The single source of everything
the site shows.
_Avoid_: repository, notes folder

**Note**:
One Markdown file in the vault.
_Avoid_: document, file, post (for a note in general)

**Mapping**:
A vault folder paired with the site folder its notes are published into.
_Avoid_: route, source mapping

**Folder note**:
The note named like its folder, which is the page for that folder, for example
`MFF/MFF.md` for a university. The notes below it belong to it.
_Avoid_: index note, hub page

### Publishing

**Published note**:
A note in a mapped folder that is neither a draft nor a placeholder.

**Draft**:
A note marked `draft: true`. It is not published, and links to it become plain
text.
_Avoid_: unpublished note, work in progress

**Placeholder**:
A note with nothing below its front matter, usually created by following a link
to a note not written yet. It is not published.
_Avoid_: empty page, stub

**Unwritten note**:
A note that a link names but that does not exist yet. A link to it becomes plain
text.
_Avoid_: broken link, missing note

**Page**:
What a published note becomes on the site: one file in the site folder of its
mapping.
_Avoid_: output file, converted note

**Slug**:
The URL-safe name of a page, made from the note's name in lowercase ASCII words
joined by hyphens.
_Avoid_: filename, permalink

**Clash**:
Two published notes whose pages would have the same slug in the same folder.
It fails the run.
_Avoid_: collision, duplicate

### Links

**Link**:
A reference from one note to another, written `[[note]]` or `[[note|text]]`.
_Avoid_: wikilink (except when contrasting with Markdown links), reference

**Heading link**:
A link to one heading of a note, written `[[note#Heading]]`. On the site it
points at that heading's anchor.
_Avoid_: section link, anchor link

**Anchor**:
The id a site gives a heading, which heading links point at.
_Avoid_: fragment, heading id

**Embed**:
A note or image shown inside another note, written `![[...]]`.
_Avoid_: transclusion, include

### Running

**Run**:
One conversion of the vault into the site. It either completes and replaces the
site's published folders, or changes nothing.
_Avoid_: sync, build

**Validation**:
The checks a run makes before writing anything. Run on their own, they are what
the pre-commit hook does.
_Avoid_: linting, verification

**Problem**:
One thing validation found, with its note, line and severity: an error stops
the run, a warning does not.
_Avoid_: issue, diagnostic

**Export**:
One named destination in the config: a site, the processor its pages are written for, and the folders published to it. A run writes one export.
_Avoid_: target, output (for the destination)

**Processor**:
What expresses a run's pages for one kind of site, such as Jekyll or normalized
Markdown: their paths, addresses, anchors, front matter and syntax.
_Avoid_: target, converter, renderer, exporter

**Normalized Markdown**:
The processor-neutral form of a page that any site can build on, defined by a
versioned specification.
_Avoid_: plain Markdown, neutral markdown, generic output
