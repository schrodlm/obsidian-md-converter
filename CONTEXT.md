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
A site folder and the vault folders whose notes are published into it. A note
belongs to the most specific mapped vault folder that holds it, so mapping
`uni` and `uni/master/courses` publishes each course once.
_Avoid_: route, source mapping

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

**Address**:
The URL a site serves a page at, which links point at. It is made of its site
folder's start, the page's path below its vault folder, and its site's shape:
`/courses/` + `nswi035-principy-distribuovanych-systemu` + `.html`.
_Avoid_: permalink, URL format

**Title**:
The name a page shows, kept in its front matter: the note's name, such as
`ABA problém`, unless the note's front matter already gives one. Every page has
one, because its slug loses the note's name.
_Avoid_: page name, heading

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

**Check**:
One kind of problem, and how validation finds it. Its name, such as
`missing-heading`, is printed with every problem it finds.
_Avoid_: rule, lint

**Problem**:
One thing a check found, with its note, line and severity. An error leaves no
sensible page to write and stops the run. A warning still leaves one, and never
stops it.
_Avoid_: issue, diagnostic

**Export**:
One named destination in the config: a site, the processor its pages are written for, and the folders published to it. A run writes one export.
_Avoid_: target, output (for the destination)

**Processor**:
How one kind of site, such as Jekyll or normalized Markdown, expects a run's
pages to be written. The core resolves every note, heading and image, and asks
the processor only where sites differ: a page's path and address, its heading
anchors, the front matter fields the site needs (such as a Jekyll layout), and
the Markdown text of structures such as images. It never reads the vault.
_Avoid_: target, converter, renderer, exporter

**Normalized Markdown**:
The processor-neutral form of a page that any site can build on, defined by a
versioned specification.
_Avoid_: plain Markdown, neutral markdown, generic output
