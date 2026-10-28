# How Obsidian resolves links

Research for [#4](https://github.com/schrodlm/obsidian-md-converter/issues/4), feeding the
"Link semantics" decision ([#7](https://github.com/schrodlm/obsidian-md-converter/issues/7)).

Question: what exactly does Obsidian do with internal links and embeds, so the converter can
reproduce it on a Jekyll (minima, kramdown) site?

## Sources

Primary sources, fetched 2026-10-28:

- **[OH-links]** Obsidian Help, *Internal links*: <https://help.obsidian.md/links>
  (source: [obsidian-help `en/Linking notes and files/Internal links.md`](https://github.com/obsidianmd/obsidian-help/blob/master/en/Linking%20notes%20and%20files/Internal%20links.md))
- **[OH-embeds]** Obsidian Help, *Embed files*: <https://help.obsidian.md/embeds>
- **[OH-aliases]** Obsidian Help, *Aliases*: <https://help.obsidian.md/aliases>
- **[OH-settings]** Obsidian Help, *Settings → Files and links*: <https://help.obsidian.md/settings>
- **[OH-attach]** Obsidian Help, *Attachments*: <https://help.obsidian.md/attachments>
- **[OH-formats]** Obsidian Help, *Accepted file formats*: <https://help.obsidian.md/file-formats>
- **[OH-ofm]** Obsidian Help, *Obsidian Flavored Markdown*: <https://help.obsidian.md/obsidian-flavored-markdown>
- **[API]** Obsidian plugin API typings, [`obsidian.d.ts`](https://github.com/obsidianmd/obsidian-api/blob/master/obsidian.d.ts)
  (`MetadataCache.getFirstLinkpathDest`, `fileToLinktext`, `parseLinktext`, `stripHeading`, `stripHeadingForLink`).
- **[JEKYLL]** Jekyll default configuration, [`lib/jekyll/configuration.rb`](https://github.com/jekyll/jekyll/blob/master/lib/jekyll/configuration.rb);
  Jekyll docs: <https://jekyllrb.com/docs/configuration/markdown/>
- **[GFM]** kramdown-parser-gfm, [`lib/kramdown/parser/gfm.rb`](https://github.com/kramdown/parser-gfm/blob/master/lib/kramdown/parser/gfm.rb)
  (`generate_gfm_header_id`); docs <https://github.com/kramdown/parser-gfm>
- **[KD]** kramdown HTML converter, `auto_ids` option: <https://kramdown.gettalong.org/converter/html.html#auto-ids>,
  and [`lib/kramdown/converter/base.rb`](https://github.com/gettalong/kramdown/blob/master/lib/kramdown/converter/base.rb) (`generate_id`).
- **[SITE]** Target site [`schrodlm/schrodlm.github.io`](https://github.com/schrodlm/schrodlm.github.io): `_config.yml`, `Gemfile.lock`, `.github/workflows/pages.yml`.
- **[VAULT]** The real vault `~/dev/second-brain` (read only): `.obsidian/app.json` and a scan of all `.md` files
  (fenced code and inline code excluded).

Claims marked **(unverified)** are widely observed Obsidian behaviour that is *not* stated in the
help docs or API typings.

## Summary for the "Link semantics" decision

1. **A link target is a vault path, matched by suffix, `.md` optional.** `[[X]]`, `[[X.md]]`,
   `[X](X.md)` and `[X](X%20Y)` are the same link [OH-links]. Folder paths start at the vault root
   and use `/` [OH-links]. Non-Markdown targets must carry their extension (`[[Figure 1.png]]`)
   [OH-links]. Resolution is "best match" for the linkpath, preferring a match relative to the
   source note, otherwise any file whose path ends in the linkpath [API `getFirstLinkpathDest`].
2. **"New link format" only affects what Obsidian *writes*, not how it *reads*.** The three options
   (shortest / relative / absolute) [OH-settings] all resolve through the same lookup. The vault
   has no `newLinkFormat` key, so it uses the default *shortest path when possible*: bare filename
   when unique, full path otherwise [API `fileToLinktext`]. `alwaysUpdateLinks: true` means file
   renames rewrite links automatically [OH-links]; it does **not** fix heading links after a heading
   is renamed (8 stale heading links in the vault, see below).
3. **Case-insensitive file matching (unverified)**: Obsidian matches link paths case-insensitively.
   13 vault links differ only in case from their file (e.g. `[[raft|Raft]]` → `notes/RAFT.md`).
   The converter should look up by lowercased path.
4. **Unresolved links are normal.** 477 of 1005 wikilinks in the vault point at notes that do not
   exist (Obsidian creates them on click [OH-links]). The converter needs an explicit policy for
   them (plain text is the obvious choice).
5. **Subpaths**: `#Heading`, `#H1#H2` (nested), `#^block-id`; `[[#Heading]]` targets the current
   note [OH-links]. `|text` sets display text [OH-links]; `|` is split before `#`
   [API `parseLinktext`].
6. **Heading anchors must be matched, not computed by one formula.** Obsidian's subpath is the
   heading text passed through `stripHeadingForLink`/`stripHeading` (punctuation like
   `# | ^ : [ ] \` replaced by spaces, whitespace collapsed) [API]. Jekyll 4 + default kramdown uses
   **the GFM parser's id algorithm**, not kramdown's own [JEKYLL][GFM]: lowercase, delete every
   character that is not a Unicode word char, `-`, space or tab, turn each space/tab into `-`
   (no collapsing), dedupe with `-1`, `-2` [GFM]. Diacritics survive (`totální-uspořádání-total-ordering`).
   So the converter should: normalise the link's heading and each target heading with
   `stripHeading`, compare case-insensitively, then emit the kramdown id of the *rendered* heading
   (after the converter rewrote any links inside it), counting duplicates in document order.
7. **Block refs `^id` have no Jekyll equivalent**: kramdown emits no id for them. Supporting them
   means the converter injects an anchor (e.g. an IAL `{: #id}` or `<a id>`). The vault has zero
   block refs today.
8. **Aliases** (`aliases:` list in front matter) are alternative names that Obsidian *inserts* as
   `[[Real name|alias]]` [OH-aliases]. Bare `[[alias]]` is possible but Obsidian avoids writing it.
   The vault has no `aliases:` at all. Low priority; if supported, add aliases as extra keys in the
   name index.
9. **Embeds `![[...]]`**: same resolution, content shown inline [OH-embeds]. Images take `|W` or
   `|WxH` sizing [OH-embeds]; notes, headings and blocks transclude; PDFs take
   `#page=N&height=H` [OH-embeds]. In the vault, every embed is an image (75 png) or a PDF (1); no
   note embeds. Three have `|`: two widths (`|500`, `|300`) and one *caption-like* non-numeric
   text (`| Suffixové pole ...`), which Obsidian treats as alt text, not size (unverified).
10. **Markdown links/images to vault files** resolve the same way as wikilinks, destination must be
    URL-encoded [OH-links]. The vault has 10 Markdown images pointing at bare filenames
    (`![Figure 1 ...](subset_performance_subset10.png)`) that live in `assets/images/`: suffix
    resolution, not relative-file resolution, is what makes these work.

## 1. Resolving the link target

### Syntax and equivalence

[OH-links]:

> Wikilink: `[[Three laws of motion]]` or `[[Three laws of motion.md]]`
> Markdown: `[Three laws of motion](Three%20laws%20of%20motion)` or `[Three laws of motion](Three%20laws%20of%20motion.md)`
> The examples above are equivalent ... and links to the same note.

> To link to a note in a folder, include the folder path before the note name. Folder paths start
> at the vault root and use forward slashes (`/`), even on Windows.

> links to file formats other than Markdown needs to include a file extension, such as `[[Figure 1.png]]`.

> A string which contains the following characters may not work as a link: `# | ^ : %% [[ ]]`.

Consequences for parsing a wikilink body: split on the first `|` (display text), then on the first
`#` (subpath) [API `parseLinktext('foo#bar')` → `{path: 'foo', subpath: '#bar'}`]. Strip `.md` if
present. A `/` in the path is always a folder separator, so a note titled `I/O streams` cannot
exist; `[[I/O streams]]` and `[[TCP/IP]]` in the vault are links to non-existent notes
`O streams` in folder `I` and `IP` in folder `TCP`. Whitespace around `|` occurs
(`[[logical contracts#Postconditions | postcondition]]`); trim both parts.

### Lookup ("best match")

The help docs do not spell out the algorithm. The API typing is the most authoritative statement
[API]:

```ts
/** Get the best match for a linkpath.
 *  @param sourcePath The path of the file containing the link, used to resolve relative links. */
metadataCache.getFirstLinkpathDest('foo/bar', 'baz/qux.md')
// TFile with path: 'baz/foo/bar.md' or 'some/other/path/foo/bar.md'
```

So the lookup is: the path relative to the source note's folder if it exists, otherwise any file
whose vault path ends with `/<linkpath>` (or equals it). For a bare name that means "any file with
that basename". When several files qualify, which one wins is not documented; the writer side
avoids the ambiguity by emitting the full path whenever the name is not unique [API
`fileToLinktext`: "If file name is unique, use the filename. If not unique, use full path."].

Recommended converter rule: (1) `<source dir>/<path>`, (2) exact vault-root path, (3) unique
suffix match on lowercased paths; if (3) is ambiguous, report it as a warning and pick
deterministically (e.g. shortest path). The vault today has **no duplicate basenames** (262 notes;
also none differing only by case), so ambiguity is a hardening concern, not a data concern.

### Case sensitivity (unverified)

Neither the help nor the API typings state it. Observed Obsidian behaviour is case-insensitive
matching, and the vault relies on it: 13 links differ from the filename only by case, e.g.
`[[raft|Raft]]` → `notes/RAFT.md`, `[[Turing machine|...]]`. Jekyll output paths are
case-sensitive on GitHub Pages, so the converter must emit the *file's* casing, not the link's.

### "New link format" and the vault settings

[OH-settings]:

> **New link format** — What links to insert when auto-generating internal links.
> Shortest path when possible / Relative path to file / Absolute path in vault.

This is a *writer* setting; any of the three forms resolves via the same lookup above. The vault's
`.obsidian/app.json` has no `newLinkFormat` and no `useMarkdownLinks`, so it uses the defaults
(shortest path, wikilinks). Other settings [VAULT]:

| key | value | meaning |
| --- | --- | --- |
| `alwaysUpdateLinks` | `true` | "Automatically update internal links" when a file is renamed [OH-settings] |
| `attachmentFolderPath` | `assets/images` | new attachments go to that folder [OH-attach]; all 120 png in the vault live there |
| `newFileLocation` / `newFileFolderPath` | `folder` / `notes` | new notes go to `notes/` unless the link has a folder path [OH-settings] |

Because attachments are referenced by bare filename (`![[split.png]]`) while living in
`assets/images/`, the converter must resolve attachments by suffix match too, not relative to the
note.

### Unresolved links

[OH-links]: "If the link points to a note that doesn't exist yet, Obsidian creates the note".
The API keeps them separately (`MetadataCache.unresolvedLinks`) [API]. In the vault, **477 of 1005
wikilinks** (~47%) have no target file (e.g. `[[proc]]`, `[[page]]`, `[[process]]` in
`notes/pagemap.md`). This is the single largest category the converter must handle, and also the
case of a target that exists but is not published.

## 2. Subpaths: headings and blocks

### Syntax [OH-links]

| form | meaning |
| --- | --- |
| `[[note#Heading]]` | heading in another note |
| `[[#Heading]]` | heading in the same note |
| `[[note#H1#H2]]` | subheading: "You can add multiple hash symbols for each subheading" |
| `[[note#^block-id]]` | block; ids "can only consist of Latin letters, numbers, and dashes" |
| `[[note#Heading\|text]]` | display text, default is the raw link text, e.g. `Example#Details` → shown as "Example > Details" in Obsidian (unverified rendering) |

Block ids are defined by ` ^id` at the end of a paragraph or list item, or on its own line after
a structured block (list, quote, callout, table) [OH-links]. "Block references are specific to
Obsidian and not part of the standard Markdown format" [OH-links].

### How Obsidian matches a heading

Obsidian's subpath is not a slug. When it writes a heading link it runs the heading through
`stripHeadingForLink` ("stripping out some bad combinations of special characters that could break
links": `'foo:#|^\\\r\n%%[[]]bar'` → `'foo bar'`), and when matching it compares with
`stripHeading` ("Normalizes headings for link matching by stripping out special characters and
shrinking consecutive spaces": `'foo!"#$%&()*+,.:;<=>?@^`{|}~\/\[\]\\\r\nbar'` → `'foo bar'`)
[API]. Real vault examples:

| heading source | link subpath |
| --- | --- |
| `## [[koordinace a synchronizace v distribuovaném systému\|Synchronizace a koordinace]]` | `#koordinace a synchronizace v distribuovaném systému Synchronizace a koordinace` |
| `#### Linearizability (Linearizovatelnost / Atomická):` | `#Linearizability (Linearizovatelnost / Atomická)` |
| `### [[FLP impossibility]]` | `#FLP Impossibility` (case differs too) |

Note the first: the subpath is built from the heading's **raw Markdown source** including the
wikilink target, not from the visible text.

Vault scan of the 78 heading links (non-embed) [VAULT]: 64 match a heading exactly, 4 only after
`stripHeading`-style normalisation (case-insensitive), 8 match no heading at all (stale after a
heading rename, e.g. `[[#Modely konzistence]]`, or `Líné vyvažování (lazy rebalancing)` vs heading
`(lazy re-balancing)`), and 2 point to a missing note. No nested `#H1#H2` links are used.
In Obsidian these open the note without scrolling (unverified); the converter should do the same
(link to the page without fragment) and warn.

### How Jekyll generates heading ids

The site builds with `bundle exec jekyll build` in GitHub Actions, using Jekyll 4.4.1, kramdown
2.5.2, kramdown-parser-gfm 1.1.0, theme minima 2.5.2, and no `kramdown:` or `markdown:` key in
`_config.yml` [SITE]. It is **not** the `github-pages` gem, so Jekyll's own defaults apply
[JEKYLL]:

```ruby
"markdown" => "kramdown",
"kramdown" => { "auto_ids" => true, "input" => "GFM", "smart_quotes" => "lsquo,rsquo,ldquo,rdquo", ... }
```

With `input: GFM` and `auto_ids: true` (and `transliterated_header_ids` off), the GFM parser
assigns ids itself and kramdown's converter algorithm [KD] is **not used** [GFM]:

```ruby
NON_WORD_RE = /[^\p{Word}\- \t]/.freeze
def generate_gfm_header_id(text)
  result = text.downcase
  result.gsub!(NON_WORD_RE, '')
  result.tr!(" \t", '-')
  @id_counter[result] += 1          # Hash.new(-1)
  counter_result = @id_counter[result]
  result << "-#{counter_result}" if counter_result > 0
  @options[:auto_id_prefix] + result
end
```

`text` is the heading's *rendered text content*: text, code spans, math, entities and smart quotes
as characters; link markup is dropped, only link text kept [GFM `update_raw_text`]. A header with
an explicit `{#id}` keeps it. Python equivalent (Ruby `\p{Word}` ≈ Python `\w` with Unicode,
which also includes `_`):

```python
def gfm_id(text, counter):
    s = re.sub(r"[^\w\- \t]", "", text.lower())
    s = s.replace(" ", "-").replace("\t", "-")
    n = counter[s] = counter.get(s, -1) + 1
    return f"{s}-{n}" if n else s
```

Confirmed by running kramdown 2.5.2 + parser-gfm locally with the Jekyll options:

| heading | id |
| --- | --- |
| `Prefix key` (twice) | `prefix-key`, `prefix-key-1` |
| `Linearizability (Linearizovatelnost / Atomická):` | `linearizability-linearizovatelnost--atomická` |
| `[Stavovost](/notes/stavovost.html)` | `stavovost` |
| `Totální uspořádání (total ordering)` | `totální-uspořádání-total-ordering` |
| ``2-3 trees & `B+` trees`` | `2-3-trees--b-trees` |
| `Custom {#my-id}` | `my-id` |

Contrast with plain kramdown's own algorithm [KD `basic_generate_id`]: strips leading non-letters,
keeps only ASCII `a-zA-Z0-9 -` (so `Totální` → `totln`), and falls back to `section`. Which one
applies is purely a function of `kramdown.input`; if the site ever sets `input: kramdown`, every
anchor changes.

### Matching the two

The two schemes are not inverses, so the converter needs, per target note:

1. Parse its headings in order (outside code fences).
2. For each heading compute `key = stripHeading(raw source).lower()` (Obsidian side) and
   `id = gfm_id(rendered text after the converter's own link rewriting)` (Jekyll side), with the
   duplicate counter.
3. Resolve `#sub` by `stripHeading(sub).lower()` → first matching `id`. For `#H1#H2`, use the last
   component, restricted to headings under H1.
4. No match → link to the page without a fragment, warn.

Doing (2) on rendered text matters: `## [[stavovost distribuovaného systému|Stavovost]]` becomes
`## [Stavovost](...)` → id `stavovost`, while its Obsidian key is
`stavovost distribuovaného systému stavovost`.

Alternative: emit explicit ids (`## Heading {#obsidian-key-slug}`) so the anchor is under the
converter's control. That avoids depending on the GFM algorithm but changes ids readers might
already have bookmarked.

### Blocks

kramdown has no notion of `^id`; the literal `^id` would appear in the HTML text. The converter
must at minimum strip trailing ` ^id` markers, and to support `#^id` links, attach an anchor:
kramdown block IAL `{: #id}` after the paragraph/list [KD syntax: <https://kramdown.gettalong.org/syntax.html#block-ials>].
Vault usage: **0 block refs, 0 `^id` links**.

## 3. Aliases

[OH-aliases]: "add `aliases` property in the note Properties. Aliases should always be formatted as
a list in YAML." When linking via an alias, "Obsidian creates the link with the alias as its
custom display text, for example `[[Artificial Intelligence|AI]]` ... Rather than just using the
alias as the link destination (`[[AI]]`)". So in a vault edited through the UI, aliases appear
only as display text, already handled by the `|` rule. A hand-written `[[AI]]` would still resolve
in Obsidian (unverified; the help doesn't state it). The vault has **no `aliases:` front matter**.

## 4. Embeds

[OH-embeds]: "add an exclamation mark (`!`) in front of an Internal link. You can embed files in
any of the Accepted file formats." Same target resolution as links.

| form | Obsidian behaviour [OH-embeds] | Jekyll mapping |
| --- | --- | --- |
| `![[note]]` | transclude the whole note | inline the converted body, or degrade to a link |
| `![[note#Heading]]` | transclude that section | inline the section, or link with anchor |
| `![[note#^id]]` | transclude the block | inline the block, or link |
| `![[img.png]]` | image | `![](/assets/...)` |
| `![[img.png\|200]]` | width 200, proportional | `<img width="200">` or `![](...){: width="200"}` (kramdown IAL) |
| `![[img.png\|200x100]]` | width x height | `width`/`height` attributes |
| `![[file.pdf]]` | PDF viewer; `#page=N`, `#height=H`, combine with `&` | `<iframe>`/`<embed>`/`<object>` or a plain link |
| `![[audio.ogg]]`, video | player | `<audio>`/`<video>` |
| `![250](https://...)` | external image, Markdown syntax, alt text used as size | strip size from alt |

Differences between embeds and images: an image is only one kind of embed; `![[...]]` decides by
file extension ([OH-formats] lists images `.avif .bmp .gif .jpeg .jpg .png .svg .webp`, audio,
video, `.pdf`, `.canvas`, `.base`, `.md`). A Markdown image `![alt](path)` pointing to a vault file
is resolved like a link [OH-links], and Obsidian reuses the size syntax in the alt text
[OH-embeds]. A non-numeric `|text` on an image embed is not a size and acts as alt text
(unverified).

Vault usage [VAULT]: 76 wikilink embeds = 75 `.png` + 1 `.pdf`
(`![[modelling_system_example.pdf]]`); 0 note/heading/block embeds. With `|`: `|500`, `|300`,
and `| Suffixové pole slova "annbansbananas" a pole $R$ a $L$` (caption). Plus 10 Markdown images
pointing at bare filenames in `assets/images/`. Attachment files: 119 png, 1 pdf.

## 5. Markdown links to notes

[OH-links]: Markdown-format links are equivalent to wikilinks, "make sure to URL encode the link
destination. For example, blank spaces become `%20`." The `.md` extension is optional and subpaths
use the same `#Heading` text (`[Section name](Example.md#Details)`). So the converter should
URL-decode the destination, then feed it through the same resolver as a wikilink. Anything with a
scheme (`https:`, `mailto:`) is external. Vault usage: 145 external Markdown links, 1 malformed
"internal" one that is really a schemeless URL (`[tiny_renderer](github.com/ssloy/tinyrenderer)`
in `publish/projects/Goob the Renderer.md`), and the 10 images above.

## Vault counts (for prioritising)

Scan of 262 notes in `~/dev/second-brain` (code excluded):

| category | count |
| --- | ---: |
| wikilinks | 1005 |
| with `\|` display text | 419 |
| target note missing | 477 |
| case differs from file | 13 |
| heading links | 78 (6 same-note `[[#...]]`) |
| block links / block ids | 0 |
| links with `/` folder path | 3 (2 are titles containing `/`) |
| wikilink embeds | 76 (75 png, 1 pdf; 0 notes) |
| embeds with `\|` | 3 (2 sizes, 1 caption) |
| Markdown images to vault files | 10 |
| notes with `aliases:` | 0 |
| duplicate basenames | 0 |
