# 3. Processors implement a small contract, and the neutral output is only one of them

Status: accepted (2026-11-10)

## Context

The same vault is published to more than one kind of site: a Jekyll site
today, and an Astro site that expects neutral Markdown. Most of a conversion
is the same for every site: reading notes, working out which note and heading
a link means, validating, and writing. What differs is how things are
expressed: file paths, addresses, heading anchors, front matter fields, and
link and image syntax. A base class with defaults that every site subclasses
was considered. It makes a similar site cheap, but it assumes that every
future site resembles the first one, and a site that differs would have to
fight its inherited behaviour.

## Decision

A processor implements a small contract: the path and address of a page, the
anchor of a heading, the front matter it writes, how a link and an image are
written, and any checks of its own. Its methods only format inputs the core
has already resolved. They never parse, look anything up or touch the disk.
Shared building blocks (the GitHub and kramdown anchor rules, address
patterns, front matter helpers) are available to every processor. The neutral
Markdown output is one implementation of the contract, built from those
blocks. A similar site subclasses it and changes, in order of preference, a
setting, an extension of a generic step, or a whole step. A different site
implements the contract directly. Every processor passes the same contract
test suite.

## Consequences

- Adding a site format means implementing one small contract and making the
  shared contract tests pass. The core is not touched.
- Many differences are settings, such as the address pattern, so some sites
  need no new processor at all, only configuration.
- What a link means is decided once, in the core, for every site.
