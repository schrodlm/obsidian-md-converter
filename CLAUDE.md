# CLAUDE.md

## Commits

- **Atomic.** One logical piece of work per commit, as small as coherent. Two fixes never share a commit.
- **Tests are their own commit.** The implementation lands first; its tests follow immediately in a separate commit, never mixed into it.
- **No AI trailers.** Commit messages carry no Co-Authored-By or session trailers — the subject line is the whole message.
- **Fixups fold in.** A correction to a commit that has not been pushed belongs squashed into it — history records the corrected work, not the correction process.
- **Moves and renames are their own commits.** Relocating or renaming a class, function, or file is one mechanical commit whose diff reads as pure movement; the behavior change that motivated it comes separately.
- **One-line messages.** The subject states the commit's single goal in imperative mood. If the message needs a body to enumerate what changed, the commit is too big — split it.
- **Two reviewers gate every commit.** Before a commit is made, two independent review passes run over the diff: one hunting correctness bugs, one hunting simplifications. Every finding is fixed or explicitly dismissed first.
