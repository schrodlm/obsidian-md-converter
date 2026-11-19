# CLAUDE.md

## Talking to the owner

Applies to every agent and subagent on this project, including reports written for another agent to pass on.

- **As concise as possible.** Lead with the answer or the next step. No surveys, no history unless asked.
- **The owner only knows what was said in the conversation.** They don't see files you read, commands you ran or research you did. Before using anything they haven't seen, give the minimal context in a line or two: what it is, and why it matters here, with one concrete example.
- **Plain words over project shorthand.** Name a file, setting or issue by what it does, not only by its name or number.

## Commits

- **Atomic.** One logical piece of work per commit, as small as coherent. Two fixes never share a commit.
- **Tests are their own commit.** The implementation lands first; its tests follow immediately in a separate commit, never mixed into it.
- **No AI trailers.** Commit messages carry no Co-Authored-By or session trailers — the subject line is the whole message.
- **Fixups fold in.** A correction to a commit that has not been pushed belongs squashed into it — history records the corrected work, not the correction process.
- **Moves and renames are their own commits.** Relocating or renaming a class, function, or file is one mechanical commit whose diff reads as pure movement; the behavior change that motivated it comes separately.
- **One-line messages.** The subject states the commit's single goal in imperative mood. If the message needs a body to enumerate what changed, the commit is too big — split it.
- **Two reviewers gate every commit.** Before a commit is made, two independent review passes run over the diff: one hunting correctness bugs, one hunting simplifications. Every finding is fixed or explicitly dismissed first.
