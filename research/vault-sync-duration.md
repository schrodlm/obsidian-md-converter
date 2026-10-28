# Why did the vault sync take 24 minutes?

Research for [#5](https://github.com/schrodlm/obsidian-md-converter/issues/5), part of the map [#2](https://github.com/schrodlm/obsidian-md-converter/issues/2).

## Answer

**The converter was not the cause. A GitHub incident was.** On 2026-10-19 the
`sync` job in both slow runs finished in 18 s and 40 s, about the same as in
the fast runs. The extra ~21 minutes is the time between the job finishing and
GitHub marking the run as completed. The Pages deploy jobs in the site repo
also sat queued for 14 to 20 minutes before a runner picked them up. Both
delays happened during a GitHub incident that degraded Actions, Pages and
webhooks between 15:24 and 16:52 UTC that day
([githubstatus incident](https://stspg.io/nrv2k8s68bwn)). The incident report
states that "Workflow status updates, logs, and checks were delayed" and that
"Pages deployments ... were delayed".

**Nothing on the map needs to change.** The converter step (`Run conversion`)
took less than half a second in every run.

## Evidence

Sources: the GitHub REST API (`actions/runs/<id>`, `actions/runs/<id>/jobs`,
`actions/runs/<id>/logs`, `deployments`, `environments`), the workflow files
`.github/workflows/sync-to-jekyll.yml` (schrodlm/second-brain) and
`.github/workflows/pages.yml` (schrodlm/schrodlm.github.io), and
`https://www.githubstatus.com/api/v2/incidents.json`. All times are UTC.

### Sync to Jekyll Site (schrodlm/second-brain)

| Run | Date | Created → job start (queue) | Job duration (steps) | Job completed | Run marked completed | Run total |
|---|---|---|---|---|---|---|
| 36625512309 (fast) | 09-29 | 7 s | 16 s (20:19:23–20:19:39) | 20:19:42 | 20:19:43 | 27 s |
| 37141398184 (fast) | 10-03 | 3 s | 14 s (17:40:53–17:41:07) | 17:41:09 | 17:41:09 | 19 s |
| 39281307393 (slow) | 10-19 | 4 s | 16 s (15:53:23–15:53:39) | 15:53:41 | **16:14:59** | 21 m 40 s |
| 39281491079 (slow) | 10-19 | 3 s | 39 s (15:54:53–15:55:32) | **15:57:41** | **16:19:27** | 24 m 37 s |

Per-step timings from the jobs API and the log timestamps:

| Step | 37141398184 (fast) | 39281307393 (slow) | 39281491079 (slow) |
|---|---|---|---|
| Checkouts (3) | 7 s | 5 s | 8 s |
| Set up Python | 0 s | 1 s | 0 s |
| Install converter (`pip install`) | 3.4 s | 5.9 s | **25.7 s** |
| Run conversion | **0.4 s** | **0.4 s** | **0.5 s** |
| Commit and push | <1 s (nothing to commit) | 1.3 s (pushed `c6499ec..d326869`) | 1.4 s (pushed `d326869..09f9915`) |

What this shows:

- **Queueing: none to speak of.** Each runner was picked up 3–7 s after the
  run was created (`system.txt` in the log archive: "Waiting for a runner to
  pick up this job..." followed by the job start a few seconds later).
- **Converter: unchanged.** `Run conversion` took 0.4–0.5 s in all runs. The
  log shows all four mappings (`uni/courses`, `notes`, `publish/posts`,
  `publish/projects`) processed within that half second.
- **pip: slower in one run, but by seconds.** In 39281491079, `pip install`
  took 26 s instead of 3–6 s (8 s before `Processing ./obsidian-md-converter`
  and 15 s collecting PyYAML). That accounts for about 20 s, not minutes, and
  fits the network degradation described in the incident.
- **Push: normal.** It took 1.3–1.4 s, and both pushes reached the site repo
  at 15:53:38 and 15:55:30.
- **Where the minutes went:** in 39281491079 the last log line
  ("Cleaning up orphan processes") is at 15:55:32, but the job's
  `completed_at` is 15:57:41, a 2-minute lag before GitHub recorded the job
  as finished. In both slow runs the run's `updated_at`/completion came about
  21 minutes after the job had finished. That is GitHub's status processing
  lagging, not work on the runner. The incident report names this effect:
  "Workflow status updates, logs, and checks were delayed".

### Build and deploy site (schrodlm/schrodlm.github.io)

`pages.yml` has `concurrency: group: pages, cancel-in-progress: false`, and a
`deploy` job that `needs: build` and targets the `github-pages` environment.
That environment has only a branch-policy rule, with no wait timer and no
required reviewers.

| Run | build job queued → started | build duration | deploy job queued → started | deploy duration | Run total |
|---|---|---|---|---|---|
| 36625559794 (fast, 09-29) | 2 s | 69 s | 4 s | 11 s | 88 s |
| 39281349442 (slow, 10-19) | 2 s | 54 s | **15:54:37 → 16:14:39 (20 m)** | 11 s | 35 m |
| 39281578904 (slow, 10-19) | **15:55:32 → 16:14:51 (19 m, waiting on `pages` concurrency)** then 2 s | 41 s | **16:15:35 → 16:29:25 (14 m)** | 8 s | 44 m |

- The site build (Jekyll, Playwright, CV PDF) took 41–69 s in every run.
- The first slow run's deploy job waited 20 minutes for a hosted runner.
- The second run could not start until the first left the `pages` concurrency
  group (16:14:50), so it inherited that wait. Its own deploy job then waited
  another 14 minutes.
- Even after `deploy-pages` finished, the `github-pages` deployment records
  were last updated at 16:35:56 and 16:39:26. This matches "Pages deployments
  ... were delayed" in the incident report.

### GitHub status, 2026-10-19

Incident "Degraded availability for GitHub Actions, GitHub Pages and
Webhooks" (critical; opened 15:49, resolved 16:56). The resolution says that
between 15:24 and 16:52 UTC Actions, Pages and Actions webhooks were degraded.
An internal Actions service that stores logs and run status lost its cache
connections, and the retry surge overloaded a shared network link. Reported
impact: delayed workflow status updates, logs and checks; 5.13% of hosted
runner jobs delayed in starting; delayed Pages deployments. Every slow
interval above (15:53–16:40) falls inside that window.

## Implications for the map

- No converter work follows from this. Conversion time is flat at about 0.5 s
  for the current vault, and the slow runs used the same pinned converter
  (`ref: v0.1.0`) as the fast ones.
- Note that the sync workflow installs the converter from the `v0.1.0` tag.
  Changes on this map will not reach the vault sync until that ref is bumped.
  This is not related to the slowdown.
- Optional, outside the converter: if a stuck run should not hold the site
  hostage, the site's `pages` concurrency group with
  `cancel-in-progress: false` serialises deploys. During an incident a second
  push waits for the first. This is a site-repo choice, not a converter
  issue, and it did not cause the delay.
- No runs have happened since 2026-10-19, so no later fast run confirms
  recovery yet. The next vault push should take about 20 s.
