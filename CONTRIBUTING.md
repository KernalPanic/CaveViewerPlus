# Contributing to CaveViewer

Thank you for improving CaveViewer. Contributions should remain usable on
Windows, macOS, and Linux and preserve the ability to work with maps much
larger than system memory.

This guide is the human-facing contribution path. Detailed development policy
and technical references are indexed in
[Development instructions and documentation](docs/development/AGENTS.md).

## Prepare the repository

Follow the [source setup guide](docs/development/source-setup.md) before making
your first change. Start each independent branch from a clean, current `main`:

```bash
git status
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feature/descriptive-name
```

Use an appropriate prefix such as `feature/`, `fix/`, or `docs/`. If the
worktree is not clean, identify and preserve those changes before switching
branches. If `main` cannot fast-forward to `origin/main`, resolve the divergence
deliberately; do not use a force update or an unrelated merge to continue.

## Define the work

Before implementation, copy
[the work-definition template](docs/development/work-definition.md) to
`.work/<work-name>.md`. Complete its A3-style master table with the observed
problem, current implementation, desired solution, ordered tasks, branch, and
status. Keep the rows ordered by implementation sequence.

Root `.work/` is ignored and is the default location for an active plan. Move
or copy a plan to `docs/development/work/` only when it needs to be shared,
reviewed in the pull request, or retained as a durable execution record. Once a
tracked copy exists, it is authoritative and must travel with the change.

Read the canonical documents relevant to the task before editing. In
particular:

- [Architecture](docs/development/architecture.md) defines component boundaries
  and dependency direction.
- [Repository layout](docs/development/repository-layout.md) defines stable
  paths and structural compatibility.
- [Coding standards](docs/development/coding-standards.md) defines implementation
  conventions.
- [Testing](docs/development/testing.md) defines test placement, commands,
  markers, and coverage expectations.
- [UX guidelines](docs/development/ux-guidelines.md),
  [design system](docs/development/design-system.md), and
  [branding](docs/development/branding.md) govern their respective presentation
  concerns.

## Implement the plan

Work through the master-plan rows in order. For each task:

1. Mark the row in progress.
2. Implement the smallest complete change.
3. Add or update tests for observable behavior and failure cleanup.
4. Run focused tests while iterating.
5. Update affected documentation and comments.
6. Record validation evidence and mark the row complete only when its desired
   result is verified.

Keep behavior changes separate from directory moves, mechanical renames, and
formatting-only changes. Do not include unrelated local edits or generated
caches, virtual environments, coverage files, build artifacts, downloaded
maps, or private signing material.

## Validate the branch

Run focused tests first, then the complete suite before handoff when practical:

```bash
.venv/bin/python -m pytest -p no:cacheprovider -q
```

Also run:

```bash
git diff --check
```

Inspect the complete diff for unrelated changes, machine-local paths, secrets,
and generated output. Perform the native platform checks required by the
relevant development document and report any platform you could not test.

## Submit a pull request

`main` is protected by the
[`protect-main` GitHub ruleset](https://github.com/CaveViewer/CaveViewer/rules/19104787).
Do not push directly to it. Once every task assigned to the branch is complete,
commit coherent changes and push the topic branch:

```bash
git push -u origin feature/descriptive-name
```

Open a pull request against `main`. Every pull-request description must contain
these sections, in this order, with relevant information in each:

1. **Summary** - one concise, outcome-oriented description of the change.
2. **Problem** - the concrete defect, risk, or maintenance cost addressed.
3. **Solution** - what changed and the important architectural or behavioral
   details.
4. **Known limitations** - retained constraints, intentionally excluded
   scope, or `None` when there are no known limitations.

Do not add a **Validation** section or validation commands, test results, CI
status, or platform-check results to the pull-request description. Keep that
evidence in the work definition and rely on the pull request's checks for its
current verification status. If the work definition is a tracked review
artifact, include its updates in the pull request; otherwise keep the ignored
local copy current with the PR reference and status.

GitHub prepopulates the required headings from
`.github/pull_request_template.md`.

The pull request must be current with `main`, and its latest commit must pass
all required GitHub checks. A result against an older base is not sufficient.
When a check fails, inspect its complete logs, correct the root cause, add the
lowest reliable regression coverage, rerun local validation, and push the
focused correction. Do not weaken a check merely to make it pass.

## Merge and clean up

After the pull request passes and is approved for merge:

1. Merge it through the protected-branch workflow.
2. Verify that GitHub reports the pull request as merged.
3. Update local `main` to the merged revision.
4. Delete the local topic branch.
5. Delete the remote topic branch if GitHub did not remove it automatically.

```bash
git switch main
git pull --ff-only origin main
git branch -d feature/descriptive-name
git push origin --delete feature/descriptive-name
```

Never delete a topic branch before verifying the merge. Record the merge and
cleanup in the work definition. If the master plan contains tasks assigned to
another branch, start that branch from the newly updated `main` and repeat the
cycle. The work is finished when every row is complete or explicitly deferred.

## Repository instructions and skills

AI coding agents must follow the root and nearest scoped `AGENTS.md` files.
Those files provide mandatory instructions and route agents to the canonical
development documents.

The repository also includes optional reusable agent workflows under
`.agents/skills/`, documented in
[Repository skills](docs/development/skills.md). An agent can use
`$caveviewer-work-cycle` for the contribution lifecycle and combine it with a
focused skill for branding, desktop UX, import lifecycle, performance, or
release work. Skills help agents follow the same process; human contributors do
not need Codex or a skill runner to contribute.

## Specialized contributions

Read the canonical [release guide](docs/development/releases.md) before changing
release workflows, packaging scripts, signing, update manifests, channels, or
version handling. Publication uses the dedicated release workflow and is not an
ordinary feature-branch action.

The project uses the conventional `src/caveviewer` package layout. Do not create
a parallel package tree or change a public cache, update, storage, package, or
application-identity contract without documenting and validating the
compatibility impact in the same change.
