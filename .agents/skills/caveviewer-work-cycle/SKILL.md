---
name: caveviewer-work-cycle
description: "Run CaveViewer implementation work from a clean, current main branch through work definition, task execution, verification, pull request, merge confirmation, and branch cleanup. Use when starting, continuing, or finishing planned repository work; not for release publication or read-only analysis."
---

# CaveViewer work cycle

Move planned repository work through a verifiable branch and pull-request
lifecycle while preserving unrelated changes and explicit authorization gates.

## Establish the current state

1. Read `CONTRIBUTING.md`, `docs/development/work-definition.md`, and
   `docs/development/testing.md`, plus every applicable `AGENTS.md` and
   task-specific canonical document.
2. Determine whether the request starts new work, continues an existing branch
   and work definition, prepares a pull request, responds to review or CI, or
   finishes a verified merge.
3. Inspect the repository root, current branch, worktree, remotes, and active
   work definition before changing Git state. Preserve unrelated user changes.

## Start new work

- Require a clean worktree before switching branches or updating `main`. If
  changes are present, identify their ownership and stop rather than stashing,
  discarding, or carrying them into new work without direction.
- Fetch `origin`, switch to `main`, and update it only by fast-forward to
  `origin/main`. Stop on divergence; do not introduce a merge commit, rebase,
  reset, or force update merely to make `main` look current.
- Create a descriptive `feature/`, `fix/`, `docs/`, or other appropriate branch
  from that verified revision.
- Before implementation, create `.work/<work-name>.md` from
  `docs/development/work-definition.md`. Record why an existing non-main base
  is required if the task intentionally continues unmerged dependent work.

Do not perform these mutations when the user requested only analysis, a plan,
or advice about a possible change.

## Execute the master plan

- Keep the A3 master table ordered by implementation sequence. Assign every row
  to its intended branch and keep problem, current implementation, desired
  solution, task details, status, and verification evidence current.
- Set one task in progress, implement the smallest complete change, run focused
  tests, and update its documentation and work row before moving on.
- Add regression coverage for observable behavior and failure cleanup. Preserve
  compatibility boundaries and report platform validation that cannot be run.
- Keep commits coherent and reviewable. Do not mix unrelated behavior, file
  moves, mechanical formatting, or another contributor's changes.
- Before branch handoff, complete every row assigned to that branch, run the
  proportionate complete suite, run `git diff --check`, inspect the final diff,
  and record the evidence in the work definition.

## Submit and iterate

- Pushing, creating or updating a pull request, merging, and deleting branches
  are external or destructive actions. Perform each only when the user's
  current request explicitly authorizes it; skill activation alone is not
  authorization.
- When the user gives the imperative instruction **“Finalize this branch
  through `origin/main` and clean up local and remote topic branches”** for the
  current working branch, treat it as explicit authorization for this complete,
  bounded lifecycle: push the working branch to `origin` without force; open a
  pull request against `main`; wait for all required checks to pass and then
  merge the pull request; update the active master plan with the pull-request
  ID; and, after verifying the merge, delete the local and remote working topic
  branches. Quoting, documenting, or asking about this instruction does not
  execute or authorize the lifecycle.
- Push the intended branch without force. Create one pull request against
  `main` with these required content sections in order: **Summary**, **Problem**,
  **Solution**, and **Known limitations**. Give each section relevant content;
  use `None` when no limitations are known. Do not include a **Validation**
  section, validation commands, test results, CI status, or platform-check
  results in the pull-request description. Keep verification evidence in the
  work definition and rely on GitHub checks for current status. Record the pull
  request reference in the work definition.
- When asked to monitor or repair the pull request, inspect complete check logs,
  distinguish code defects from runner or service failures, implement the root
  fix, add the lowest reliable regression coverage, update the work record, and
  push the verified correction.
- Do not weaken tests, bypass protected-branch checks, broaden permissions, or
  rewrite shared history to obtain a green pull request.

## Merge, clean up, and continue

1. Before merging, verify the pull request is current with `main`, all required
   checks pass, and the user has authorized the merge.
2. Verify the pull request actually merged; do not infer success from a command
   starting or a branch disappearing.
3. Switch to `main`, fetch `origin`, and fast-forward to the merged revision.
4. Delete the local topic branch only after merge verification. Delete the
   remote branch only when it still exists and that cleanup is authorized.
5. Record the merge reference, final validation, cleanup, and any deferred work
   in the work definition.
6. If the master plan assigns remaining rows to another branch, begin again
   from the newly updated `main`. Finish only when every row is complete or its
   deferral is explicit.

Stop and report the exact condition instead of improvising when the worktree is
dirty, `main` diverges, required checks fail without a reproducible cause, the
PR is not merged, or a requested mutation lacks authorization. Release
publication follows `$caveviewer-release`, not this skill.
