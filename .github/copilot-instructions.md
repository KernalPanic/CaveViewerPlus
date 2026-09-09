# CaveViewer Copilot instructions

Follow the repository-level `AGENTS.md` and any nearer scoped `AGENTS.md` file.
Detailed standards live under `docs/development/`.

- Preserve unrelated changes and keep behavioral edits separate from file
  moves.
- Respect the `core` to `gui` dependency direction and main-thread ownership of
  Tk/OpenGL resources.
- Preserve atomic cache publication, cancellation cleanup, and public update
  paths.
- Add focused tests for behavior and failure cleanup. Run the relevant tests,
  the complete suite when practical, and `git diff --check` before handoff.
- When opening a pull request, follow the description structure in
  `CONTRIBUTING.md` and keep validation information out of the description.
- Do not add generated artifacts, downloaded maps, credentials, or private
  signing material.
