"""Contract tests for shared agent instructions and work definitions."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_instruction_hierarchy_is_present_and_scoped() -> None:
    scoped_instruction_paths = (
        "docs/development/AGENTS.md",
        "src/AGENTS.md",
        "src/caveviewer/core/AGENTS.md",
        "src/caveviewer/gui/AGENTS.md",
        "tests/AGENTS.md",
    )

    assert (REPOSITORY_ROOT / "AGENTS.md").is_file()
    for relative_path in scoped_instruction_paths:
        assert (REPOSITORY_ROOT / relative_path).is_file()
        assert "Inherits:" in _read(relative_path)


def test_root_instructions_delegate_development_policy_and_require_startup_checks() -> None:
    instructions = _read("AGENTS.md")

    assert "docs/development/AGENTS.md" in instructions
    assert "## Session startup" in instructions
    assert "Inspect the active branch and Git status" in instructions
    assert "focused and complete validation" in instructions
    assert "## Architecture and compatibility" not in instructions
    assert "## Common commands" not in instructions


def test_development_instructions_index_every_canonical_document() -> None:
    development_root = REPOSITORY_ROOT / "docs" / "development"
    instructions = _read("docs/development/AGENTS.md")
    canonical_documents = sorted(
        path.name
        for path in development_root.glob("*.md")
        if path.name != "AGENTS.md"
    )

    assert canonical_documents
    for filename in canonical_documents:
        assert f"]({filename})" in instructions


def test_jetbrains_rule_delegates_to_canonical_instructions() -> None:
    rule = _read(".aiassistant/rules/repository-instructions.md")

    assert "Always follow the root `AGENTS.md`" in rule
    assert "root `.work/` by default" in rule
    assert "docs/development/work/" in rule
    assert "apply: always" in rule
    assert "**Always** project rule" in rule

    gitignore = _read(".gitignore")
    assert "!.aiassistant/rules/" in gitignore
    assert "!.aiassistant/rules/*.md" in gitignore
    assert "/.work/" in gitignore
    assert "docs/development/.agents/" not in gitignore


def test_shared_pycharm_workflows_remain_visible_to_jetbrains_agents() -> None:
    aiignore = _read(".aiignore")
    assert ".run/*" in aiignore
    assert "!.run/GitHub - *.run.xml" in aiignore

    shared_actions = sorted((REPOSITORY_ROOT / ".run").glob("GitHub - *.run.xml"))
    assert shared_actions
    for action in shared_actions:
        action_text = action.read_text(encoding="utf-8")
        assert "$PROJECT_DIR$" in action_text


def test_work_definition_and_discovery_docs_use_local_work_by_default() -> None:
    template = _read("docs/development/work-definition.md")
    readme = _read("docs/development/AGENTS.md")
    required_columns = (
        "Problem",
        "Current implementation",
        "Desired solution",
        "Task details",
        "Branch",
        "Status",
    )

    assert ".work/<work-name>.md" in template
    assert "docs/development/work/<work-name>.md" in template
    assert "only when" in template
    assert "ignored root `.work/<work-name>.md`" in template
    assert "docs/development/.agents/" not in template
    assert "vertical-align: top" in template
    assert "failed build or release workflow" in template.lower()
    for column in required_columns:
        assert column in template

    assert ".work/<work-name>.md" in readme
    assert "docs/development/work/<work-name>.md" in readme
    assert "only when" in readme
    assert "docs/development/.agents/" not in readme
    assert ".aiassistant/rules/repository-instructions.md" in readme

    assistance = _read("docs/development/ai-assistance.md")
    normalized_assistance = " ".join(assistance.split())
    assert "## PyCharm contributor setup" in assistance
    assert "Ctrl+Shift+N" in assistance
    assert "it is not a browser for existing tracked rule files" in normalized_assistance
    assert "Do not create a duplicate rule" in normalized_assistance


def test_contributing_guide_describes_the_complete_work_cycle() -> None:
    contributing = _read("CONTRIBUTING.md")
    normalized_contributing = " ".join(contributing.split())

    for reference in (
        "docs/development/AGENTS.md",
        "docs/development/work-definition.md",
        "docs/development/testing.md",
        "docs/development/skills.md",
    ):
        assert reference in contributing

    for workflow_contract in (
        "git pull --ff-only origin main",
        ".work/<work-name>.md",
        "git diff --check",
        "git branch -d",
        "git push origin --delete",
        "$caveviewer-work-cycle",
        ".agents/skills/",
    ):
        assert workflow_contract in contributing

    assert "Never delete a topic branch before verifying the merge" in normalized_contributing
    assert "human contributors do not need Codex" in normalized_contributing


def test_pull_request_description_template_matches_contributor_policy() -> None:
    contributing = _read("CONTRIBUTING.md")
    template = _read(".github/pull_request_template.md")
    copilot_instructions = _read(".github/copilot-instructions.md")
    headings = (
        "## Summary",
        "## Problem",
        "## Solution",
        "## Known limitations",
    )

    assert [template.index(heading) for heading in headings] == sorted(
        template.index(heading) for heading in headings
    )
    assert "## Validation" not in template
    assert "Do not add a **Validation** section" in contributing
    assert "test results" in contributing
    assert "evidence in the work definition" in contributing
    assert "CONTRIBUTING.md" in copilot_instructions
    assert "keep validation information out" in copilot_instructions
