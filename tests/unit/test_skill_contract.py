"""Contract tests for repository-scoped CaveViewer skills."""

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPOSITORY_ROOT / ".agents" / "skills"
EXPECTED_SKILLS = {
    "caveviewer-branding",
    "caveviewer-desktop-ux",
    "caveviewer-import-lifecycle",
    "caveviewer-performance",
    "caveviewer-release",
    "caveviewer-screenshot-polish",
    "caveviewer-work-cycle",
}
SKILL_NAME_PATTERN = re.compile(r"caveviewer-[a-z0-9]+(?:-[a-z0-9]+)*\Z")
DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"docs/development/[a-z0-9]+(?:-[a-z0-9]+)*\.md"
)


def _parse_frontmatter(skill_path: Path) -> tuple[dict[str, str], str]:
    text = skill_path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"missing YAML frontmatter: {skill_path}"

    opening, header, body = text.split("---", maxsplit=2)
    assert opening == ""
    fields: dict[str, str] = {}
    for line in header.strip().splitlines():
        key, separator, value = line.partition(":")
        assert separator, f"invalid frontmatter line in {skill_path}: {line}"
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields, body.strip()


def test_repository_skill_inventory_is_complete() -> None:
    assert SKILLS_ROOT.is_dir()
    actual_skills = {
        path.name for path in SKILLS_ROOT.iterdir() if path.is_dir()
    }
    assert actual_skills == EXPECTED_SKILLS

    skill_guide = (
        REPOSITORY_ROOT / "docs" / "development" / "skills.md"
    ).read_text(encoding="utf-8")
    for skill_name in EXPECTED_SKILLS:
        assert f"`${skill_name}`" in skill_guide


def test_skill_entrypoints_have_valid_identity_and_document_routes() -> None:
    descriptions: set[str] = set()

    for skill_name in sorted(EXPECTED_SKILLS):
        skill_root = SKILLS_ROOT / skill_name
        skill_path = skill_root / "SKILL.md"
        assert skill_path.is_file()
        assert not (skill_root / "README.md").exists()

        fields, body = _parse_frontmatter(skill_path)
        assert set(fields) == {"name", "description"}
        assert fields["name"] == skill_name
        assert SKILL_NAME_PATTERN.fullmatch(fields["name"])
        assert len(fields["name"]) <= 64
        assert fields["description"]
        assert len(fields["description"]) <= 1024
        assert "<" not in fields["description"]
        assert ">" not in fields["description"]
        assert fields["description"] not in descriptions
        assert body
        assert "[TODO:" not in body
        descriptions.add(fields["description"])

        document_references = set(DOCUMENT_REFERENCE_PATTERN.findall(body))
        assert document_references, f"no canonical document route: {skill_path}"
        for relative_path in document_references:
            assert (REPOSITORY_ROOT / relative_path).is_file(), relative_path


def test_repository_skills_are_not_explicitly_ignored() -> None:
    gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "/.agents/" not in gitignore
    assert ".agents/skills/" not in gitignore


def test_work_cycle_skill_preserves_lifecycle_and_authorization_gates() -> None:
    work_cycle = (
        SKILLS_ROOT / "caveviewer-work-cycle" / "SKILL.md"
    ).read_text(encoding="utf-8")
    normalized_work_cycle = " ".join(work_cycle.split())

    for lifecycle_contract in (
        "docs/development/work-definition.md",
        "fast-forward",
        "focused tests",
        "pull request actually merged",
        "Delete the local topic branch only after merge verification",
        "begin again from the newly updated `main`",
    ):
        assert lifecycle_contract in normalized_work_cycle

    assert "current request explicitly authorizes it" in normalized_work_cycle
    assert "skill activation alone is not" in normalized_work_cycle
    assert "$caveviewer-release" in normalized_work_cycle


def test_work_cycle_skill_preserves_pull_request_description_contract() -> None:
    work_cycle = (
        SKILLS_ROOT / "caveviewer-work-cycle" / "SKILL.md"
    ).read_text(encoding="utf-8")
    normalized_work_cycle = " ".join(work_cycle.split())

    for section_name in (
        "**Summary**",
        "**Problem**",
        "**Solution**",
        "**Known limitations**",
    ):
        assert section_name in normalized_work_cycle

    assert "Do not include a **Validation** section" in normalized_work_cycle
    assert (
        "Keep verification evidence in the work definition"
        in normalized_work_cycle
    )
