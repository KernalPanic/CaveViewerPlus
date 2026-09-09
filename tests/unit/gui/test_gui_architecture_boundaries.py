"""Architecture guardrails for GUI package boundaries."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GUI_ROOT = REPO_ROOT / "src" / "caveviewer" / "gui"
GUI_PLATFORM_ROOT = GUI_ROOT / "platform"
GUI_FEATURES_ROOT = GUI_ROOT / "features"
FEATURE_POLICY_MODULE = GUI_FEATURES_ROOT / "policies.py"
PLATFORM_RUNTIME_MODULE = GUI_PLATFORM_ROOT / "runtime.py"
RETIRED_SPLASH_ADAPTER_MODULES = (
    GUI_PLATFORM_ROOT / "base.py",
    GUI_PLATFORM_ROOT / "default.py",
    GUI_PLATFORM_ROOT / "factory.py",
    GUI_PLATFORM_ROOT / "linux.py",
    GUI_PLATFORM_ROOT / "macos.py",
    GUI_PLATFORM_ROOT / "windows.py",
)
UPDATE_PACKAGE_REVEAL_MODULE = GUI_PLATFORM_ROOT / "update_package_reveal.py"
UPDATE_PACKAGE_STORAGE_MODULE = GUI_PLATFORM_ROOT / "update_package_storage.py"
UPDATE_MANAGER_MODULE = GUI_ROOT / "update_manager.py"
STANDARD_LIBRARY_MAPS_MODULE = GUI_ROOT / "standard_library_maps.py"
VIEWER_WINDOW_MODULE = GUI_ROOT / "viewer_window.py"
VIEWER_SESSION_COORDINATOR_MODULES = (
    GUI_ROOT / "viewer_action_dispatch.py",
    GUI_ROOT / "viewer_capture_workflow.py",
    GUI_ROOT / "viewer_frame_scheduler.py",
    GUI_ROOT / "viewer_workflow.py",
)
APP_MODULE = "caveviewer.app"
_LEGACY_STATIC_PRESENTATION_ACCESSORS = {
    "ui_font_family",
    "font_candidates",
    "splash_layout_policy",
    "preferences_dialog_layout_policy",
    "dialog_layout_policy",
    "bookmark_save_modifier",
    "primary_shortcut_modifier_label",
    "tk_primary_modifier_name",
    "mouse_look_button_name",
    "compact_manual_controls_layout",
    "default_text_antialiasing_mode",
    "supports_tk_display_scaling",
    "command_modifier_uses_control_fallback",
    "shift_digit_bookmark_save_fallback",
    "option_left_mouse_look_enabled",
    "viewer_uses_glfw_native_initial_size",
}
@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    detail: str


def _gui_python_files() -> list[Path]:
    return sorted(path for path in GUI_ROOT.rglob("*.py") if path.is_file())


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _constructor_calls(tree: ast.Module, constructor_name: str) -> list[ast.Call]:
    """Return calls that construct a named feature-boundary type.

    Import aliases are included so the boundary cannot be bypassed by renaming
    a direct import. Attribute calls are safe to treat as constructions here:
    both types have deliberately unique names within the GUI package.
    """
    imported_names = {constructor_name}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name == constructor_name:
                imported_names.add(alias.asname or alias.name)

    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in imported_names:
            calls.append(node)
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == constructor_name
        ):
            calls.append(node)
    return calls


def _is_self_attribute(node: ast.expr, attribute_name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == attribute_name
    )


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef | None:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                return member
    return None


def _assignment_values(
    method: ast.FunctionDef, target_attribute: str
) -> list[ast.expr]:
    values: list[ast.expr] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        if any(_is_self_attribute(target, target_attribute) for target in node.targets):
            values.append(node.value)
    return values


def _uses_update_checker_client(expression: ast.expr, client_name: str) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "update_checker"
        and node.attr == client_name
        for node in ast.walk(expression)
    )


def _format_violations(violations: list[Violation]) -> str:
    return "\n".join(
        f"{violation.path.relative_to(REPO_ROOT)}:{violation.lineno}: "
        f"{violation.detail}"
        for violation in violations
    )


def test_gui_modules_do_not_import_app_layer():
    violations: list[Violation] = []

    for path in _gui_python_files():
        for node in ast.walk(_parse_module(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == APP_MODULE or alias.name.startswith(
                        f"{APP_MODULE}."
                    ):
                        violations.append(
                            Violation(path, node.lineno, f"imports {alias.name}")
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_app_module = module == APP_MODULE or module.startswith(
                    f"{APP_MODULE}."
                )
                imports_app_from_package = module == "caveviewer" and any(
                    alias.name == "app" for alias in node.names
                )
                imports_app_relative = node.level >= 2 and (
                    module == "app"
                    or module.startswith("app.")
                    or any(alias.name == "app" for alias in node.names)
                )
                if (
                    imports_app_module
                    or imports_app_from_package
                    or imports_app_relative
                ):
                    violations.append(
                        Violation(path, node.lineno, "imports caveviewer.app")
                    )

    assert not violations, _format_violations(violations)


def test_platform_checks_stay_inside_gui_platform_adapters():
    violations: list[Violation] = []

    for path in _gui_python_files():
        if path.is_relative_to(GUI_PLATFORM_ROOT):
            continue

        tree = _parse_module(path)
        sys_aliases = set()
        os_aliases = set()
        platform_aliases = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_name = alias.name
                    local_name = alias.asname or imported_name.partition(".")[0]
                    if imported_name == "sys":
                        sys_aliases.add(local_name)
                    elif imported_name == "os":
                        os_aliases.add(local_name)
                    elif imported_name == "platform":
                        platform_aliases.add(local_name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if module == "sys" and alias.name == "platform":
                        violations.append(
                            Violation(path, node.lineno, "imports sys.platform")
                        )
                    elif module == "os" and alias.name == "name":
                        violations.append(
                            Violation(path, node.lineno, "imports os.name")
                        )
                    elif module == "platform" and alias.name in {
                        "machine",
                        "system",
                    }:
                        violations.append(
                            Violation(
                                path,
                                node.lineno,
                                f"imports platform.{alias.name}",
                            )
                        )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if isinstance(node.value, ast.Name):
                if node.value.id in sys_aliases and node.attr == "platform":
                    violations.append(
                        Violation(path, node.lineno, "uses sys.platform")
                    )
                elif node.value.id in os_aliases and node.attr == "name":
                    violations.append(Violation(path, node.lineno, "uses os.name"))
                elif (
                    node.value.id in platform_aliases
                    and node.attr in {"machine", "system"}
                ):
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            f"uses platform.{node.attr}",
                        )
                    )

    assert not violations, _format_violations(violations)


def test_feature_policies_do_not_import_platform_or_side_effect_modules():
    """Keep feature decisions as pure transforms of injected capability facts."""
    forbidden_modules = {"os", "platform", "subprocess", "sys", "tkinter"}
    violations: list[Violation] = []

    for path in sorted(GUI_FEATURES_ROOT.rglob("*.py")):
        for node in ast.walk(_parse_module(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.partition(".")[0]
                    if module_root in forbidden_modules:
                        violations.append(
                            Violation(path, node.lineno, f"imports {alias.name}")
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "caveviewer.gui.platform" or module.startswith(
                    "caveviewer.gui.platform."
                ):
                    violations.append(
                        Violation(path, node.lineno, f"imports {module}")
                    )
                elif module.partition(".")[0] in forbidden_modules:
                    violations.append(
                        Violation(path, node.lineno, f"imports {module}"))

    assert not violations, _format_violations(violations)


def test_feature_decisions_are_constructed_only_by_policies():
    """Keep product availability decisions in the pure policy module."""
    violations: list[Violation] = []

    for path in _gui_python_files():
        if path == FEATURE_POLICY_MODULE:
            continue
        for node in _constructor_calls(_parse_module(path), "FeatureDecision"):
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    "constructs FeatureDecision outside gui.features.policies",
                )
            )

    assert not violations, _format_violations(violations)


def test_feature_gate_registry_is_composed_only_by_platform_runtime():
    """Keep process-stable gate composition in one runtime boundary."""
    violations: list[Violation] = []

    for path in _gui_python_files():
        if path == PLATFORM_RUNTIME_MODULE:
            continue
        for node in _constructor_calls(_parse_module(path), "FeatureGateRegistry"):
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    "constructs FeatureGateRegistry outside gui.platform.runtime",
                )
            )

    assert not violations, _format_violations(violations)


def test_update_manager_requires_runtime_owned_typed_update_clients():
    """Keep update policy and network traffic on the runtime-owned contract."""
    manager = _parse_module(UPDATE_MANAGER_MODULE)
    initializer = _class_method(manager, "UpdateManager", "__init__")
    assert initializer is not None, "UpdateManager.__init__ is required"

    expected_clients = {
        "_check_for_update": "check_for_update_target",
        "_download_update": "download_update_target",
    }
    violations: list[Violation] = []

    for attribute_name, client_name in expected_clients.items():
        assignments = _assignment_values(initializer, attribute_name)
        if not assignments:
            violations.append(
                Violation(
                    UPDATE_MANAGER_MODULE,
                    initializer.lineno,
                    f"does not assign self.{attribute_name}",
                )
            )
        elif not any(
            _uses_update_checker_client(assignment, client_name)
            for assignment in assignments
        ):
            violations.append(
                Violation(
                    UPDATE_MANAGER_MODULE,
                    assignments[0].lineno,
                    f"does not default self.{attribute_name} to "
                    f"update_checker.{client_name}",
                )
            )

    keyword_only_arguments = {
        argument.arg for argument in initializer.args.kwonlyargs
    }
    if "platform_runtime" not in keyword_only_arguments:
        violations.append(
            Violation(
                UPDATE_MANAGER_MODULE,
                initializer.lineno,
                "does not require an injected platform_runtime",
            )
        )

    compatibility_arguments = {
        "platform_adapter",
        "desktop_services",
        "update_package_reveal_adapter",
        "update_package_storage_adapter",
    }
    for argument_name in sorted(keyword_only_arguments & compatibility_arguments):
        violations.append(
            Violation(
                UPDATE_MANAGER_MODULE,
                initializer.lineno,
                f"accepts retired direct compatibility input {argument_name}",
            )
        )

    if any(
        isinstance(node, ast.Name) and node.id == "create_platform_runtime"
        for node in ast.walk(manager)
    ):
        violations.append(
            Violation(
                UPDATE_MANAGER_MODULE,
                initializer.lineno,
                "constructs a platform runtime instead of requiring one",
            )
        )

    assert not violations, _format_violations(violations)


def test_standard_library_maps_do_not_import_update_compatibility_api():
    """Keep map downloads independent from updater compatibility behavior."""
    module = _parse_module(STANDARD_LIBRARY_MAPS_MODULE)
    violations: list[Violation] = []

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "caveviewer.gui.update_checker" for alias in node.names
            ):
                violations.append(
                    Violation(
                        STANDARD_LIBRARY_MAPS_MODULE,
                        node.lineno,
                        "imports caveviewer.gui.update_checker",
                    )
                )
        elif isinstance(node, ast.ImportFrom) and (
            node.module == "caveviewer.gui.update_checker"
            or (node.level > 0 and node.module == "update_checker")
            or (
                (node.module == "caveviewer.gui" or node.level > 0)
                and any(alias.name == "update_checker" for alias in node.names)
            )
        ):
            violations.append(
                Violation(
                    STANDARD_LIBRARY_MAPS_MODULE,
                    node.lineno,
                    "imports caveviewer.gui.update_checker",
                )
            )

    assert not violations, _format_violations(violations)


def test_retired_splash_platform_adapter_stays_removed():
    """Keep platform behavior on focused contracts after removing the broad one."""
    retired_names = {
        "SplashPlatformAdapter",
        "get_platform_adapter",
        "get_splash_platform_adapter",
    }
    violations: list[Violation] = []

    for path in RETIRED_SPLASH_ADAPTER_MODULES:
        if path.exists():
            violations.append(Violation(path, 1, "retired adapter module exists"))

    for path in _gui_python_files():
        for node in ast.walk(_parse_module(path)):
            if isinstance(node, ast.Name) and node.id in retired_names:
                violations.append(
                    Violation(path, node.lineno, f"references {node.id}")
                )
            elif isinstance(node, ast.Attribute) and node.attr == "platform_adapter":
                violations.append(
                    Violation(path, node.lineno, "references broad platform_adapter")
                )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in retired_names:
                        violations.append(
                            Violation(path, node.lineno, f"imports {alias.name}")
                        )

    assert not violations, _format_violations(violations)


def test_desktop_notification_actions_stay_inside_platform_boundary():
    """Keep best-effort notification execution behind its typed gate."""
    violations: list[Violation] = []

    for path in _gui_python_files():
        if path.is_relative_to(GUI_PLATFORM_ROOT):
            continue
        for node in ast.walk(_parse_module(path)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"notify", "withdraw_notification"}:
                continue
            receiver = node.func.value
            is_named_service = (
                isinstance(receiver, ast.Name)
                and receiver.id == "desktop_services"
            )
            is_instance_service = (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "_desktop_services"
            )
            if is_named_service or is_instance_service:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        f"calls DesktopServices.{node.func.attr} outside platform",
                    )
                )

    assert not violations, _format_violations(violations)


def test_desktop_inhibition_acquisition_stays_inside_platform_boundary():
    """Keep optional inhibitor acquisition behind its typed gate."""
    violations: list[Violation] = []

    for path in _gui_python_files():
        if path.is_relative_to(GUI_PLATFORM_ROOT):
            continue
        for node in ast.walk(_parse_module(path)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "inhibit_idle_suspend":
                continue
            receiver = node.func.value
            is_named_service = (
                isinstance(receiver, ast.Name)
                and receiver.id == "desktop_services"
            )
            is_instance_service = (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "_desktop_services"
            )
            if is_named_service or is_instance_service:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        "calls DesktopServices.inhibit_idle_suspend outside platform",
                    )
                )

    assert not violations, _format_violations(violations)


def test_viewer_does_not_construct_platform_services_at_module_import():
    """Keep process-owned platform construction out of viewer module import."""
    viewer_module = _parse_module(GUI_ROOT / "viewer_window.py")
    factory_names = {"get_desktop_services"}
    violations: list[Violation] = []

    for node in viewer_module.body:
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
            continue
        for descendant in ast.walk(node):
            if (
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Name)
                and descendant.func.id in factory_names
            ):
                violations.append(
                    Violation(
                        GUI_ROOT / "viewer_window.py",
                        descendant.lineno,
                        f"constructs {descendant.func.id} during module import",
                    )
                )

    assert not violations, _format_violations(violations)


def test_viewer_session_coordinators_do_not_import_opengl():
    """Keep viewer session policy usable without a window or GL context."""
    prohibited_modules = {"moderngl", "moderngl_window"}
    violations: list[Violation] = []

    for path in VIEWER_SESSION_COORDINATOR_MODULES:
        for node in ast.walk(_parse_module(path)):
            if isinstance(node, ast.Import):
                imported_modules = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported_modules = {(node.module or "").split(".", 1)[0]}
            else:
                continue
            for module in imported_modules & prohibited_modules:
                violations.append(
                    Violation(path, node.lineno, f"imports OpenGL module {module}")
                )

    assert not violations, _format_violations(violations)


def test_viewer_window_composes_non_gl_state_through_workflow_coordinator():
    """Keep production controller identity on one session-scoped owner."""
    viewer_module = _parse_module(VIEWER_WINDOW_MODULE)
    initializer = _class_method(viewer_module, "CaveViewerWindow", "__init__")
    assert initializer is not None

    coordinator_assignments = _assignment_values(
        initializer,
        "_workflow_coordinator",
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ViewerWorkflowCoordinator"
        for expression in coordinator_assignments
        for node in ast.walk(expression)
    )

    independently_owned_attributes = {
        "_action_dispatcher",
        "_artifact_capture_presentation",
        "_capture_workflow",
        "_frame_scheduler",
        "_manual_dive_trace_controller",
        "_map_opening_progress_session",
        "_recording_controller",
        "_slice_export_controller",
        "_slice_selection_controller",
    }
    violations = [
        Violation(
            VIEWER_WINDOW_MODULE,
            initializer.lineno,
            f"constructs independent self.{attribute_name} in __init__",
        )
        for attribute_name in sorted(independently_owned_attributes)
        if _assignment_values(initializer, attribute_name)
    ]

    assert not violations, _format_violations(violations)


def test_gui_consumers_do_not_call_legacy_static_presentation_accessors():
    """Keep static UI conventions on PresentationProfile, not the broad adapter."""
    violations: list[Violation] = []

    for path in _gui_python_files():
        if path.is_relative_to(GUI_PLATFORM_ROOT):
            continue
        for node in ast.walk(_parse_module(path)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in _LEGACY_STATIC_PRESENTATION_ACCESSORS:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        f"calls legacy static presentation accessor {node.func.attr}()",
                    )
                )

    assert not violations, _format_violations(violations)


def test_gui_modules_have_ownership_docstrings():
    violations: list[Violation] = []

    for path in _gui_python_files():
        module = _parse_module(path)
        docstring = ast.get_docstring(module)
        if not docstring:
            violations.append(Violation(path, 1, "missing module docstring"))
            continue

        first_line = docstring.strip().splitlines()[0].strip()
        if first_line.startswith("caveviewer.gui."):
            violations.append(
                Violation(path, 1, f"placeholder module docstring: {first_line}")
            )

    assert not violations, _format_violations(violations)
