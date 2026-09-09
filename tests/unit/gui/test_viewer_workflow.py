"""Tests for session-scoped non-OpenGL viewer workflow composition."""

from __future__ import annotations

from dataclasses import replace

import pytest

from caveviewer.gui.viewer_action_dispatch import ViewerKeyPressActions
from caveviewer.gui.viewer_capture_workflow import CaptureOverlayMode, CaptureOwner
from caveviewer.gui.viewer_frame_scheduler import ViewerFramePhase
from caveviewer.gui.viewer_session import (
    ViewerLaunchMode,
    ViewerSession,
    ViewerSessionConfig,
)
from caveviewer.gui.viewer_workflow import (
    ViewerShutdownPhase,
    ViewerWorkflowCoordinator,
    ViewerWorkflowSnapshot,
)


def _session() -> ViewerSession:
    return ViewerSession(
        ViewerSessionConfig(
            mode=ViewerLaunchMode.READY_CACHE,
            cache_dir="/cache",
            manifest={},
        )
    )


def _snapshot(**changes) -> ViewerWorkflowSnapshot:
    state = ViewerWorkflowSnapshot(
        setup_complete=True,
        closing_requested=False,
        iconified=False,
        import_active=False,
        map_loaded=True,
        capture_close_pending=False,
        recording_owned=False,
        recording_armed=False,
        recording_active=False,
        manual_dive_trace_countdown_active=False,
        manual_dive_trace_active=False,
        manual_dive_trace_finalizing=False,
        slice_countdown_active=False,
        slice_selection_active=False,
        slice_saving=False,
        slice_export_active=False,
    )
    return replace(state, **changes)


def test_render_request_resolves_cross_workflow_policy_from_one_snapshot():
    coordinator = ViewerWorkflowCoordinator(_session())

    request = coordinator.render_request(
        _snapshot(
            capture_close_pending=True,
            recording_owned=True,
            manual_dive_trace_countdown_active=True,
            slice_countdown_active=True,
            slice_export_active=True,
        )
    )

    assert request.phase is ViewerFramePhase.FINALIZING_CAPTURE
    assert request.capture_owner is CaptureOwner.VIDEO
    assert request.active_capture_owner is None
    assert (
        request.capture_overlay_mode
        is CaptureOverlayMode.MANUAL_DIVE_TRACE_COUNTDOWN
    )
    assert request.slice_work_pending
    assert request.slice_interaction_active


def test_render_request_keeps_active_capture_distinct_from_finalizers():
    coordinator = ViewerWorkflowCoordinator(_session())

    request = coordinator.render_request(
        _snapshot(
            manual_dive_trace_finalizing=True,
            slice_selection_active=True,
        )
    )

    assert request.capture_owner is CaptureOwner.DIVE_TRACE
    assert request.active_capture_owner is CaptureOwner.SLICE


def test_coordinator_owns_controller_instances_for_exactly_one_session():
    first = ViewerWorkflowCoordinator(_session(), recording_frame_interval=0.1)
    second = ViewerWorkflowCoordinator(_session(), recording_frame_interval=0.2)

    assert first.recording.frame_interval == 0.1
    assert second.recording.frame_interval == 0.2
    assert first.frame_scheduler is not second.frame_scheduler
    assert first.capture is not second.capture
    assert first.map_opening is not second.map_opening


def test_import_controller_factory_runs_once_per_coordinator():
    coordinator = ViewerWorkflowCoordinator(_session())
    created = []

    def factory():
        controller = object()
        created.append(controller)
        return controller

    assert coordinator.ensure_import_controller(factory) is created[0]
    assert coordinator.ensure_import_controller(factory) is created[0]
    assert len(created) == 1


def test_key_action_dispatch_is_exposed_as_one_coordinator_transition():
    coordinator = ViewerWorkflowCoordinator(_session())
    calls = []

    def action(name: str, handled: bool = False):
        def run() -> bool:
            calls.append(name)
            return handled

        return run

    actions = ViewerKeyPressActions(
        **{
            name: action(name, handled=name == "begin_screen")
            for name in ViewerKeyPressActions.__dataclass_fields__
        }
    )

    assert coordinator.dispatch_key_press(actions)
    assert calls == [
        "window_shortcut",
        "capture_escape",
        "recorded_dive",
        "begin_screen",
    ]


def test_benchmark_completion_and_shutdown_are_idempotent():
    class Benchmark:
        finished = False

        def __init__(self):
            self.reasons = []

        def finish(self, *, reason):
            self.reasons.append(reason)
            self.finished = True

    coordinator = ViewerWorkflowCoordinator(_session())
    benchmark = Benchmark()
    coordinator.set_benchmark_controller(benchmark)
    coordinator.capture.begin_exit_finalization()

    assert coordinator.finish_benchmark(reason="completed")
    assert not coordinator.finish_benchmark(reason="viewer_closed")
    assert benchmark.reasons == ["completed"]
    assert coordinator.begin_shutdown()
    assert not coordinator.capture.close_pending
    assert not coordinator.begin_shutdown()

    coordinator.complete_shutdown()
    coordinator.complete_shutdown()

    assert coordinator.shutdown_phase is ViewerShutdownPhase.CLOSED


def test_window_shutdown_marks_workflow_closed_after_render_cleanup_failure():
    from caveviewer.gui import viewer_window

    window = object.__new__(viewer_window.CaveViewerWindow)
    coordinator = ViewerWorkflowCoordinator(_session())
    window._workflow_coordinator = coordinator
    window._closing_requested = False
    window._has_map_loaded = False
    window.__dict__["_import_controller"] = type(
        "ImportController",
        (),
        {"active": False},
    )()
    window._release_window_resources = lambda: (_ for _ in ()).throw(
        RuntimeError("release failed")
    )

    with pytest.raises(RuntimeError, match="release failed"):
        window._complete_window_close()

    assert window._closing_requested
    assert coordinator.shutdown_phase is ViewerShutdownPhase.CLOSED
    window._complete_window_close()


def test_window_controller_adapters_return_session_owned_controllers():
    from caveviewer.gui import viewer_window

    window = object.__new__(viewer_window.CaveViewerWindow)
    coordinator = ViewerWorkflowCoordinator(_session())
    window._workflow_coordinator = coordinator

    assert window._ensure_frame_scheduler() is coordinator.frame_scheduler
    assert window._ensure_capture_workflow() is coordinator.capture
    assert window._ensure_action_dispatcher() is coordinator.actions
    assert window._ensure_recording_controller() is coordinator.recording
    assert window._ensure_map_opening_progress_session() is coordinator.map_opening
    assert (
        window._ensure_manual_dive_trace_controller()
        is coordinator.manual_dive_trace
    )
    assert window._ensure_slice_selection_controller() is coordinator.slice_selection
    assert window._ensure_slice_export_controller() is coordinator.slice_export
    assert (
        window._ensure_artifact_capture_presentation()
        is coordinator.artifact_presentation
    )
