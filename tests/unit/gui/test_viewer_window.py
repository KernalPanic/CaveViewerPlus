"""Tests for viewer-window startup sizing."""

from __future__ import annotations

import hashlib
import inspect
import logging
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from caveviewer.core.capabilities import (
    CapabilityResult,
    ViewerLaunchRoute,
    ViewerLaunchTarget,
    WindowBackendPlan,
    WindowSystem,
)
from caveviewer.core.map import cache_paths
from caveviewer.core.preferences import runtime_settings
from caveviewer.gui import recording, viewer_window
from caveviewer.gui.features import (
    FeatureDecision,
    FeatureId,
    FeatureState,
    decide_video_recording,
)
from caveviewer.gui.manual_dive_trace import ManualDivePose, ManualDiveTraceResult
from caveviewer.gui.platform.app_identity import tk_root_options
from caveviewer.gui.platform.presentation import select_presentation_profile
from caveviewer.gui.platform.runtime import VideoRecordingPreflight, ViewerLaunchPreflight
from caveviewer.gui.platform.viewer_launch import ViewerLaunchError


def _viewer_session(*, platform_runtime=None):
    return viewer_window.ViewerSession(
        viewer_window.ViewerSessionConfig(
            mode=viewer_window.ViewerLaunchMode.READY_CACHE,
            cache_dir="/cache",
            textures_dir="/textures",
            manifest={"chunks": {}},
            platform_runtime=platform_runtime,
        )
    )


def _pending_import_session():
    return viewer_window.ViewerSession(
        viewer_window.ViewerSessionConfig(
            mode=viewer_window.ViewerLaunchMode.PENDING_IMPORT,
            pending_import=viewer_window.PendingImportRequest(
                model_descriptor={"obj_path": "/maps/cave.obj"},
                textures_dir="/maps",
            ),
        )
    )


def test_viewer_default_framebuffer_uses_multisampling_for_graphics_edges():
    assert viewer_window.CaveViewerWindow.samples == 4


def test_pyglet_close_event_is_claimed_before_the_default_close_handler():
    calls = []

    class NativeWindow:
        def push_handlers(self, **handlers):
            self.handlers = handlers

    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(name="pyglet", _window=NativeWindow())
    window.on_close = lambda: calls.append("close")

    window._claim_backend_close_event()

    assert window.wnd._window.handlers["on_close"]() is True
    assert calls == ["close"]


def test_non_pyglet_close_event_is_left_to_its_native_backend():
    calls = []
    native_window = SimpleNamespace(
        push_handlers=lambda **_handlers: calls.append("claimed")
    )
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(name="glfw", _window=native_window)

    window._claim_backend_close_event()

    assert calls == []


class FakeImportInhibitor:
    def __init__(self, calls):
        self._calls = calls

    def close(self):
        self._calls.append(("close_inhibitor",))


class FakeImportProcess:
    def __init__(self, calls=None, exitcode=None):
        self._calls = [] if calls is None else calls
        self.exitcode = exitcode
        self.joined = False
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self.exitcode is None and not self.terminated

    def join(self, timeout=None):
        self._calls.append(("join_process", timeout))
        self.joined = True
        if self.exitcode is None:
            self.exitcode = 0

    def terminate(self):
        self._calls.append(("terminate_process",))
        self.terminated = True
        self.exitcode = -15

    def kill(self):
        self._calls.append(("kill_process",))
        self.killed = True
        self.exitcode = -9


class FakeLogger:
    def __init__(self):
        self.error_messages = []
        self.info_messages = []
        self.warning_messages = []
        self.debug_messages = []

    @staticmethod
    def _format(message, args):
        return message % args if args else str(message)

    def error(self, message, *args):
        self.error_messages.append(self._format(message, args))

    def info(self, message, *args):
        self.info_messages.append(self._format(message, args))

    def warning(self, message, *args):
        self.warning_messages.append(self._format(message, args))

    def debug(self, message, *args):
        self.debug_messages.append(self._format(message, args))




class FakeFuture:
    def __init__(self, result=None, *, done=True, exception=None):
        self._result = result
        self._done = done
        self._exception = exception
        self.cancelled = False
        self.result_called = False

    def done(self):
        return self._done

    def result(self):
        self.result_called = True
        if self._exception is not None:
            raise self._exception
        return self._result

    def cancel(self):
        self.cancelled = True
        return True


class FakeExecutor:
    def __init__(self, future):
        self.future = future
        self.submit_calls = []
        self.shutdown_calls = []

    def submit(self, fn, *args, **kwargs):
        self.submit_calls.append((fn, args, kwargs))
        return self.future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


class FakeTextureValidationManager:
    def __init__(self):
        self.calls = 0

    def validate_textures(self):
        self.calls += 1
        return {"found": ["texture.jpg"], "missing": []}


def test_benchmark_route_prefetch_stats_reports_missing_route_cells():
    window = viewer_window.CaveViewerWindow.__new__(viewer_window.CaveViewerWindow)
    window._benchmark_route_prefetch_cells = frozenset(
        {
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
        }
    )
    window.world = SimpleNamespace(
        loaded_cells={(0, 0, 0)},
        _pending={(1, 0, 0)},
        _failed_cells={},
    )

    stats = window._benchmark_route_prefetch_stats()

    assert stats == {
        "active": True,
        "ready": False,
        "expected_cells": 3,
        "loaded_cells": 1,
        "pending_cells": 1,
        "failed_cells": 0,
        "missing_cells": 2,
        "coverage_pct": pytest.approx(100.0 / 3.0),
    }


def test_slice_storage_uses_runtime_settings_snapshot(tmp_path, monkeypatch):
    storage_dir = tmp_path / "snapshot-maps"
    snapshot = runtime_settings.resolve_runtime_settings(
        preferences={"map_library_dir": str(storage_dir)},
        environ={},
        platform=runtime_settings.RuntimePlatformFacts(
            platform_name="linux",
            os_name="posix",
            home=tmp_path,
        ),
    )
    from caveviewer.gui import preferences

    monkeypatch.setattr(
        preferences,
        "load_preferences",
        lambda: (_ for _ in ()).throw(AssertionError("legacy preferences read")),
    )
    window = viewer_window.CaveViewerWindow.__new__(viewer_window.CaveViewerWindow)
    window._runtime_settings = snapshot

    assert window._slice_storage_directory() == str(storage_dir)
    assert storage_dir.is_dir()


class FakeImportThread:
    def __init__(self, alive=True):
        self._alive = alive
        self.join_calls = []

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        self._alive = False


def _import_window():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._viewer_session = _pending_import_session()
    window._import_active = False
    window._import_is_startup = False
    window._import_thread = None
    window._import_process = None
    window._import_command_queue = None
    window._import_queue = None
    window._import_cache_dir = None
    window._import_stop_event = None
    window._import_pause_requested = False
    window._import_model_format = None
    window._import_map_name = ""
    window._import_progress_stage = ""
    window._import_progress_fraction = 0.0
    window._import_progress_title = ""
    window._import_progress_note = ""
    window._import_resuming_from_checkpoint = False
    window._import_pause_notice_until = None
    window._import_pause_notice_close_after = False
    window._import_pause_notice_map_name = ""
    window._import_pause_notice_title = "Import paused"
    window._import_pause_notice_stage = "resume point saved"
    window._import_pause_notice_note = ""
    window._has_map_loaded = False
    window._pending_import_started = False
    window._pending_import_splash_rendered = False
    return window


def _wait_for_import_worker(window):
    window._import_thread.join(timeout=2.0)
    assert not window._import_thread.is_alive()


def _queued_import_messages(window):
    messages = []
    while not window._import_queue.empty():
        messages.append(window._import_queue.get_nowait())
    return messages


def _recording_window():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._platform_runtime = None
    window._recording_output_dir = "/tmp"
    window._recording_countdown_started_at = None
    window._recording_countdown_until = None
    window._recording_session = None
    window._recording_output_path = None
    window._recording_size = None
    window._recording_viewport = None
    window._recording_readback_framebuffer = None
    window._recording_readback_slots = []
    window._recording_readback_pending = []
    window._recording_readback_byte_count = 0
    window._recording_last_stage_ms = 0.0
    window._recording_last_drain_ms = 0.0
    window._recording_next_frame_time = None
    window._recording_frame_interval = 1.0 / 30.0
    window._recording_frame_queue = None
    window._recording_dropped_frames = 0
    window._recording_stop_results = queue.Queue()
    window._recording_stop_thread = None
    window._recording_status_message = None
    window._recording_status_detail = None
    window._recording_status_kind = None
    window._recording_status_until = None
    return window


def _begin_exit_capture_finalization(
    window,
    *,
    status_presented_at: float | None = None,
):
    """Prepare a lightweight viewer double for capture-finalization tests."""
    workflow = window._ensure_capture_workflow()
    workflow.begin_exit_finalization()
    workflow.exit_status_presented_at = status_presented_at
    return workflow


def test_viewer_uses_injected_recording_process_adapter_before_legacy_factory(
    monkeypatch,
):
    adapter = object()
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._platform_runtime = SimpleNamespace(recording_process_adapter=adapter)

    assert window._active_recording_process_adapter() is adapter


def test_recording_target_uses_shared_preflight_with_injected_runtime(monkeypatch):
    target = viewer_window.VideoRecordingTarget("/usr/bin/ffmpeg", "/recordings")
    capability = CapabilityResult.available(
        target,
        reason_code="video_recording_target_available",
    )
    preflight = VideoRecordingPreflight(
        capability=capability,
        decision=decide_video_recording(capability),
    )
    calls = []
    runtime_instance = object()

    def preflight_factory(
        output_directory,
        *,
        ffmpeg_resolver=None,
        platform_runtime=None,
    ):
        calls.append((output_directory, ffmpeg_resolver, platform_runtime))
        return preflight

    window = _recording_window()
    window._platform_runtime = runtime_instance
    monkeypatch.setattr(viewer_window, "video_recording_preflight", preflight_factory)

    assert window._recording_target_if_available() is target
    assert calls[0][0] == "/tmp"
    assert calls[0][1] is not None
    assert calls[0][2] is runtime_instance


def test_recording_target_uses_shared_preflight_without_runtime(monkeypatch):
    target = viewer_window.VideoRecordingTarget("/usr/bin/ffmpeg", "/recordings")
    capability = CapabilityResult.available(
        target,
        reason_code="video_recording_target_available",
    )
    preflight = VideoRecordingPreflight(
        capability=capability,
        decision=decide_video_recording(capability),
    )
    calls = []

    def preflight_factory(
        output_directory,
        *,
        ffmpeg_resolver=None,
        platform_runtime=None,
    ):
        calls.append((output_directory, ffmpeg_resolver, platform_runtime))
        return preflight

    window = _recording_window()
    monkeypatch.setattr(viewer_window, "video_recording_preflight", preflight_factory)

    assert window._recording_target_if_available() is target
    assert calls[0][0] == "/tmp"
    assert calls[0][1] is not None
    assert calls[0][2] is None


def test_map_import_inhibitor_uses_the_runtime_desktop_service():
    calls = []
    inhibitor = object()
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._platform_runtime = SimpleNamespace(
        desktop_services=SimpleNamespace(
            inhibit_idle_suspend=lambda reason: calls.append(reason) or inhibitor
        )
    )

    assert window._acquire_import_inhibitor("Crystal Cave") is inhibitor
    assert calls == ["Importing Crystal Cave"]


def _active_recording_session(
    *,
    process=None,
    frame_queue: queue.Queue | None = None,
    output_path: str = "/recordings/cave.mp4",
    output_size: tuple[int, int] = (2, 2),
    viewport: tuple[int, int, int, int] = (0, 0, 2, 2),
) -> recording.RecordingEncoderSession:
    if process is None:
        process = SimpleNamespace(poll=lambda: None)
    if frame_queue is None:
        frame_queue = queue.Queue(maxsize=2)
    return recording.RecordingEncoderSession(
        process=process,
        output_path=output_path,
        output_size=output_size,
        viewport=viewport,
        frame_queue=frame_queue,
    )


def _manual_trace_camera(position=(1.0, 2.0, 3.0)):
    return SimpleNamespace(
        position=np.asarray(position, dtype=np.float64),
        forward=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64),
        up=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64),
        right=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        move_speed=4.0,
    )


class FakeManualDiveTrace:
    def __init__(self):
        self.writer_failed = False
        self.output_path = "/maps/cave/_guided_dives/trace.jsonl"
        self.observed = []
        self.stopped = []
        self.cancel_calls = 0
        self.discontinuities = []
        self.result = None

    def observe(self, pose):
        self.observed.append(pose)

    def stop(self, pose, *, reason):
        self.stopped.append((pose, reason))
        return self.output_path

    def cancel(self):
        self.cancel_calls += 1
        return True

    def mark_discontinuity(self, before, after, *, reason):
        self.discontinuities.append((before, after, reason))

    def poll_result(self):
        return self.result


def _pending_manual_trace_writer(
    recorder,
    *,
    show_completion: bool = True,
    reveal_on_success: bool = True,
):
    return viewer_window._PendingManualDiveTraceWriter(
        recorder=recorder,
        show_completion=show_completion,
        reveal_on_success=reveal_on_success,
    )


@pytest.mark.parametrize(
    ("presentation_profile", "primary_modifiers"),
    [
        (select_presentation_profile(platform_name="unsupported"), SimpleNamespace(ctrl=True)),
        (select_presentation_profile(platform_name="darwin"), SimpleNamespace(command=True)),
    ],
)
def test_manual_trace_hotkey_uses_platform_primary_modifier(
    presentation_profile,
    primary_modifiers,
):
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = presentation_profile
    window._has_map_loaded = True
    window.wnd = SimpleNamespace(keys=SimpleNamespace(T=84))
    window._keys_down = set()
    window._key_resolve_cache = {}
    window._raw_command_modifier_down = lambda: False
    calls = []
    window._toggle_manual_dive_trace = lambda: calls.append("toggle") or True

    assert window._handle_manual_dive_trace_hotkey(84, primary_modifiers) is True
    assert calls == ["toggle"]
    assert window._handle_manual_dive_trace_hotkey(85, primary_modifiers) is False
    assert (
        window._handle_manual_dive_trace_hotkey(84, SimpleNamespace(shift=True))
        is False
    )


def _capture_hotkey_window(active_owner):
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = select_presentation_profile(
        platform_name="unsupported"
    )
    window._has_map_loaded = True
    window.wnd = SimpleNamespace(keys=SimpleNamespace(R=82, T=84, C=67))
    window._keys_down = set()
    window._key_resolve_cache = {}
    window._raw_command_modifier_down = lambda: False
    window._capture_owner = lambda: active_owner
    toggle_calls = []
    status_calls = []
    window._toggle_recording = lambda: toggle_calls.append(
        viewer_window.CaptureOwner.VIDEO
    )
    window._toggle_manual_dive_trace = lambda: toggle_calls.append(
        viewer_window.CaptureOwner.DIVE_TRACE
    ) or True
    window._toggle_slice = lambda: toggle_calls.append(
        viewer_window.CaptureOwner.SLICE
    ) or True
    window._show_capture_status = (
        lambda *args, **kwargs: status_calls.append((args, kwargs))
    )
    return window, toggle_calls, status_calls


@pytest.mark.parametrize(
    ("active_owner", "handler_name", "key"),
    (
        (
            viewer_window.CaptureOwner.VIDEO,
            "_handle_manual_dive_trace_hotkey",
            84,
        ),
        (viewer_window.CaptureOwner.VIDEO, "_handle_slice_hotkey", 67),
        (
            viewer_window.CaptureOwner.DIVE_TRACE,
            "_handle_recording_hotkey",
            82,
        ),
        (viewer_window.CaptureOwner.DIVE_TRACE, "_handle_slice_hotkey", 67),
        (viewer_window.CaptureOwner.SLICE, "_handle_recording_hotkey", 82),
        (
            viewer_window.CaptureOwner.SLICE,
            "_handle_manual_dive_trace_hotkey",
            84,
        ),
    ),
)
def test_capture_hotkeys_silently_ignore_a_different_active_owner(
    active_owner,
    handler_name,
    key,
):
    window, toggle_calls, status_calls = _capture_hotkey_window(active_owner)

    assert getattr(window, handler_name)(key, SimpleNamespace(ctrl=True)) is True

    assert toggle_calls == []
    assert status_calls == []


@pytest.mark.parametrize(
    ("active_owner", "handler_name", "key"),
    (
        (viewer_window.CaptureOwner.VIDEO, "_handle_recording_hotkey", 82),
        (
            viewer_window.CaptureOwner.DIVE_TRACE,
            "_handle_manual_dive_trace_hotkey",
            84,
        ),
        (viewer_window.CaptureOwner.SLICE, "_handle_slice_hotkey", 67),
    ),
)
def test_active_capture_owner_shortcut_still_reaches_finish_and_save(
    active_owner,
    handler_name,
    key,
):
    window, toggle_calls, status_calls = _capture_hotkey_window(active_owner)

    assert getattr(window, handler_name)(key, SimpleNamespace(ctrl=True)) is True

    assert toggle_calls == [active_owner]
    assert status_calls == []


def test_viewer_claims_escape_before_the_backend_can_preempt_key_dispatch():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(exit_key=256)

    window._claim_backend_escape_key()

    assert window.wnd.exit_key is None


def test_escape_hotkey_uses_cancel_then_delayed_close_for_active_capture():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(keys=SimpleNamespace(ESCAPE=256))
    window._key_resolve_cache = {}
    calls = []
    window._capture_owner = lambda: viewer_window.CaptureOwner.VIDEO
    window._begin_escape_capture_cancellation = (
        lambda: calls.append("cancel_then_close") or True
    )
    window.on_close = lambda: calls.append("close_now")

    assert window._handle_capture_escape_hotkey(256) is True
    assert calls == ["cancel_then_close"]
    assert window._handle_capture_escape_hotkey(257) is False


def test_escape_hotkey_closes_immediately_without_an_active_capture():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(keys=SimpleNamespace(ESCAPE=256))
    window._key_resolve_cache = {}
    window._capture_owner = lambda: None
    calls = []
    window.on_close = lambda: calls.append("close_now")

    assert window._handle_capture_escape_hotkey(256) is True

    assert calls == ["close_now"]


@pytest.mark.parametrize(
    ("owner", "expected"),
    (
        (viewer_window.CaptureOwner.VIDEO, "video"),
        (viewer_window.CaptureOwner.DIVE_TRACE, "trace"),
        (viewer_window.CaptureOwner.SLICE, "slice"),
    ),
)
def test_unified_escape_routes_to_the_single_capture_owner(owner, expected):
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._exit_capture_finalization_active = lambda: False
    window._capture_owner = lambda: owner
    calls = []
    window._cancel_recording_capture = lambda: calls.append("video") or True
    window._cancel_manual_dive_trace_capture = lambda: calls.append("trace") or True
    window._cancel_slice_interaction = lambda: calls.append("slice") or True

    assert window._cancel_active_capture() is True
    assert calls == [expected]


def test_manual_trace_countdown_hides_picker_and_manual_help(monkeypatch):
    calls = []
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 40.0)
    window = _recording_window()
    window._has_map_loaded = True
    window._manual_dive_trace = None
    window._primary_shortcut_label = lambda: "Ctrl"
    window.color_picker = SimpleNamespace(hide=lambda: calls.append("hide_picker"))
    window.controls_overlay = SimpleNamespace(
        is_manual_mode=True,
        hide_help=lambda: calls.append("hide_help"),
    )

    assert window._start_manual_dive_trace_countdown() is True

    controller = window._ensure_manual_dive_trace_controller()
    assert calls == ["hide_picker", "hide_help"]
    assert controller.countdown_started_at == 40.0
    assert controller.countdown_until == 44.0


def test_recording_cannot_start_while_a_dive_trace_owns_capture():
    window = _recording_window()
    window._has_map_loaded = True
    window._manual_dive_trace = FakeManualDiveTrace()

    window._start_recording_countdown()

    assert window._recording_countdown_until is None
    assert window._recording_status_message == "Capture in progress"
    assert window._recording_status_detail == (
        "Finish or cancel the current dive trace before starting a new video recording."
    )


def test_dive_trace_cannot_start_while_video_owns_capture():
    window = _recording_window()
    window._has_map_loaded = True
    window._manual_dive_trace = None
    window._recording_session = _active_recording_session()

    assert window._start_manual_dive_trace_countdown() is False

    assert not window._ensure_manual_dive_trace_controller().countdown_active
    assert window._recording_status_message == "Capture in progress"
    assert window._recording_status_detail == (
        "Finish or cancel the current video recording before starting a new dive trace."
    )


def test_dive_trace_cannot_start_while_video_is_finalizing():
    window = _recording_window()
    window._has_map_loaded = True
    window._manual_dive_trace = None
    window._recording_stop_thread = object()

    assert window._start_manual_dive_trace_countdown() is False

    assert not window._ensure_manual_dive_trace_controller().countdown_active
    assert window._recording_status_message == "Capture in progress"
    assert window._recording_status_detail == (
        "Finish or cancel the current video recording before starting a new dive trace."
    )


def test_recording_cannot_start_while_a_dive_trace_is_finalizing():
    window = _recording_window()
    window._has_map_loaded = True
    window._manual_dive_trace_writers = [
        _pending_manual_trace_writer(FakeManualDiveTrace())
    ]

    window._start_recording_countdown()

    assert window._recording_countdown_until is None
    assert window._recording_status_message == "Capture in progress"
    assert window._recording_status_detail == (
        "Finish or cancel the current dive trace before starting a new video recording."
    )


def test_recording_cannot_start_while_slice_selection_owns_capture():
    window = _recording_window()
    window._has_map_loaded = True
    selection = window._ensure_slice_selection_controller()
    selection.start_countdown(now=0.0, start_number=0)
    assert selection.begin_selection((1.0, 2.0, 3.0))

    window._start_recording_countdown()

    assert window._recording_countdown_until is None
    assert window._recording_status_message == "Capture in progress"
    assert window._recording_status_detail == (
        "Finish or cancel the current cave slice before starting a new video recording."
    )


def test_dive_trace_cannot_start_while_slice_selection_owns_capture():
    window = _recording_window()
    window._has_map_loaded = True
    window._manual_dive_trace = None
    selection = window._ensure_slice_selection_controller()
    selection.start_countdown(now=0.0, start_number=0)
    assert selection.begin_selection((1.0, 2.0, 3.0))

    assert window._start_manual_dive_trace_countdown() is False

    assert not window._ensure_manual_dive_trace_controller().countdown_active
    assert window._recording_status_message == "Capture in progress"
    assert window._recording_status_detail == (
        "Finish or cancel the current cave slice before starting a new dive trace."
    )


def test_manual_trace_toggle_cancels_existing_countdown(monkeypatch):
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)
    window = _recording_window()
    window._manual_dive_trace = None
    controller = window._ensure_manual_dive_trace_controller()
    controller.start_countdown(now=7.0, start_number=3)

    assert window._toggle_manual_dive_trace() is True

    assert not controller.countdown_active
    assert window._recording_status_message == "Dive trace canceled"
    assert window._recording_status_kind == "cancel"
    assert window._recording_status_until == pytest.approx(13.0)


def test_escape_cancels_recording_countdown_without_starting_a_writer(monkeypatch):
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)
    window = _recording_window()
    window._ensure_recording_controller().start_countdown(
        now=7.0,
        start_number=3,
    )

    assert window._cancel_recording_capture() is True

    assert window._recording_countdown_until is None
    assert window._recording_session is None
    assert window._recording_stop_thread is None
    assert window._recording_status_message == "Video canceled"
    assert window._recording_status_detail == "No video was saved."
    assert window._recording_status_until == pytest.approx(13.0)


def test_escape_turns_pending_video_publication_into_cleanup():
    class FakeCancelEvent:
        def __init__(self):
            self.was_set = False

        def set(self):
            self.was_set = True

    window = _recording_window()
    cancel_event = FakeCancelEvent()
    window._recording_stop_thread = object()
    window._recording_stop_cancel_event = cancel_event

    assert window._cancel_recording_capture() is True

    assert cancel_event.was_set is True
    assert window._recording_status_message == "Canceling video…"
    assert window._recording_status_until is None


def test_escape_cancels_active_trace_and_waits_for_disk_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)
    window = _recording_window()
    recorder = FakeManualDiveTrace()
    window._manual_dive_trace = recorder
    window._manual_dive_trace_writers = []

    assert window._cancel_manual_dive_trace_capture() is True

    assert recorder.cancel_calls == 1
    assert window._manual_dive_trace is None
    assert len(window._manual_dive_trace_writers) == 1
    pending = window._manual_dive_trace_writers[0]
    assert pending.show_completion is True
    assert pending.reveal_on_success is False
    assert window._recording_status_message == "Canceling dive trace…"
    assert window._recording_status_until is None

    recorder.result = ManualDiveTraceResult(
        output_path=str(tmp_path / "trace.jsonl"),
        partial_path=str(tmp_path / ".trace.jsonl.part"),
        completed=False,
        error=None,
        canceled=True,
    )
    window._update_manual_dive_trace(now=11.0)

    assert window._manual_dive_trace_writers == []
    assert window._recording_status_message == "Dive trace canceled"
    assert window._recording_status_detail == "No dive trace was saved."
    assert window._recording_status_kind == "cancel"
    assert window._recording_status_until == pytest.approx(14.0)


def test_manual_trace_starts_only_after_its_countdown_expires():
    window = _recording_window()
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = []
    controller = window._ensure_manual_dive_trace_controller()
    controller.start_countdown(now=10.0, start_number=3)
    started = []
    window._start_manual_dive_trace = lambda: started.append("started") or True

    window._update_manual_dive_trace(now=13.99)
    assert started == []
    assert controller.countdown_active

    window._update_manual_dive_trace(now=14.0)
    assert started == ["started"]
    assert not controller.countdown_active


def test_manual_trace_countdown_uses_the_shared_countdown_overlay():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(size=(800, 600))
    window.UI_TEXT_SCALE = 1.28
    calls = []
    window._render_recording_countdown_scrim = lambda size: calls.append(
        ("scrim", size)
    )
    window.import_progress_panel = SimpleNamespace(
        draw_countdown_number=lambda **kwargs: calls.append(("number", kwargs))
    )
    controller = window._ensure_manual_dive_trace_controller()
    controller.start_countdown(now=10.0, start_number=3)

    window._render_countdown_overlay(
        now=10.1,
        controller=controller,
        start_number=3,
        title="Prepare to plan a dive",
        note="Press Ctrl+T again to stop. Press Esc to cancel.",
    )

    assert calls[0] == ("scrim", (800, 600))
    assert calls[1] == (
        "number",
        {
            "center_x": 400.0,
            "center_y": 300.0,
            "window_size": (800, 600),
            "number": 3,
            "progress": pytest.approx(0.025),
            "fixed_text_scale": 1.28,
            "stage": "Prepare to plan a dive",
            "note": "Press Ctrl+T again to stop. Press Esc to cancel.",
        },
    )


def test_capture_countdown_pairs_stop_with_escape_cancellation():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._primary_shortcut_label = lambda: "Ctrl"

    assert window._countdown_cancel_note("R") == (
        "Press Ctrl+R again to stop. Press Esc to cancel."
    )


def test_capture_status_uses_import_style_message_note_layout(monkeypatch):
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(size=(800, 600))
    window.UI_TEXT_SCALE = 1.28
    controller = window._ensure_recording_controller()
    controller.show_status(
        "Video saved",
        detail="Opening its location…",
        kind="success",
        duration=3.0,
        now=10.0,
    )
    calls = []
    window._render_recording_countdown_scrim = lambda *args, **kwargs: calls.append(
        ("scrim", args, kwargs)
    )
    window.import_progress_panel = SimpleNamespace(
        draw_circle_label=lambda **kwargs: calls.append(("circle", kwargs))
    )
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)

    window._render_capture_status_message((800, 600))

    assert calls == [
        ("scrim", ((800, 600),), {"alpha": 0.42}),
        (
            "circle",
            {
                "center_x": 400.0,
                "center_y": 300.0,
                "window_size": (800, 600),
                "label": "OK",
                "progress": 1.0,
                "pixel_size": 5.2,
                "fixed_text_scale": 1.28,
                "stage": "Video saved",
                "note": "Opening its location…",
            },
        ),
    ]


def test_saving_capture_status_uses_an_indeterminate_circle(monkeypatch):
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(size=(800, 600))
    window.UI_TEXT_SCALE = 1.28
    controller = window._ensure_recording_controller()
    controller.show_status(
        "Saving dive trace…",
        detail="Finishing the file. Keep CaveViewer open.",
        kind="info",
        duration=None,
        now=10.0,
    )
    calls = []
    window._render_recording_countdown_scrim = lambda *args, **kwargs: None
    window.import_progress_panel = SimpleNamespace(
        draw_circle_label=lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)

    window._render_capture_status_message((800, 600))

    assert calls == [
        {
            "center_x": 400.0,
            "center_y": 300.0,
            "window_size": (800, 600),
            "label": "...",
            "progress": None,
            "pixel_size": 3.8,
            "fixed_text_scale": 1.28,
            "stage": "Saving dive trace…",
            "note": "Finishing the file. Keep CaveViewer open.",
        }
    ]


def test_manual_trace_map_change_cancels_a_pending_countdown():
    window = _recording_window()
    window._manual_dive_trace = None
    controller = window._ensure_manual_dive_trace_controller()
    controller.start_countdown(now=10.0, start_number=3)

    assert window._stop_manual_dive_trace(reason="map_changed") is False
    assert not controller.countdown_active


@pytest.mark.parametrize(
    ("presentation_profile", "primary_modifiers"),
    [
        (select_presentation_profile(platform_name="unsupported"), SimpleNamespace(ctrl=True)),
        (select_presentation_profile(platform_name="darwin"), SimpleNamespace(command=True)),
    ],
)
def test_recording_hotkey_starts_or_stops_with_the_platform_primary_modifier(
    presentation_profile,
    primary_modifiers,
):
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = presentation_profile
    window._has_map_loaded = True
    window.wnd = SimpleNamespace(keys=SimpleNamespace(R=82))
    window._keys_down = set()
    window._key_resolve_cache = {}
    window._raw_command_modifier_down = lambda: False
    window._recording_is_armed = lambda: False
    calls = []
    window._toggle_recording = lambda: calls.append("toggle")

    assert window._handle_recording_hotkey(82, primary_modifiers) is True
    assert calls == ["toggle"]
    assert (
        window._handle_recording_hotkey(82, SimpleNamespace(shift=True)) is False
    )


def test_recorded_dive_space_hotkey_toggles_pause_only_while_active():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(keys=SimpleNamespace(SPACE=32))
    window._recorded_dive_controller = SimpleNamespace(active=True)
    calls = []
    window._toggle_recorded_dive_pause = lambda: calls.append("toggle") or True

    assert window._handle_recorded_dive_hotkey(32, None) is True
    assert calls == ["toggle"]
    assert window._handle_recorded_dive_hotkey(31, None) is False

    window._recorded_dive_controller.active = False
    assert window._handle_recorded_dive_hotkey(32, None) is False


def test_recorded_dive_readiness_requires_next_pose_chunks_on_gpu():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._initial_chunks_loaded = True
    window._initial_visual_ready = True
    pose = SimpleNamespace(position=(51.0, 2.0, 3.0))
    window._recorded_dive_controller = SimpleNamespace(
        candidate_elapsed=lambda **_kwargs: 1.0,
        trace=SimpleNamespace(pose_at=lambda _elapsed: pose),
    )
    world = SimpleNamespace(
        config=SimpleNamespace(load_radius_cells=3),
        cell_for_position=lambda _position: (1, 0, 0),
        available_cells_in_radius=lambda _center, _radius: frozenset({(1, 0, 0)}),
        loaded_cells={(1, 0, 0)},
        _failed_cells={},
        _lock=None,
    )
    window.world = world

    assert window._recorded_dive_chunks_ready(now=10.0) is True

    world.loaded_cells.clear()
    assert window._recorded_dive_chunks_ready(now=10.0) is False


def test_paused_recorded_dive_does_not_probe_chunks_until_resume():
    calls = []

    class PausedController:
        active = True
        state = viewer_window.recorded_dive.RecordedDivePlaybackState.PAUSED

        def update(self, camera, *, now, chunks_ready):
            calls.append((camera, now, chunks_ready))
            return self.state

    window = object.__new__(viewer_window.CaveViewerWindow)
    window.camera = object()
    window._recorded_dive_controller = PausedController()
    window._refresh_recorded_dive_prefetch = lambda: None
    window._recorded_dive_chunks_ready = lambda **_kwargs: pytest.fail(
        "paused inspection must defer chunk readiness until resume"
    )

    window._update_recorded_dive(now=10.0)

    assert calls == [(window.camera, 10.0, True)]


def test_recorded_dive_progress_reports_trace_time_and_loading_state():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._recorded_dive_controller = SimpleNamespace(
        active=True,
        state=viewer_window.recorded_dive.RecordedDivePlaybackState.BUFFERING,
        elapsed_s=65.0,
        trace=SimpleNamespace(duration_s=125.0),
    )
    calls = []
    window._render_dive_status_prompt = (
        lambda window_size, **kwargs: calls.append((window_size, kwargs))
    )

    assert window._render_recorded_dive_progress((800, 600)) is True
    assert calls == [
        (
            (800, 600),
            {
                "title": "Recorded Dive",
                "note": "Loading nearby cave chunks… 1:05 / 2:05",
            },
        )
    ]


def test_paused_recorded_dive_progress_explains_orientation_inspection():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._recorded_dive_controller = SimpleNamespace(
        active=True,
        state=viewer_window.recorded_dive.RecordedDivePlaybackState.PAUSED,
        elapsed_s=65.0,
        trace=SimpleNamespace(duration_s=125.0),
    )
    calls = []
    window._render_dive_status_prompt = (
        lambda window_size, **kwargs: calls.append((window_size, kwargs))
    )

    assert window._render_recorded_dive_progress((800, 600)) is True
    assert calls == [
        (
            (800, 600),
            {
                "title": "Recorded Dive",
                "note": (
                    "Paused for inspection at 1:05 / 2:05. "
                    "Look around; Space resumes."
                ),
            },
        )
    ]


def test_paused_recorded_dive_keyboard_input_allows_look_only():
    look_calls = []

    window = object.__new__(viewer_window.CaveViewerWindow)
    window.camera = SimpleNamespace(
        look=lambda yaw, pitch: look_calls.append((yaw, pitch)),
        barrel_roll=lambda _roll: pytest.fail("paused inspection must not roll"),
    )
    window._move_camera = lambda *_args: pytest.fail(
        "paused inspection must not move"
    )
    window._continuous_input_intent = lambda _dt: SimpleNamespace(
        has_motion=True,
        has_look=True,
        has_roll=True,
        yaw_delta=12.0,
        pitch_delta=-5.0,
    )

    window._handle_paused_recorded_dive_input(0.1)

    assert look_calls == [(12.0, -5.0)]


def test_paused_recorded_dive_mouse_look_does_not_cancel_playback():
    look_calls = []
    stopped = []

    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window._closing_requested = False
    window.color_picker = None
    window._option_look_active = lambda: False
    window._mouse_look_active = True
    window._last_mouse_pos = (0, 0)
    window._recorded_dive_controller = SimpleNamespace(
        active=True,
        state=viewer_window.recorded_dive.RecordedDivePlaybackState.PAUSED,
    )
    window._stop_recorded_dive = lambda *, reason: stopped.append(reason)
    window.camera = SimpleNamespace(
        look=lambda yaw, pitch: look_calls.append((yaw, pitch))
    )

    window._handle_mouse_look_motion(10, 20, 2, -3)

    assert stopped == []
    assert look_calls == [(2, -3)]


def test_paused_recorded_dive_ignores_speed_scroll_input():
    speed_changes = []

    window = object.__new__(viewer_window.CaveViewerWindow)
    window._recorded_dive_controller = SimpleNamespace(
        active=True,
        state=viewer_window.recorded_dive.RecordedDivePlaybackState.PAUSED,
    )
    window.camera = SimpleNamespace(
        adjust_speed=lambda amount: speed_changes.append(amount)
    )

    window.on_mouse_scroll_event(0.0, 2.0)

    assert speed_changes == []


def _fly_speed_event_window(*, waiting_for_begin: bool = False):
    speed_changes = []
    keys = SimpleNamespace(
        ACTION_PRESS=1,
        ACTION_RELEASE=0,
        ACTION_REPEAT=2,
        MINUS=45,
        EQUAL=61,
    )
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window._input_is_suppressed = lambda: False
    window.controls_overlay = SimpleNamespace(
        is_waiting_for_begin=waiting_for_begin,
        is_ready_to_begin=False,
    )
    window.wnd = SimpleNamespace(keys=keys)
    window._keys_down = set()
    window._recorded_dive_controller = None
    window.camera = SimpleNamespace(
        adjust_speed=lambda step: speed_changes.append(step),
        move_speed=4.0,
    )
    window._shift_is_down = lambda modifiers: bool(getattr(modifiers, "shift", False))
    window._handle_window_shortcut = lambda _key, _modifiers: False
    window._handle_recorded_dive_hotkey = lambda _key, _modifiers: False
    window._handle_bookmark_hotkey = lambda _key, _modifiers: False
    window._handle_manual_dive_trace_hotkey = lambda _key, _modifiers: False
    window._handle_recording_hotkey = lambda _key, _modifiers: False
    window._handle_reset_view_shortcut = lambda _key, _modifiers: False
    return window, keys, speed_changes


def test_fly_speed_hotkeys_adjust_base_speed_and_ignore_shift_without_motion_state():
    window, keys, speed_changes = _fly_speed_event_window()

    window.on_key_event(keys.MINUS, keys.ACTION_PRESS, SimpleNamespace())
    window.on_key_event(keys.EQUAL, keys.ACTION_REPEAT, SimpleNamespace())
    window.on_key_event(keys.EQUAL, keys.ACTION_PRESS, SimpleNamespace(shift=True))
    window.on_key_event(keys.EQUAL, keys.ACTION_RELEASE, SimpleNamespace())

    assert speed_changes == [-1, 1]
    assert window._keys_down == set()


def test_fly_speed_hotkeys_respect_startup_and_paused_dive_gates():
    window, keys, speed_changes = _fly_speed_event_window(
        waiting_for_begin=True
    )

    window.on_key_event(keys.MINUS, keys.ACTION_PRESS, SimpleNamespace())
    assert speed_changes == []

    window.controls_overlay.is_waiting_for_begin = False
    window._recorded_dive_controller = SimpleNamespace(
        active=True,
        state=viewer_window.recorded_dive.RecordedDivePlaybackState.PAUSED,
    )
    window.on_key_event(keys.EQUAL, keys.ACTION_PRESS, SimpleNamespace())

    assert speed_changes == []


def test_fly_speed_hotkey_does_not_stop_an_active_recorded_dive():
    window, keys, speed_changes = _fly_speed_event_window()
    window._recorded_dive_controller = SimpleNamespace(
        active=True,
        state=viewer_window.recorded_dive.RecordedDivePlaybackState.PLAYING,
    )
    stopped = []
    window._stop_recorded_dive = lambda *, reason: stopped.append(reason)

    window.on_key_event(keys.MINUS, keys.ACTION_PRESS, SimpleNamespace())

    assert speed_changes == [-1]
    assert stopped == []


def test_fly_speed_hotkey_is_safe_without_a_camera_or_when_input_is_suppressed():
    window, keys, speed_changes = _fly_speed_event_window()
    window.camera = None

    window.on_key_event(keys.MINUS, keys.ACTION_PRESS, SimpleNamespace())

    assert speed_changes == []

    window.camera = SimpleNamespace(
        adjust_speed=lambda step: speed_changes.append(step),
        move_speed=4.0,
    )
    window._input_is_suppressed = lambda: True
    window.on_key_event(keys.EQUAL, keys.ACTION_PRESS, SimpleNamespace())

    assert speed_changes == []


def test_manual_trace_samples_the_current_camera_pose():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.camera = _manual_trace_camera()
    recorder = FakeManualDiveTrace()
    window._manual_dive_trace = recorder
    window._manual_dive_trace_writers = []

    window._update_manual_dive_trace()
    window.camera.position[:] = (4.0, 5.0, 6.0)
    window._update_manual_dive_trace()

    assert [pose.position for pose in recorder.observed] == [
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    ]


def test_manual_trace_starts_in_the_explicit_map_root(tmp_path, monkeypatch):
    map_root = tmp_path / "Devils Eye"
    created = []

    class FakeRecorder:
        def __init__(self, output_dir, *, map_context):
            self.output_dir = output_dir
            self.map_context = map_context
            created.append(self)

        def start(self, _pose):
            return self.output_dir / "trace.jsonl"

    window = object.__new__(viewer_window.CaveViewerWindow)
    window._has_map_loaded = True
    window._manual_dive_trace = None
    window.camera = _manual_trace_camera()
    window.map_root = str(map_root)
    window.manifest = {
        "source_obj": "cave.obj",
        "guided_dive_identity": {
            "version": 1,
            "source_sha256": "a" * 64,
            "cache_manifest_sha256": "b" * 64,
        },
    }
    window._primary_shortcut_label = lambda: "Ctrl"
    monkeypatch.setattr(
        viewer_window.manual_dive_trace,
        "ManualDiveTraceRecorder",
        FakeRecorder,
    )

    assert window._start_manual_dive_trace() is True
    assert created[0].output_dir == map_root / "_guided_dives"
    assert window._manual_dive_trace is created[0]


def test_manual_trace_does_not_start_without_a_map_root():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._has_map_loaded = True
    window._manual_dive_trace = None
    window.camera = _manual_trace_camera()
    window.map_root = None

    assert window._start_manual_dive_trace() is False


def test_manual_trace_stop_is_nonblocking_and_keeps_writer_for_polling():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.camera = _manual_trace_camera()
    recorder = FakeManualDiveTrace()
    window._manual_dive_trace = recorder
    window._manual_dive_trace_writers = []

    assert window._stop_manual_dive_trace(reason="map_changed") is True

    assert window._manual_dive_trace is None
    pending_writer = window._manual_dive_trace_writers[0]
    assert pending_writer.recorder is recorder
    assert pending_writer.show_completion is False
    assert pending_writer.reveal_on_success is False
    assert recorder.stopped == [
        (
            ManualDivePose.from_camera(window.camera),
            "map_changed",
        )
    ]


def test_user_stopped_manual_trace_shows_persistent_shared_saving_status():
    window = _recording_window()
    window.camera = _manual_trace_camera()
    recorder = FakeManualDiveTrace()
    window._manual_dive_trace = recorder
    window._manual_dive_trace_writers = []

    assert window._stop_manual_dive_trace(reason="user_stopped") is True

    assert window._recording_status_message == "Saving dive trace…"
    assert window._recording_status_detail == (
        "Finishing the file. Press Esc to cancel. Keep CaveViewer open."
    )
    assert window._recording_status_until is None
    pending_writer = window._manual_dive_trace_writers[0]
    assert pending_writer.show_completion is True
    assert pending_writer.reveal_on_success is True


def test_completed_manual_trace_confirms_and_reveals_published_file(tmp_path):
    output_path = tmp_path / "trace.jsonl"
    output_path.write_text('{"record": "trace_completed"}\n', encoding="utf-8")
    revealed = []

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            revealed.append(path)

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )
    recorder = FakeManualDiveTrace()
    recorder.result = ManualDiveTraceResult(
        output_path=str(output_path),
        partial_path=str(tmp_path / ".trace.jsonl.part"),
        completed=True,
        error=None,
    )
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = [_pending_manual_trace_writer(recorder)]

    window._update_manual_dive_trace(now=10.0)

    assert window._manual_dive_trace_writers == []
    assert window._recording_status_message == "Dive trace saved"
    assert window._recording_status_detail == "Opening its location…"
    assert window._recording_status_kind == "success"
    assert window._recording_status_until == pytest.approx(13.0)
    assert revealed == []

    window._update_manual_dive_trace(now=12.99)
    assert revealed == []

    window._update_manual_dive_trace(now=13.0)
    assert revealed == [str(output_path)]


def test_background_trace_completion_is_silent_and_does_not_reveal(tmp_path):
    revealed = []

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            revealed.append(path)

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )
    pending = FakeManualDiveTrace()
    failed = FakeManualDiveTrace()
    failed.result = ManualDiveTraceResult(
        output_path=str(tmp_path / "trace.jsonl"),
        partial_path=str(tmp_path / ".trace.jsonl.part"),
        completed=False,
        error="disk full",
    )
    window._manual_dive_trace = None
    pending_writer = _pending_manual_trace_writer(
        pending,
        show_completion=False,
        reveal_on_success=False,
    )
    failed_writer = _pending_manual_trace_writer(
        failed,
        show_completion=False,
        reveal_on_success=False,
    )
    window._manual_dive_trace_writers = [pending_writer, failed_writer]

    window._update_manual_dive_trace()

    assert window._manual_dive_trace_writers == [pending_writer]
    assert window._recording_status_message is None
    assert revealed == []


def test_map_changed_trace_save_is_silent_after_successful_publication(tmp_path):
    revealed = []

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            revealed.append(path)

    output_path = tmp_path / "trace.jsonl"
    output_path.write_text('{"record": "trace_completed"}\n', encoding="utf-8")
    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )
    recorder = FakeManualDiveTrace()
    recorder.result = ManualDiveTraceResult(
        output_path=str(output_path),
        partial_path=str(tmp_path / ".trace.jsonl.part"),
        completed=True,
        error=None,
    )
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = [
        _pending_manual_trace_writer(
            recorder,
            show_completion=False,
            reveal_on_success=False,
        )
    ]

    window._update_manual_dive_trace(now=10.0)
    window._update_manual_dive_trace(now=20.0)

    assert window._recording_status_message is None
    assert revealed == []


def test_failed_user_stopped_trace_shows_shared_failure_without_reveal(tmp_path):
    revealed = []

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            revealed.append(path)

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )
    recorder = FakeManualDiveTrace()
    recorder.result = ManualDiveTraceResult(
        output_path=str(tmp_path / "trace.jsonl"),
        partial_path=str(tmp_path / ".trace.jsonl.part"),
        completed=False,
        error="disk full",
    )
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = [_pending_manual_trace_writer(recorder)]

    window._update_manual_dive_trace(now=10.0)

    assert window._recording_status_message == "Could not save dive trace"
    assert window._recording_status_detail == "disk full"
    assert window._recording_status_kind == "error"
    assert revealed == []


def test_manual_trace_reveal_failure_keeps_saved_status(tmp_path, monkeypatch):
    output_path = tmp_path / "trace.jsonl"
    output_path.write_text('{"record": "trace_completed"}\n', encoding="utf-8")
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)

    class FailingSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            raise RuntimeError(f"blocked: {path}")

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FailingSavedArtifactRevealAdapter()
    )
    recorder = FakeManualDiveTrace()
    recorder.result = ManualDiveTraceResult(
        output_path=str(output_path),
        partial_path=str(tmp_path / ".trace.jsonl.part"),
        completed=True,
        error=None,
    )
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = [_pending_manual_trace_writer(recorder)]

    window._update_manual_dive_trace(now=10.0)

    assert window._recording_status_message == "Dive trace saved"
    assert window._recording_status_kind == "success"

    window._update_manual_dive_trace(now=13.0)
    assert logger.warning_messages == [
        f"Could not reveal saved dive trace {output_path}: blocked: {output_path}"
    ]


def test_manual_trace_marks_bookmark_recall_as_discontinuity():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._has_map_loaded = True
    window.camera = _manual_trace_camera()
    window._bookmarks = {
        1: {
            "position": [10.0, 20.0, 30.0],
            "yaw": 0.5,
            "pitch": 0.25,
        }
    }
    window._navigation_position_is_allowed = lambda _position: True
    window.controls_overlay = SimpleNamespace(show_panel=lambda: None)
    recorder = FakeManualDiveTrace()
    window._manual_dive_trace = recorder

    assert window._recall_bookmark_slot(1) is True

    before, after, reason = recorder.discontinuities[0]
    assert before.position == (1.0, 2.0, 3.0)
    assert after.position == (10.0, 20.0, 30.0)
    assert reason == "bookmark_recall"


@pytest.mark.parametrize(
    "owner",
    (
        viewer_window.CaptureOwner.VIDEO,
        viewer_window.CaptureOwner.DIVE_TRACE,
        viewer_window.CaptureOwner.SLICE,
    ),
)
def test_active_capture_does_not_render_top_screen_prompt(owner):
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._active_capture_owner = lambda: owner
    window._primary_shortcut_label = lambda: "Ctrl"
    calls = []
    window._render_dive_status_prompt = (
        lambda window_size, **kwargs: calls.append((window_size, kwargs))
    )

    assert window._render_active_capture_instruction((800, 600)) is False

    assert calls == []


def test_desktop_relative_window_size_uses_eighty_percent_per_axis(monkeypatch):
    class FakeRoot:
        def __init__(self):
            self.withdrawn = False
            self.destroyed = False

        def withdraw(self):
            self.withdrawn = True

        def winfo_screenwidth(self):
            return 1920

        def winfo_screenheight(self):
            return 1080

        def destroy(self):
            self.destroyed = True

    root = FakeRoot()
    root_options = []
    monkeypatch.setitem(
        sys.modules,
        "tkinter",
        SimpleNamespace(Tk=lambda **options: root_options.append(options) or root),
    )

    assert viewer_window._desktop_relative_window_size() == (1536, 864)
    assert root_options == [tk_root_options()]
    assert root.withdrawn is True
    assert root.destroyed is True


def test_desktop_relative_window_size_reuses_existing_tk_root(monkeypatch):
    class ExistingRoot:
        def __init__(self):
            self.destroyed = False
            self.withdrawn = False

        def winfo_exists(self):
            return True

        def winfo_screenwidth(self):
            return 2560

        def winfo_screenheight(self):
            return 1440

        def withdraw(self):
            self.withdrawn = True

        def destroy(self):
            self.destroyed = True

    root = ExistingRoot()
    monkeypatch.setitem(
        sys.modules,
        "tkinter",
        SimpleNamespace(
            _default_root=root,
            Tk=lambda **_options: (_ for _ in ()).throw(
                AssertionError("must not create a second Tk root")
            ),
        ),
    )

    assert viewer_window._desktop_relative_window_size() == (2048, 1152)
    assert root.withdrawn is False
    assert root.destroyed is False


def test_desktop_relative_window_size_uses_default_for_bad_existing_root():
    root = SimpleNamespace(
        winfo_screenwidth=lambda: 0,
        winfo_screenheight=lambda: 1440,
    )

    assert viewer_window._desktop_relative_window_size(root) == (1600, 1000)


def test_desktop_relative_window_size_does_not_replace_bad_live_default_root(
    monkeypatch,
):
    root = SimpleNamespace(
        winfo_exists=lambda: True,
        winfo_screenwidth=lambda: 0,
        winfo_screenheight=lambda: 1440,
    )
    monkeypatch.setitem(
        sys.modules,
        "tkinter",
        SimpleNamespace(
            _default_root=root,
            Tk=lambda **_options: (_ for _ in ()).throw(
                AssertionError("must not create a second Tk root")
            ),
        ),
    )

    assert viewer_window._desktop_relative_window_size() == (1600, 1000)


def test_window_pixel_ratio_uses_framebuffer_size():
    window = SimpleNamespace(size=(1000, 700), buffer_size=(2000, 1400))

    assert viewer_window._window_pixel_ratio(window) == 2.0


def test_window_pixel_ratio_falls_back_for_missing_backend_data():
    assert viewer_window._window_pixel_ratio(SimpleNamespace(size=(1000, 700))) == 1.0


def test_viewer_ui_surface_size_prefers_framebuffer_size_for_scaled_dpi():
    window = SimpleNamespace(size=(1600, 1000), buffer_size=(2048, 1280))

    assert viewer_window._viewer_ui_surface_size(window) == (2048, 1280)


def test_viewer_ui_scale_grows_on_large_viewer_surfaces():
    assert viewer_window._viewer_ui_scale_for_window_size((1536, 864), {}) == 1.0
    assert viewer_window._viewer_ui_scale_for_window_size((2048, 1152), {}) == pytest.approx(
        4 / 3
    )
    assert viewer_window._viewer_ui_scale_for_window_size((3840, 2160), {}) == 1.45


def test_viewer_ui_scale_env_override_is_developer_only_escape_hatch():
    assert viewer_window._viewer_ui_scale_for_window_size(
        (1536, 864), {"CAVEVIEWER_VIEWER_UI_SCALE": "1.25"}
    ) == 1.25
    assert viewer_window._viewer_ui_scale_for_window_size(
        (1536, 864), {"CAVEVIEWER_VIEWER_UI_SCALE": "bad"}
    ) == 1.0


def test_viewer_overlay_text_scale_uses_platform_default():
    assert viewer_window._viewer_overlay_text_scale(
        select_presentation_profile(platform_name="unsupported"), 1.28, {}
    ) == 1.28
    assert viewer_window._viewer_overlay_text_scale(
        select_presentation_profile(platform_name="darwin"), 1.28, {}
    ) == pytest.approx(1.472)


def test_viewer_overlay_text_scale_env_override_still_wins():
    assert viewer_window._viewer_overlay_text_scale(
        select_presentation_profile(platform_name="darwin"),
        1.28,
        {"CAVEVIEWER_UI_TEXT_SCALE": "1.1"},
    ) == 1.1
    assert viewer_window._viewer_overlay_text_scale(
        select_presentation_profile(platform_name="darwin"),
        1.28,
        {"CAVEVIEWER_UI_TEXT_SCALE": "bad"},
    ) == pytest.approx(1.472)


def test_optional_ms_formatter_reports_disabled_timer():
    assert viewer_window.CaveViewerWindow._format_optional_ms(None) == "n/a"
    assert viewer_window.CaveViewerWindow._format_optional_ms(9.34) == "9.3ms"














def test_map_initial_camera_ignores_navigation_start_metadata():
    manifest = {
        "chunks": {
            "first": {
                "bounds_min": [-10.0, -4.0, 2.0],
                "bounds_max": [10.0, 4.0, 6.0],
            }
        },
        "navigation": {
            "routes": [{"certified_start_position": [100.5, 20.5, -30.5]}]
        },
    }

    position = viewer_window._map_initial_camera_position(manifest)

    assert np.allclose(position, [0.0, 0.0, 4.0])




def test_map_initial_camera_uses_first_manifest_chunk_bounds_center():
    manifest = {
        "chunks": {
            "first": {
                "bounds_min": [-10.0, -4.0, 2.0],
                "bounds_max": [10.0, 4.0, 6.0],
            }
        }
    }

    position = viewer_window._map_initial_camera_position(manifest)

    assert np.allclose(position, [0.0, 0.0, 4.0])










































def test_recording_countdown_hides_picker_and_manual_help(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 40.0)
    window = _recording_window()
    window._has_map_loaded = True
    window._recording_output_dir = str(tmp_path)
    window._resolve_ffmpeg_path = lambda: "/usr/bin/ffmpeg"
    window.color_picker = SimpleNamespace(hide=lambda: calls.append("hide_picker"))
    window.controls_overlay = SimpleNamespace(
        is_manual_mode=True,
        hide_help=lambda: calls.append("hide_help"),
    )

    window._start_recording_countdown()

    assert calls == ["hide_picker", "hide_help"]
    assert window._recording_countdown_started_at == 40.0
    assert window._recording_countdown_until == 44.0


def test_recording_countdown_reports_missing_encoder_before_hiding_ui(monkeypatch):
    calls = []
    window = _recording_window()
    window._has_map_loaded = True
    window._resolve_ffmpeg_path = lambda: None
    window.color_picker = SimpleNamespace(hide=lambda: calls.append("hide_picker"))
    window.controls_overlay = SimpleNamespace(
        is_manual_mode=True,
        hide_help=lambda: calls.append("hide_help"),
    )

    window._start_recording_countdown()

    assert calls == []
    assert window._recording_countdown_until is None
    assert window._recording_status_message == "Recording unavailable"
    assert window._recording_status_detail == (
        "Video recording requires ffmpeg. Install it or set CAVEVIEWER_FFMPEG."
    )


def test_recording_toggle_cancels_existing_countdown(monkeypatch):
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)
    window = _recording_window()
    window._recording_countdown_started_at = 7.0
    window._recording_countdown_until = 11.0

    window._toggle_recording()

    assert window._recording_countdown_started_at is None
    assert window._recording_countdown_until is None
    assert window._recording_status_message == "Video canceled"
    assert window._recording_status_kind == "cancel"
    assert window._recording_status_until == pytest.approx(13.0)


def test_recording_signal_writer_stop_replaces_full_frame_with_sentinel():
    window = _recording_window()
    frame_queue = queue.Queue(maxsize=1)
    frame_queue.put_nowait(b"old-frame")

    window._recording_signal_writer_stop(frame_queue)

    assert frame_queue.get_nowait() is None


def test_recording_enqueue_frame_reports_encoder_backpressure_once(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    window = _recording_window()
    window._recording_frame_queue = queue.Queue(maxsize=1)
    window._recording_frame_queue.put_nowait(b"queued-frame")

    assert window._recording_enqueue_frame(b"new-frame") is False
    assert window._recording_enqueue_frame(b"newer-frame") is False

    assert window._recording_dropped_frames == 2
    assert logger.warning_messages == [
        "Recording encoder is falling behind; dropping video frames."
    ]


def test_start_recording_encoder_sends_output_size_to_ffmpeg_without_scale_filter(
    monkeypatch,
    tmp_path,
):
    popen_calls = []
    created_threads = []

    class FakeProcess:
        stdin = SimpleNamespace(write=lambda _frame: None, close=lambda: None)
        stderr = SimpleNamespace(readline=lambda: b"")
        returncode = None

    class FakeBuffer:
        def release(self):
            pass

    class FakeThread:
        def __init__(self, *, target, args=(), daemon=None, name=None):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.name = name
            self.started = False
            created_threads.append(self)

        def start(self):
            self.started = True

    class FakeCtx:
        viewport = (0, 0, 1280, 720)
        screen = SimpleNamespace(viewport=(0, 0, 1280, 720), size=(1280, 720))

        def simple_framebuffer(self, *_args, **_kwargs):
            raise AssertionError("direct-size recording should not allocate a readback framebuffer")

        def __init__(self):
            self.buffer_reserves = []

        def buffer(self, *, reserve):
            self.buffer_reserves.append(reserve)
            return FakeBuffer()

    monkeypatch.setattr(
        recording.subprocess,
        "Popen",
        lambda cmd, **kwargs: popen_calls.append((cmd, kwargs)) or FakeProcess(),
    )
    monkeypatch.setattr(recording.threading, "Thread", FakeThread)
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 100.0)

    window = _recording_window()
    ctx = FakeCtx()
    window.ctx = ctx
    window._recording_output_dir = str(tmp_path)
    window._recording_max_height = 1080
    window._recording_fps = 30
    window._recording_crf = 23
    window._resolve_ffmpeg_path = lambda: "/usr/bin/ffmpeg"

    assert window._start_recording_encoder() is True

    cmd, popen_kwargs = popen_calls[0]
    assert cmd[cmd.index("-s") + 1] == "1280x720"
    assert cmd[cmd.index("-pix_fmt") + 1] == "rgb24"
    assert cmd[cmd.index("-vf") + 1] == "vflip"
    assert not any("scale=" in part for part in cmd)
    assert popen_kwargs["stdin"] is recording.subprocess.PIPE
    assert window._recording_session is not None
    assert window._recording_frame_queue is window._recording_session.frame_queue
    assert window._recording_size == (1280, 720)
    assert window._recording_readback_framebuffer is None
    assert ctx.buffer_reserves == [1280 * 720 * 3] * 3
    assert len(window._recording_readback_slots) == 3
    assert window._recording_readback_byte_count == 1280 * 720 * 3
    assert [thread.started for thread in created_threads] == [True, True]


def test_start_recording_encoder_allocates_output_sized_readback_framebuffer(
    monkeypatch,
    tmp_path,
):
    popen_calls = []

    class FakeFramebuffer:
        def __init__(self, size):
            self.size = size
            self.viewport = None

        def release(self):
            raise AssertionError("new recording framebuffer should stay owned after start")

    class FakeBuffer:
        def release(self):
            pass

    class FakeProcess:
        stdin = SimpleNamespace(write=lambda _frame: None, close=lambda: None)
        stderr = SimpleNamespace(readline=lambda: b"")
        returncode = None

    class FakeThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    class FakeCtx:
        viewport = (0, 0, 4000, 2000)
        screen = SimpleNamespace(viewport=(0, 0, 4000, 2000), size=(4000, 2000))

        def __init__(self):
            self.simple_framebuffer_calls = []
            self.buffer_reserves = []
            self.framebuffer = FakeFramebuffer((2000, 1000))

        def simple_framebuffer(self, size, components=4):
            self.simple_framebuffer_calls.append((size, components))
            return self.framebuffer

        def buffer(self, *, reserve):
            self.buffer_reserves.append(reserve)
            return FakeBuffer()

    monkeypatch.setattr(
        recording.subprocess,
        "Popen",
        lambda cmd, **kwargs: popen_calls.append((cmd, kwargs)) or FakeProcess(),
    )
    monkeypatch.setattr(recording.threading, "Thread", FakeThread)
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 100.0)

    ctx = FakeCtx()
    window = _recording_window()
    window.ctx = ctx
    window._recording_output_dir = str(tmp_path)
    window._recording_max_height = 1000
    window._recording_fps = 30
    window._recording_crf = 23
    window._resolve_ffmpeg_path = lambda: "/usr/bin/ffmpeg"

    assert window._start_recording_encoder() is True

    cmd, _popen_kwargs = popen_calls[0]
    assert ctx.simple_framebuffer_calls == [((2000, 1000), 4)]
    assert ctx.buffer_reserves == [2000 * 1000 * 3] * 3
    assert ctx.framebuffer.viewport == (0, 0, 2000, 1000)
    assert cmd[cmd.index("-s") + 1] == "2000x1000"
    assert cmd[cmd.index("-pix_fmt") + 1] == "rgb24"
    assert cmd[cmd.index("-vf") + 1] == "vflip"
    assert not any("scale=" in part for part in cmd)
    assert window._recording_size == (2000, 1000)
    assert window._recording_viewport == (0, 0, 4000, 2000)
    assert window._recording_readback_framebuffer is ctx.framebuffer
    assert len(window._recording_readback_slots) == 3
    assert window._recording_readback_byte_count == 2000 * 1000 * 3


def test_start_recording_encoder_rechecks_gate_before_starting_ffmpeg(monkeypatch):
    class FakeBuffer:
        def release(self):
            pass

    class FakeCtx:
        viewport = (0, 0, 1280, 720)
        screen = SimpleNamespace(viewport=(0, 0, 1280, 720), size=(1280, 720))

        def buffer(self, *, reserve):
            return FakeBuffer()

    available = CapabilityResult.available(
        viewer_window.VideoRecordingTarget("/usr/bin/ffmpeg", "/recordings"),
        reason_code="video_recording_target_available",
    )
    unavailable = CapabilityResult.unavailable(
        reason_code="video_recording_output_directory_unavailable",
    )
    preflights = iter(
        (
            VideoRecordingPreflight(
                capability=available,
                decision=decide_video_recording(available),
            ),
            VideoRecordingPreflight(
                capability=unavailable,
                decision=decide_video_recording(unavailable),
            ),
        )
    )
    window = _recording_window()
    window.ctx = FakeCtx()
    window._recording_max_height = 1080
    window._recording_fps = 30
    window._recording_crf = 23
    window._recording_preflight = lambda: next(preflights)
    monkeypatch.setattr(
        recording,
        "start_encoder_session",
        lambda **_kwargs: pytest.fail("ffmpeg must not start after a failed recheck"),
    )

    assert window._start_recording_encoder() is False

    assert window._recording_status_message == "Recording unavailable"
    assert window._recording_status_detail == (
        "Video recording cannot save to the selected folder."
    )
    assert window._recording_readback_slots == []


def test_start_recording_encoder_uses_runtime_process_adapter(monkeypatch, tmp_path):
    startup_kwargs = {"creationflags": 17}
    process_adapter_calls = []
    encoder_calls = []
    target = viewer_window.VideoRecordingTarget("/usr/bin/ffmpeg", str(tmp_path))

    class FakeRecordingProcessAdapter:
        def encoder_popen_kwargs(self):
            process_adapter_calls.append(True)
            return startup_kwargs

    def start_encoder_session(**kwargs):
        encoder_calls.append(kwargs)
        return SimpleNamespace(
            frame_queue=queue.Queue(),
            output_path=kwargs["output_path"],
            output_size=kwargs["output_size"],
            viewport=kwargs["viewport"],
        )

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        recording_process_adapter=FakeRecordingProcessAdapter()
    )
    window._recording_target_if_available = lambda: target
    window._recording_capture_viewport = lambda: (0, 0, 2, 2)
    window._recording_output_size = lambda _width, _height: (2, 2)
    window._create_recording_readback_framebuffer = lambda *_args: None
    window._create_recording_readback_buffers = lambda *_args: None
    window._recording_fps = 30
    window._recording_crf = 23
    monkeypatch.setattr(recording, "start_encoder_session", start_encoder_session)

    assert window._start_recording_encoder() is True
    assert process_adapter_calls == [True]
    assert encoder_calls[0]["popen_startup_kwargs"] == startup_kwargs


def test_recording_skips_framebuffer_read_when_writer_queue_is_full(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)

    class FakeScreen:
        def __init__(self):
            self.read_calls = []

        def read(self, **kwargs):
            self.read_calls.append(kwargs)
            return b"x" * 12

    screen = FakeScreen()
    window = _recording_window()
    window.ctx = SimpleNamespace(viewport=(0, 0, 2, 2), screen=screen)
    window._recording_size = (2, 2)
    window._recording_viewport = (0, 0, 2, 2)
    window._recording_next_frame_time = 10.0
    window._recording_frame_interval = 1.0
    frame_queue = queue.Queue(maxsize=1)
    frame_queue.put_nowait(b"queued-frame")
    window._recording_session = _active_recording_session(frame_queue=frame_queue)
    window._recording_frame_queue = frame_queue

    read_ms = window._recording_update_after_scene(12.2)

    assert read_ms == 0.0
    assert screen.read_calls == []
    assert window._recording_dropped_frames == 3
    assert window._recording_next_frame_time == pytest.approx(13.0)
    assert logger.warning_messages == [
        "Recording encoder is falling behind; dropping video frames."
    ]


def test_recording_stages_scaled_framebuffer_when_output_size_differs():
    class FakeScreen:
        def __init__(self):
            self.viewport = (9, 8, 7, 6)
            self.read_into_calls = []

        def read_into(self, *args, **kwargs):
            self.read_into_calls.append((args, kwargs))

    class FakeFramebuffer:
        def __init__(self):
            self.viewport = (1, 2, 3, 4)
            self.read_into_calls = []

        def read_into(self, *args, **kwargs):
            self.read_into_calls.append((args, kwargs))

    class FakeBuffer:
        pass

    class FakeCtx:
        def __init__(self, screen):
            self.screen = screen
            self.copy_framebuffer_calls = []

        def copy_framebuffer(self, dst, src):
            self.copy_framebuffer_calls.append(
                (dst, src, src.viewport, dst.viewport)
            )

    screen = FakeScreen()
    readback_framebuffer = FakeFramebuffer()
    buffer = FakeBuffer()
    window = _recording_window()
    window.ctx = FakeCtx(screen)
    window._recording_size = (2, 2)
    window._recording_viewport = (0, 0, 4, 4)
    window._recording_readback_framebuffer = readback_framebuffer
    window._recording_readback_slots = [
        viewer_window._RecordingReadbackSlot(buffer)
    ]
    window._recording_readback_byte_count = 12

    assert window._recording_stage_frame() is True

    assert screen.read_into_calls == []
    assert window.ctx.copy_framebuffer_calls == [
        (
            readback_framebuffer,
            screen,
            (0, 0, 4, 4),
            (0, 0, 2, 2),
        )
    ]
    assert readback_framebuffer.read_into_calls == [
        (
            (buffer,),
            {"viewport": (0, 0, 2, 2), "components": 3, "alignment": 1},
        )
    ]
    assert screen.viewport == (9, 8, 7, 6)
    assert readback_framebuffer.viewport == (1, 2, 3, 4)
    assert window._recording_readback_slots[0].in_flight is True
    assert window._recording_readback_pending == window._recording_readback_slots


def test_recording_stage_frame_uses_direct_render_callback_when_available():
    render_calls = []

    class FakeFramebuffer:
        def __init__(self):
            self.read_into_calls = []

        def read_into(self, *args, **kwargs):
            self.read_into_calls.append((args, kwargs))

    class FakeBuffer:
        pass

    class FakeCtx:
        screen = SimpleNamespace()

        def copy_framebuffer(self, *_args):
            raise AssertionError("direct recording render should bypass framebuffer copy")

    readback_framebuffer = FakeFramebuffer()
    buffer = FakeBuffer()
    window = _recording_window()
    window.ctx = FakeCtx()
    window._recording_size = (2, 2)
    window._recording_viewport = (0, 0, 4, 4)
    window._recording_readback_framebuffer = readback_framebuffer
    window._recording_readback_slots = [
        viewer_window._RecordingReadbackSlot(buffer)
    ]
    window._recording_readback_byte_count = 12

    assert window._recording_stage_frame(
        render_frame=lambda framebuffer, size: render_calls.append(
            (framebuffer, size)
        )
    ) is True

    assert render_calls == [(readback_framebuffer, (2, 2))]
    assert readback_framebuffer.read_into_calls == [
        (
            (buffer,),
            {"viewport": (0, 0, 2, 2), "components": 3, "alignment": 1},
        )
    ]
    assert window._recording_readback_slots[0].in_flight is True
    assert window._recording_readback_pending == window._recording_readback_slots


def test_recording_drains_oldest_staged_frame_when_ring_is_full(monkeypatch):
    ticks = iter([70.0, 70.004])
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: next(ticks))

    class FakeBuffer:
        def __init__(self, data):
            self.data = data
            self.read_calls = []

        def read(self, *, size):
            self.read_calls.append(size)
            return self.data

    buffers = [FakeBuffer(bytes((index,)) * 12) for index in range(3)]
    slots = [
        viewer_window._RecordingReadbackSlot(buffer, in_flight=True)
        for buffer in buffers
    ]
    window = _recording_window()
    window._recording_size = (2, 2)
    window._recording_viewport = (0, 0, 2, 2)
    window._recording_readback_slots = slots
    window._recording_readback_pending = slots.copy()
    window._recording_readback_byte_count = 12
    window._recording_frame_queue = queue.Queue(maxsize=3)

    read_ms = window._recording_drain_staged_frames()

    assert read_ms == pytest.approx(4.0)
    assert buffers[0].read_calls == [12]
    assert window._recording_frame_queue.get_nowait() == b"\x00" * 12
    assert slots[0].in_flight is False
    assert window._recording_readback_pending == slots[1:]


def test_recording_late_capture_drops_frames_instead_of_enqueuing_duplicates(
    monkeypatch,
):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    ticks = iter([50.0, 50.006])
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: next(ticks))

    class FakeScreen:
        def __init__(self):
            self.read_into_calls = []

        def read_into(self, *args, **kwargs):
            self.read_into_calls.append((args, kwargs))

    class FakeBuffer:
        pass

    screen = FakeScreen()
    buffer = FakeBuffer()
    window = _recording_window()
    window.ctx = SimpleNamespace(viewport=(0, 0, 2, 2), screen=screen)
    window._recording_size = (2, 2)
    window._recording_viewport = (0, 0, 2, 2)
    window._recording_readback_slots = [
        viewer_window._RecordingReadbackSlot(buffer)
    ]
    window._recording_readback_byte_count = 12
    window._recording_next_frame_time = 10.0
    window._recording_frame_interval = 1.0
    frame_queue = queue.Queue(maxsize=5)
    window._recording_session = _active_recording_session(frame_queue=frame_queue)
    window._recording_frame_queue = frame_queue

    read_ms = window._recording_update_after_scene(12.2)

    assert read_ms == pytest.approx(6.0)
    assert screen.read_into_calls == [
        (
            (buffer,),
            {"viewport": (0, 0, 2, 2), "components": 3, "alignment": 1},
        )
    ]
    assert window._recording_frame_queue.qsize() == 0
    assert window._recording_readback_pending == window._recording_readback_slots
    assert window._recording_readback_slots[0].in_flight is True
    assert window._recording_dropped_frames == 2
    assert window._recording_next_frame_time == pytest.approx(13.0)
    assert logger.warning_messages == [
        "Recording encoder is falling behind; dropping video frames."
    ]


def test_frame_spike_log_reports_recording_read_time():
    source = inspect.getsource(viewer_window.CaveViewerWindow._render_interactive_frame)

    assert "recording_read=" in source
    assert "recording_stage=" in source
    assert "recording_drain=" in source


def test_render_loop_uses_nonblocking_throttle_instead_of_sleep():
    source = inspect.getsource(viewer_window.CaveViewerWindow.on_render)

    assert "time.sleep(" not in source
    assert "frame_scheduler.is_due(" in source
    assert "_render_interactive_frame(current_time, frame_time)" in source


def test_viewer_window_delegates_recording_encoder_ownership():
    source = inspect.getsource(viewer_window)

    assert "import subprocess" not in source
    assert "subprocess.Popen" not in source
    assert "target=self._recording_writer_loop" not in source
    assert "target=self._recording_stderr_reader" not in source


def test_recording_success_confirms_before_revealing_saved_file(monkeypatch):
    revealed = []
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            revealed.append(path)

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )

    window._apply_recording_stop_result(
        recording.RecordingStopResult(
            output_path="/recordings/cave.mp4",
            returncode=0,
            stderr_text="",
            writer_error=None,
            dropped_frames=0,
            show_message=True,
            reveal_on_success=True,
        )
    )

    assert window._recording_status_message == "Video saved"
    assert window._recording_status_detail == "Opening its location…"
    assert window._recording_status_kind == "success"
    assert window._recording_status_until == pytest.approx(13.0)
    assert revealed == []

    window._drain_due_saved_artifact_reveals(now=12.99)
    assert revealed == []

    window._drain_due_saved_artifact_reveals(now=13.0)
    assert revealed == ["/recordings/cave.mp4"]


def test_recording_success_uses_injected_runtime_reveal_adapter(monkeypatch):
    revealed = []
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            revealed.append(path)

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )

    window._apply_recording_stop_result(
        recording.RecordingStopResult(
            output_path="/recordings/cave.mp4",
            returncode=0,
            stderr_text="",
            writer_error=None,
            dropped_frames=0,
            show_message=True,
            reveal_on_success=True,
        )
    )

    assert window._recording_status_message == "Video saved"
    assert revealed == []
    window._drain_due_saved_artifact_reveals(now=13.0)
    assert revealed == ["/recordings/cave.mp4"]


def test_recording_success_does_not_reveal_after_background_stop():
    revealed = []

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            revealed.append(path)

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )

    window._apply_recording_stop_result(
        recording.RecordingStopResult(
            output_path="/recordings/cave.mp4",
            returncode=0,
            stderr_text="",
            writer_error=None,
            dropped_frames=0,
            show_message=False,
        )
    )

    assert window._recording_status_message is None
    assert revealed == []


def test_exit_finalization_keeps_its_video_status_and_does_not_reveal_files(
    monkeypatch,
):
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)
    window = _recording_window()
    _begin_exit_capture_finalization(window)
    window._show_capture_status(
        "Finishing video",
        "Saving the last frames. CaveViewer will close automatically.",
        duration=None,
        now=10.0,
    )

    window._apply_recording_stop_result(
        recording.RecordingStopResult(
            output_path="/recordings/cave.mp4",
            returncode=0,
            stderr_text="",
            writer_error=None,
            dropped_frames=0,
            show_message=True,
            reveal_on_success=True,
        )
    )

    assert window._recording_status_message == "Finishing video"
    assert window._recording_status_detail == (
        "Saving the last frames. CaveViewer will close automatically."
    )
    assert window._ensure_artifact_capture_presentation().take_due_reveals(now=20.0) == ()


def test_interrupted_recording_success_can_confirm_without_revealing():
    revealed = []

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            revealed.append(path)

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )

    window._apply_recording_stop_result(
        recording.RecordingStopResult(
            output_path="/recordings/cave.mp4",
            returncode=0,
            stderr_text="",
            writer_error=None,
            dropped_frames=0,
            show_message=True,
            reveal_on_success=False,
        )
    )

    assert window._recording_status_message == "Video saved"
    assert window._recording_status_detail is None
    window._drain_due_saved_artifact_reveals(now=999.0)
    assert revealed == []


def test_recording_reveal_failure_keeps_saved_status(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)

    class FakeSavedArtifactRevealAdapter:
        def reveal_saved_artifact(self, path):
            raise RuntimeError(f"blocked: {path}")

    window = _recording_window()
    window._platform_runtime = SimpleNamespace(
        saved_artifact_reveal_adapter=FakeSavedArtifactRevealAdapter()
    )

    window._apply_recording_stop_result(
        recording.RecordingStopResult(
            output_path="/recordings/cave.mp4",
            returncode=0,
            stderr_text="",
            writer_error=None,
            dropped_frames=0,
            show_message=True,
            reveal_on_success=True,
        )
    )

    assert window._recording_status_message == "Video saved"
    assert window._recording_status_kind == "success"
    assert logger.warning_messages == []

    window._drain_due_saved_artifact_reveals(now=13.0)
    assert logger.warning_messages == [
        "Could not reveal saved video /recordings/cave.mp4: "
        "blocked: /recordings/cave.mp4"
    ]


def test_escape_canceled_recording_releases_buffers_and_removes_mp4(tmp_path):
    output_path = tmp_path / "partial.mp4"
    output_path.write_bytes(b"partial recording")

    class FakeProcess:
        stdin = None
        returncode = 0

        @staticmethod
        def wait(timeout=None):
            return 0

    class FakeFramebuffer:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    class FakeBuffer:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    framebuffer = FakeFramebuffer()
    buffer = FakeBuffer()
    slot = viewer_window._RecordingReadbackSlot(buffer, in_flight=True)
    window = _recording_window()
    session = _active_recording_session(
        process=FakeProcess(),
        output_path=str(output_path),
    )
    window._recording_session = session
    window._recording_output_path = str(output_path)
    window._recording_size = (2, 2)
    window._recording_viewport = (0, 0, 2, 2)
    window._recording_readback_framebuffer = framebuffer
    window._recording_readback_slots = [slot]
    window._recording_readback_pending = [slot]
    window._recording_readback_byte_count = 12
    window._recording_frame_queue = session.frame_queue
    window.wnd = SimpleNamespace(is_closing=True)
    reset_reasons = []
    window._reset_transient_input_state = reset_reasons.append

    assert window._begin_escape_capture_cancellation() is True

    assert framebuffer.released is True
    assert buffer.released is True
    assert window._recording_readback_slots == []
    assert window._recording_frame_queue is None
    assert window.wnd.is_closing is False
    assert reset_reasons == ["canceling capture before close"]
    assert window._escape_capture_cancellation_active()
    assert not window._exit_capture_finalization_active()
    assert window._recording_status_message == "Canceling video…"
    assert "Finishing" not in window._recording_status_message
    window._recording_stop_thread.join(timeout=2.0)
    assert not window._recording_stop_thread.is_alive()
    window._drain_recording_stop_results()

    assert not output_path.exists()
    assert window._recording_status_message == "Video canceled"
    assert window._recording_status_detail == "No video was saved."
    assert window._recording_status_kind == "cancel"
    assert window._recording_status_until is not None
    assert window._recording_stop_thread is None


def test_stop_recording_kills_encoder_after_timeout_and_reports_failure(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 20.0)

    class TimeoutProcess:
        stdin = None
        returncode = None

        def __init__(self):
            self.killed = False
            self.wait_calls = []

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                raise recording.subprocess.TimeoutExpired("ffmpeg", timeout)
            self.returncode = -9

        def kill(self):
            self.killed = True

    class FakeFramebuffer:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    class FakeBuffer:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    process = TimeoutProcess()
    framebuffer = FakeFramebuffer()
    buffer = FakeBuffer()
    slot = viewer_window._RecordingReadbackSlot(buffer, in_flight=True)
    window = _recording_window()
    session = _active_recording_session(process=process)
    session.append_stderr("No space left on device")
    window._recording_session = session
    window._recording_output_path = "/recordings/cave.mp4"
    window._recording_size = (640, 480)
    window._recording_viewport = (0, 0, 640, 480)
    window._recording_readback_framebuffer = framebuffer
    window._recording_readback_slots = [slot]
    window._recording_readback_pending = [slot]
    window._recording_readback_byte_count = 640 * 480 * 3
    window._recording_next_frame_time = 20.0
    window._recording_frame_queue = session.frame_queue

    window._stop_recording(show_message=True, reveal_on_success=True)

    assert window._recording_status_message == "Saving video…"
    assert window._recording_status_detail == (
        "Finishing the file. Press Esc to cancel. Keep CaveViewer open."
    )
    assert window._recording_stop_thread is not None
    window._recording_stop_thread.join(timeout=1.0)
    assert not window._recording_stop_thread.is_alive()
    window._drain_recording_stop_results()

    assert process.killed is True
    assert process.wait_calls == [8.0, None]
    assert framebuffer.released is True
    assert buffer.released is True
    assert slot.in_flight is False
    assert window._recording_session is None
    assert window._recording_output_path is None
    assert window._recording_readback_framebuffer is None
    assert window._recording_readback_slots == []
    assert window._recording_readback_pending == []
    assert window._recording_readback_byte_count == 0
    assert window._recording_frame_queue is None
    assert window._recording_status_message == "Could not save video"
    assert window._recording_status_detail == "Disk may be full"
    assert window._recording_status_kind == "error"
    assert window._recording_stop_thread is None
    assert any("Recording encoder exited with code -9" in message for message in logger.warning_messages)


def test_window_shortcut_leaves_control_w_unhandled():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = select_presentation_profile(
        platform_name="unsupported"
    )
    window.wnd = SimpleNamespace(keys=SimpleNamespace(W=87, O=79))
    window._keys_down = set()
    window._key_resolve_cache = {}
    closed = []
    window.on_close = lambda: closed.append("closed")

    assert window._handle_window_shortcut(87, SimpleNamespace(ctrl=True)) is False
    assert closed == []


def test_window_shortcut_opens_map_only_when_loaded():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = select_presentation_profile(
        platform_name="unsupported"
    )
    window.wnd = SimpleNamespace(keys=SimpleNamespace(W=87, O=79))
    window._keys_down = set()
    window._key_resolve_cache = {}
    calls = []
    window._handle_open_button_click = lambda: calls.append("open")

    window._has_map_loaded = False
    window._import_active = False
    assert window._handle_window_shortcut(79, SimpleNamespace(ctrl=True)) is True
    assert calls == []

    window._has_map_loaded = True
    window._import_active = False
    assert window._handle_window_shortcut(79, SimpleNamespace(ctrl=True)) is True
    assert calls == ["open"]


def test_window_shortcut_keeps_import_pause_available_as_an_undocumented_chord():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = select_presentation_profile(
        platform_name="unsupported"
    )
    window.wnd = SimpleNamespace(keys=SimpleNamespace(P=80, O=79))
    window._keys_down = set()
    window._key_resolve_cache = {}
    window._import_active = True
    calls = []
    window._request_import_pause = lambda: calls.append("pause")

    assert window._handle_window_shortcut(
        80,
        SimpleNamespace(ctrl=True, shift=True),
    ) is True

    assert calls == ["pause"]


def test_open_action_uses_runtime_and_handles_unavailable_directory_selection(
    monkeypatch,
):
    window = object.__new__(viewer_window.CaveViewerWindow)
    runtime = object()
    logger = FakeLogger()
    calls = []

    def unavailable_picker(*, platform_runtime=None):
        calls.append(platform_runtime)
        raise viewer_window.DesktopServiceError("Directory selection unavailable.")

    monkeypatch.setattr(viewer_window, "_LOG", logger)
    monkeypatch.setattr(viewer_window, "pick_folder_dialog", unavailable_picker)
    window._platform_runtime = runtime

    window._handle_open_button_click()

    assert calls == [runtime]
    assert logger.warning_messages == [
        "Map folder selection unavailable: Directory selection unavailable."
    ]


def test_window_shortcut_leaves_control_a_unhandled():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = select_presentation_profile(
        platform_name="unsupported"
    )
    window.wnd = SimpleNamespace(keys=SimpleNamespace(W=87, O=79, A=65))
    window._keys_down = set()
    window._key_resolve_cache = {}

    assert window._handle_window_shortcut(65, SimpleNamespace(ctrl=True)) is False


def test_window_shortcut_leaves_command_a_unhandled_on_macos():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = select_presentation_profile(
        platform_name="darwin"
    )
    window.wnd = SimpleNamespace(keys=SimpleNamespace(W=87, O=79, A=65))
    window._keys_down = set()
    window._key_resolve_cache = {}
    window._raw_command_modifier_down = lambda: False

    assert window._handle_window_shortcut(65, SimpleNamespace(command=True)) is False


def test_window_shortcut_leaves_command_w_unhandled_on_macos():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._presentation_profile = select_presentation_profile(
        platform_name="darwin"
    )
    window.wnd = SimpleNamespace(keys=SimpleNamespace(W=87, O=79))
    window._keys_down = set()
    window._key_resolve_cache = {}
    window._raw_command_modifier_down = lambda: False
    closed = []
    window.on_close = lambda: closed.append("closed")

    assert window._handle_window_shortcut(87, SimpleNamespace(command=True)) is False
    assert window._handle_window_shortcut(87, SimpleNamespace()) is False
    assert closed == []

def test_linux_launch_defers_sizing_to_glfw_workarea(monkeypatch):
    calls = []
    target = ViewerLaunchTarget(
        ViewerLaunchRoute.GLFW_MODERNGL,
        WindowBackendPlan(WindowSystem.AUTO, (WindowSystem.X11,)),
    )
    preflight = ViewerLaunchPreflight(
        capability=CapabilityResult.available(
            target,
            reason_code="viewer_launch_glfw_route_available",
        ),
        decision=FeatureDecision(
            feature=FeatureId.VIEWER_LAUNCH,
            state=FeatureState.ENABLED,
            reason_code="viewer_launch_glfw_route_available",
            explanation="The viewer window is available.",
            route=target.route_key,
        ),
    )
    monkeypatch.setattr(
        viewer_window,
        "_presentation_profile_for_runtime",
        lambda _runtime: select_presentation_profile(platform_name="linux"),
    )
    monkeypatch.setattr(
        viewer_window,
        "_desktop_relative_window_size",
        lambda: (_ for _ in ()).throw(
            AssertionError("Linux sizing must not mix Tk and GLFW coordinates")
        ),
    )
    monkeypatch.setattr(
        viewer_window,
        "create_window_backend_adapter",
        lambda: SimpleNamespace(
            launch_viewer=lambda launch_target, request: calls.append(
                (launch_target, request)
            )
        ),
    )
    monkeypatch.setattr(
        viewer_window,
        "viewer_launch_preflight",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr(
        viewer_window,
        "authorized_viewer_launch_target",
        lambda _preflight: target,
    )

    session = _viewer_session()
    viewer_window._launch_viewer_window(session)

    assert viewer_window.CaveViewerWindow.window_size == (1600, 1000)
    assert calls[0][0] is target
    request = calls[0][1]
    assert request.config_class is not viewer_window.CaveViewerWindow
    assert issubclass(request.config_class, viewer_window.CaveViewerWindow)
    assert request.config_class._viewer_session is session
    assert request.config_class.window_size == (1600, 1000)
    assert request.runner is viewer_window._run_moderngl_window_config
    assert request.window_size_fraction == 0.8
    assert request.fallback_window_size == (1600, 1000)
    assert request.force_resizable_window is True


def test_session_window_config_classes_keep_sequential_launches_isolated():
    first_session = _viewer_session()
    second_session = _pending_import_session()

    first_config = viewer_window._session_window_config_class(
        first_session,
        window_size=(800, 600),
    )
    second_config = viewer_window._session_window_config_class(
        second_session,
        window_size=(1200, 900),
    )

    assert first_config is not second_config
    assert first_config._viewer_session is first_session
    assert first_config.window_size == (800, 600)
    assert second_config._viewer_session is second_session
    assert second_config.window_size == (1200, 900)
    assert not hasattr(viewer_window.CaveViewerWindow, "_viewer_session")


def test_viewer_launch_uses_injected_runtime_presentation_profile(monkeypatch):
    calls = []
    target = ViewerLaunchTarget(
        ViewerLaunchRoute.GLFW_MODERNGL,
        WindowBackendPlan(WindowSystem.AUTO, (WindowSystem.X11,)),
    )
    preflight = ViewerLaunchPreflight(
        capability=CapabilityResult.available(
            target,
            reason_code="viewer_launch_glfw_route_available",
        ),
        decision=FeatureDecision(
            feature=FeatureId.VIEWER_LAUNCH,
            state=FeatureState.ENABLED,
            reason_code="viewer_launch_glfw_route_available",
            explanation="The viewer window is available.",
            route=target.route_key,
        ),
    )
    window_backend_adapter = SimpleNamespace(
        launch_viewer=lambda launch_target, request: calls.append(
            (launch_target, request)
        )
    )
    runtime = SimpleNamespace(
        presentation_profile=select_presentation_profile(platform_name="linux"),
        viewer_launch_preflight=lambda: preflight,
        window_backend_adapter=window_backend_adapter,
    )
    monkeypatch.setattr(
        viewer_window,
        "create_window_backend_adapter",
        lambda: pytest.fail("launch must use the injected runtime window adapter"),
    )
    monkeypatch.setattr(
        viewer_window,
        "authorized_viewer_launch_target",
        lambda received_preflight: target if received_preflight is preflight else None,
    )

    viewer_window._launch_viewer_window(_viewer_session(platform_runtime=runtime))

    assert calls[0][0] is target
    assert calls[0][1].window_size_fraction == 0.8
    assert calls[0][1].fallback_window_size == (1600, 1000)


def test_viewer_launch_refuses_disabled_preflight_before_window_execution(monkeypatch):
    disabled_preflight = ViewerLaunchPreflight(
        capability=CapabilityResult.unavailable(
            reason_code="viewer_launch_display_unavailable",
        ),
        decision=FeatureDecision(
            feature=FeatureId.VIEWER_LAUNCH,
            state=FeatureState.DISABLED,
            reason_code="viewer_launch_display_unavailable",
            explanation="The viewer cannot start because no supported display is available.",
        ),
    )
    monkeypatch.setattr(
        viewer_window,
        "viewer_launch_preflight",
        lambda **_kwargs: disabled_preflight,
    )
    monkeypatch.setattr(
        viewer_window,
        "create_window_backend_adapter",
        lambda: pytest.fail(
            "a disabled viewer route must not initialize a window"
        ),
    )

    with pytest.raises(ViewerLaunchError, match="no supported display"):
        viewer_window._launch_viewer_window(_viewer_session())


def test_run_viewer_forwards_map_root_to_the_deferred_window_load(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "managed-cache"
    map_root = tmp_path / "Devils Eye"
    launched = []
    monkeypatch.setattr(
        viewer_window.chunker,
        "load_manifest",
        lambda cache: {"cache": cache},
    )
    monkeypatch.setattr(
        viewer_window,
        "_launch_viewer_window",
        lambda session: launched.append(
            (
                session.config.cache_dir,
                session.config.textures_dir,
                session.config.map_root,
            )
        ),
    )

    viewer_window.run_viewer(
        str(cache_dir),
        str(cache_dir),
        map_root=map_root,
    )

    assert launched == [
        (str(cache_dir), str(cache_dir), str(map_root.resolve()))
    ]
    assert not hasattr(viewer_window.CaveViewerWindow, "cave_map_root")


def test_pending_import_failure_suppresses_only_its_native_close_signal(monkeypatch):
    descriptor = {"format": "glb", "glb_path": "/maps/cave.glb"}

    def fail_after_recording_outcome(session):
        session.record_outcome(
            kind="import_failed",
            message="cache build already active",
            suggestion="wait, then retry",
        )
        raise SystemExit(1)

    monkeypatch.setattr(
        viewer_window,
        "_launch_viewer_window",
        fail_after_recording_outcome,
    )

    outcome = viewer_window.run_viewer_with_pending_import(descriptor, "/maps")

    assert outcome == viewer_window.ViewerSessionOutcome(
        kind="import_failed",
        message="cache build already active",
        suggestion="wait, then retry",
    )
    assert not hasattr(viewer_window.CaveViewerWindow, "cave_pending_import")
    assert not hasattr(viewer_window.CaveViewerWindow, "cave_session_outcome")


def test_pending_import_does_not_suppress_unrelated_native_exit(monkeypatch):
    descriptor = {"format": "glb", "glb_path": "/maps/cave.glb"}
    monkeypatch.setattr(
        viewer_window,
        "_launch_viewer_window",
        lambda _session: (_ for _ in ()).throw(SystemExit(7)),
    )

    with pytest.raises(SystemExit) as raised:
        viewer_window.run_viewer_with_pending_import(descriptor, "/maps")

    assert raised.value.code == 7


def test_pending_import_failure_does_not_hide_an_unrelated_viewer_exception(
    monkeypatch,
):
    descriptor = {"format": "glb", "glb_path": "/maps/cave.glb"}

    def fail_after_recording_outcome(session):
        session.record_outcome(
            kind="import_failed",
            message="cache build already active",
        )
        raise ValueError("renderer teardown failed")

    monkeypatch.setattr(
        viewer_window,
        "_launch_viewer_window",
        fail_after_recording_outcome,
    )

    with pytest.raises(ValueError, match="renderer teardown failed"):
        viewer_window.run_viewer_with_pending_import(descriptor, "/maps")


def test_run_viewer_benchmark_records_scenario_and_cache_identity(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest_bytes = b'{"chunks": {}, "mtl_materials": {}}\n'
    (cache_dir / viewer_window.chunker.MANIFEST_NAME).write_bytes(manifest_bytes)
    scenario = SimpleNamespace(
        name="gold",
        fingerprint="scenario-sha",
        window_size=(640, 480),
        render_distance=4,
    )
    calls = []
    monkeypatch.setenv("CAVEVIEWER_MEMORY_UTILIZATION_TARGET", "12")
    monkeypatch.setenv("CAVEVIEWER_GPU_MEMORY_UTILIZATION_TARGET", "65")
    monkeypatch.delenv("CAVEVIEWER_GPU_MEMORY_GB", raising=False)
    monkeypatch.setenv("CAVEVIEWER_TEXTURE_RESIDENT_CACHE_MB", "768")
    monkeypatch.setenv("CAVEVIEWER_IO_WORKERS", "3")
    monkeypatch.setenv("CAVEVIEWER_IO_RESERVED_CPUS", "2")
    monkeypatch.setenv("CAVEVIEWER_UPLOAD_CHUNKS_PER_FRAME", "5")
    monkeypatch.setenv("CAVEVIEWER_UPLOAD_GROUPS_PER_FRAME", "7")
    monkeypatch.setenv("CAVEVIEWER_UPLOAD_TIME_BUDGET_MS", "9.5")

    monkeypatch.setattr(
        viewer_window.chunker,
        "load_manifest",
        lambda cache: {"cache": cache},
    )

    def fake_launch(session, *, window_size_override=None):
        calls.append(
            (
                window_size_override,
                session.config.benchmark,
            )
        )

    monkeypatch.setattr(viewer_window, "_launch_viewer_window", fake_launch)

    summary_path = viewer_window.run_viewer_benchmark(
        str(cache_dir),
        str(cache_dir),
        scenario,
        str(tmp_path / "out"),
    )

    config = calls[0][1]
    assert summary_path == str(tmp_path / "out" / "summary.json")
    assert calls[0][0] is None
    assert config.scenario is scenario
    assert config.environment["scenario_fingerprint"] == "scenario-sha"
    assert config.environment["cache_manifest_sha256"] == hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    assert config.environment["streaming_settings"] == {
        "render_distance_chunks": 4,
        "system_ram_target_percent": "12",
        "gpu_memory_target_percent": "65",
        "gpu_memory_override_gb": "",
        "texture_resident_cache_mb": "768",
        "io_workers": "3",
        "io_reserved_cpus": "2",
        "upload_chunks_per_frame": "5",
        "upload_groups_per_frame": "7",
        "upload_time_budget_ms": "9.5",
    }
    assert len(config.environment["streaming_settings_fingerprint"]) == 64
    assert not hasattr(viewer_window.CaveViewerWindow, "cave_benchmark_config")


def test_moderngl_runner_closes_and_destroys_window_on_keyboard_interrupt(monkeypatch):
    calls = []

    class FakeWindow:
        is_closing = False

        def close(self):
            calls.append("close")
            self.is_closing = True

        def destroy(self):
            calls.append("destroy")

    fake_window = FakeWindow()
    fake_config = SimpleNamespace(wnd=fake_window)
    fake_config_class = type("FakeConfigClass", (), {})
    created = []

    def create_window_config_instance(config_class, args=None):
        created.append((config_class, args))
        return fake_config

    def run_window_config_instance(config):
        assert config is fake_config
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        viewer_window.mglw,
        "create_window_config_instance",
        create_window_config_instance,
    )
    monkeypatch.setattr(
        viewer_window.mglw,
        "run_window_config_instance",
        run_window_config_instance,
    )

    with pytest.raises(KeyboardInterrupt):
        viewer_window._run_moderngl_window_config(
            fake_config_class,
            args=["--window", "glfw"],
        )

    assert created == [(fake_config_class, ["--window", "glfw"])]
    assert calls == ["close", "destroy"]


def test_moderngl_runner_does_not_close_window_after_normal_loop(monkeypatch):
    calls = []

    class FakeWindow:
        is_closing = False

        def close(self):
            calls.append("close")

        def destroy(self):
            calls.append("destroy")

    fake_config = SimpleNamespace(wnd=FakeWindow())

    monkeypatch.setattr(
        viewer_window.mglw,
        "create_window_config_instance",
        lambda _config_class, args=None: fake_config,
    )
    monkeypatch.setattr(
        viewer_window.mglw,
        "run_window_config_instance",
        lambda _config: calls.append("run"),
    )

    viewer_window._run_moderngl_window_config(type("FakeConfigClass", (), {}))

    assert calls == ["run"]


def test_moderngl_runner_records_native_window_checkpoints(monkeypatch):
    checkpoints = []
    fake_config = SimpleNamespace(wnd=SimpleNamespace())
    fake_config_class = type("FakeConfigClass", (), {})
    monkeypatch.setattr(
        viewer_window.mglw,
        "create_window_config_instance",
        lambda _config_class, args=None: fake_config,
    )
    monkeypatch.setattr(
        viewer_window.mglw,
        "run_window_config_instance",
        lambda _config: None,
    )
    monkeypatch.setattr(
        viewer_window,
        "record_runtime_stage",
        lambda stage, **context: checkpoints.append((stage, context)),
    )

    viewer_window._run_moderngl_window_config(fake_config_class)

    assert [stage for stage, _context in checkpoints] == [
        "viewer_window_config_create_begin",
        "viewer_window_config_created",
        "viewer_window_loop_begin",
        "viewer_window_loop_returned",
        "viewer_window_cleanup_begin",
        "viewer_window_cleanup_complete",
    ]
    assert checkpoints[0][1]["config_class"] == "FakeConfigClass"


class _ScaledStepperProbe:
    BUTTON_SIZE = viewer_window.StepperControl.BUTTON_SIZE
    VALUE_BOX_WIDTH = viewer_window.StepperControl.VALUE_BOX_WIDTH
    GAP = viewer_window.StepperControl.GAP

    def __init__(self, label: str = "BRIGHTNESS"):
        self.label = label
        self._geometry_scale = viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_SCALE
        self._text_scale = viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_TEXT_SCALE
        self._label_text_scale = (
            viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_LABEL_TEXT_SCALE
        )

    def set_scale(
        self,
        *,
        text_scale: float,
        geometry_scale: float,
        label_text_scale: float | None = None,
    ) -> None:
        self._text_scale = text_scale
        self._label_text_scale = (
            text_scale if label_text_scale is None else label_text_scale
        )
        self._geometry_scale = geometry_scale

    def total_width(self):
        return (
            self.BUTTON_SIZE * self._geometry_scale * 2
            + self.VALUE_BOX_WIDTH * self._geometry_scale
            + self.GAP * self._geometry_scale * 2
        )

    def total_height(self):
        return self.BUTTON_SIZE * self._geometry_scale


def _right_column_probe_window():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._layout_cache_size = None
    window._layout_cache_result = None
    window._viewer_ui_scale = 1.0
    window._right_column_panel_scale = viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_SCALE
    window._right_column_panel_text_scale = (
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_TEXT_SCALE
    )
    window._right_column_panel_label_text_scale = (
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_LABEL_TEXT_SCALE
    )
    window.light_stepper = _ScaledStepperProbe("BRIGHTNESS")
    window.ambient_stepper = _ScaledStepperProbe("GLOBAL LIGHT")
    window.render_distance_stepper = _ScaledStepperProbe("DISTANCE")
    window.render_mode_buttons = object.__new__(viewer_window.RenderModeButtons)
    window.render_mode_buttons._geometry_scale = (
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_SCALE
    )
    window.render_mode_buttons._text_scale = (
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_BUTTON_TEXT_SCALE
    )
    window.render_mode_buttons._render_cache_key = None
    return window


def test_right_column_panel_uses_compact_default_footprint():
    window = _right_column_probe_window()

    window_size = (1536, 864)
    column = window._right_column_layout(window_size)
    x0, y0, x1, y1 = window._right_column_panel_rect(window_size, column)

    assert 0 <= x0 < x1 <= window_size[0]
    assert 0 <= y0 < y1 <= window_size[1]
    # GLOBAL LIGHT is wider than the compact stepper itself. Its bitmap-font
    # width varies slightly across supported Python/platform combinations, so
    # derive the compact footprint from the actual rendered measurements.
    label_size = viewer_window.bitmap_font.pixel_size_at_text_scale(
        viewer_window.StepperControl.LABEL_TEXT_SIZE,
        viewer_window.StepperControl.FIXED_TEXT_SCALE
        * window._right_column_label_text_scale(),
    )
    button_width = viewer_window.RenderModeButtons.BUTTON_WIDTH * (
        window.render_mode_buttons._group_layout(
            window_size,
            column["buttons_top_y"],
        )["scale"]
    )
    content_width = max(
        window.light_stepper.total_width(),
        window.ambient_stepper.total_width(),
        window.render_distance_stepper.total_width(),
        button_width,
        *(
            viewer_window.bitmap_font.text_width_px(stepper.label, label_size)
            for stepper in (
                window.light_stepper,
                window.ambient_stepper,
                window.render_distance_stepper,
            )
        ),
    )
    assert x1 - x0 == pytest.approx(
        content_width
        + 2
        * viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_SIDE_PAD
        * window._right_column_ui_scale()
    )
    assert y1 - y0 <= 455


def test_right_column_centers_controls_when_labels_expand_the_panel():
    """A wide label must not leave the control column right-aligned in its panel."""
    window = _right_column_probe_window()
    window_size = (480, 540)

    column = window._right_column_layout(window_size)
    panel_x0, _panel_y0, panel_x1, _panel_y1 = window._right_column_panel_rect(
        window_size,
        column,
    )
    panel_center_x = (panel_x0 + panel_x1) / 2.0

    assert column["content_center_x"] == pytest.approx(panel_center_x)
    for anchor_name, stepper in (
        ("brightness_anchor", window.light_stepper),
        ("ambient_anchor", window.ambient_stepper),
        ("render_distance_anchor", window.render_distance_stepper),
    ):
        anchor_x, _anchor_y = column[anchor_name]
        assert anchor_x + stepper.total_width() / 2.0 == pytest.approx(
            panel_center_x
        )

    button_x0, _button_y0, button_x1, _button_y1 = (
        window.render_mode_buttons._button_rect_px(
            0,
            window_size,
            column["buttons_top_y"],
            column["button_right_inset"],
        )
    )
    assert (button_x0 + button_x1) / 2.0 == pytest.approx(panel_center_x)

    label_size = viewer_window.bitmap_font.pixel_size_at_text_scale(
        viewer_window.StepperControl.LABEL_TEXT_SIZE,
        viewer_window.StepperControl.FIXED_TEXT_SCALE
        * window._right_column_label_text_scale(),
    )
    side_pad = (
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_SIDE_PAD
        * window._right_column_ui_scale()
    )
    for anchor_name, stepper in (
        ("brightness_anchor", window.light_stepper),
        ("ambient_anchor", window.ambient_stepper),
        ("render_distance_anchor", window.render_distance_stepper),
    ):
        label_width = viewer_window.bitmap_font.text_width_px(
            stepper.label,
            label_size,
        )
        anchor_x, _anchor_y = column[anchor_name]
        label_x0 = anchor_x + (stepper.total_width() - label_width) / 2.0
        assert label_x0 >= panel_x0 + side_pad - 1e-6
        assert label_x0 + label_width <= panel_x1 - side_pad + 1e-6


def test_right_column_panel_scales_up_on_large_viewer_surfaces():
    baseline = _right_column_probe_window()
    large = _right_column_probe_window()

    base_column = baseline._right_column_layout((1536, 864))
    base_rect = baseline._right_column_panel_rect((1536, 864), base_column)
    large_column = large._right_column_layout((2048, 1152))
    large_rect = large._right_column_panel_rect((2048, 1152), large_column)

    assert large._right_column_ui_scale() == pytest.approx(4 / 3)
    assert large.light_stepper._geometry_scale == pytest.approx(
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_SCALE * 4 / 3
    )
    assert large.light_stepper._text_scale == pytest.approx(
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_TEXT_SCALE
    )
    assert large.light_stepper._label_text_scale == pytest.approx(
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_LABEL_TEXT_SCALE
    )
    assert large.light_stepper._label_text_scale > large.light_stepper._text_scale
    assert large.render_mode_buttons._text_scale == pytest.approx(
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_BUTTON_TEXT_SCALE
    )
    assert large.render_mode_buttons._text_scale < large.light_stepper._text_scale
    assert large_rect[2] - large_rect[0] > base_rect[2] - base_rect[0]
    assert large_rect[3] - large_rect[1] > base_rect[3] - base_rect[1]
    assert 0 <= large_rect[0] < large_rect[2] <= 2048
    assert 0 <= large_rect[1] < large_rect[3] <= 1152


def test_right_column_panel_scales_from_framebuffer_on_scaled_dpi():
    window = _right_column_probe_window()
    window.wnd = SimpleNamespace(size=(1600, 1000), buffer_size=(2048, 1280))

    column = window._right_column_layout((1600, 1000))
    rect = window._right_column_panel_rect((1600, 1000), column)

    assert window._right_column_ui_scale() == pytest.approx(4 / 3)
    assert window.light_stepper._geometry_scale == pytest.approx(
        viewer_window.CaveViewerWindow.RIGHT_COLUMN_PANEL_SCALE * 4 / 3
    )
    assert 0 <= rect[0] < rect[2] <= 1600
    assert 0 <= rect[1] < rect[3] <= 1000


def test_initial_chunk_readiness_respects_budget_limited_wanted_count():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(config=SimpleNamespace(max_loaded_chunks=100))

    assert window._initial_chunk_load_is_ready(
        {"loaded": 3, "total_available": 1655, "wanted": 3}
    ) is True
    assert window._initial_chunk_load_is_ready(
        {"loaded": 2, "total_available": 1655, "wanted": 3}
    ) is False


def test_initial_chunk_readiness_waits_for_startup_wanted_cells():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(config=SimpleNamespace(max_loaded_chunks=100))

    assert window._initial_chunk_load_is_ready(
        {
            "loaded_wanted": 6,
            "total_available": 1655,
            "wanted": 27,
        }
    ) is False
    assert window._initial_chunk_load_is_ready(
        {
            "loaded_wanted": 27,
            "total_available": 1655,
            "wanted": 27,
        }
    ) is True


def test_initial_chunk_readiness_counts_failed_wanted_chunks():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(config=SimpleNamespace(max_loaded_chunks=100))

    assert window._initial_chunk_load_is_ready(
        {
            "loaded_wanted": 2,
            "failed_wanted": 1,
            "total_available": 1655,
            "wanted": 3,
        }
    ) is True


def test_map_load_reset_restores_initial_chunk_readiness_state():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._initial_chunks_loaded = True
    window._initial_visual_ready = True
    window._initial_visual_ready_frames = 3
    window._initial_visual_ready_visible_chunks = 6
    window._initial_visual_ready_required_textures = 8
    window._initial_visual_ready_resident_textures = 8
    window._initial_visual_ready_visible_textures = 7
    window._initial_visual_ready_missing_textures = 1
    window._initial_visual_ready_expected_chunks = 9
    window._initial_visual_ready_covered_chunks = 8
    window._initial_visual_ready_missing_chunks = 1
    window._initial_visual_ready_coverage_pct = 88.0
    window._initial_visual_ready_logged = True
    window._chunk_prep_progress = 1.0
    window._chunk_prep_complete_until = 12.0
    window._chunk_prep_completion_armed = True
    window.manifest = {"source_obj": "/maps/cave.obj"}
    session = viewer_window.MapOpeningProgressSession()
    window._map_opening_progress_session = session
    import_frame = session.observe_import(
        "cave.obj",
        "writing chunk files",
        1.0,
        note="",
    )
    calls = []
    window.import_progress_panel = SimpleNamespace(
        reset_progress=lambda: calls.append("reset")
    )

    window._reset_initial_chunk_loading_state()

    assert window._initial_chunks_loaded is False
    assert window._initial_visual_ready is False
    assert window._initial_visual_ready_frames == 0
    assert window._initial_visual_ready_expected_chunks == 0
    assert window._initial_visual_ready_covered_chunks == 0
    assert window._initial_visual_ready_coverage_pct == 100.0
    assert window._initial_visual_ready_logged is False
    assert window._chunk_prep_progress == 0.0
    assert window._chunk_prep_complete_until is None
    assert window._chunk_prep_completion_armed is False
    assert calls == []
    streaming_frame = session.observe_streaming("cave.obj", 0.0)
    assert streaming_frame.session_id == import_frame.session_id
    assert streaming_frame.fraction == pytest.approx(0.90)


class _FakeMoveCamera:
    def __init__(self, position, moved_position):
        self.position = np.array(position, dtype=np.float64)
        self._moved_position = np.array(moved_position, dtype=np.float64)

    def move(
        self,
        _forward_amt,
        _right_amt,
        _up_amt,
        _dt,
        _speed_multiplier,
    ):
        self.position = self._moved_position.copy()




def test_camera_move_allows_flight_beyond_cave_volume():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.camera = _FakeMoveCamera(
        position=[5.0, 9.0, 5.0],
        moved_position=[500.0, 250.0, -400.0],
    )

    window._move_camera(0.0, 0.0, 1.0, 1.0, 1.0)

    assert window.camera.position.tolist() == [500.0, 250.0, -400.0]






def test_initial_visual_readiness_waits_for_settled_scene_frames():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(config=SimpleNamespace(max_loaded_chunks=100))
    window._initial_chunks_loaded = True
    window._initial_visual_ready = False
    window._initial_visual_ready_frames = 0
    window._initial_visual_ready_visible_chunks = 0
    window._initial_visual_ready_logged = True
    window._chunk_upload_states = {}

    stats = {
        "loaded_wanted": 27,
        "loaded": 27,
        "pending": 0,
        "ready": 0,
        "wanted": 27,
        "total_available": 1655,
    }

    first = window._initial_visual_readiness_stats(stats, 12)
    second = window._initial_visual_readiness_stats(stats, 12)
    third = window._initial_visual_readiness_stats(stats, 12)

    assert first["visual_ready"] is False
    assert second["visual_ready"] is False
    assert third["visual_ready"] is True
    assert window._initial_visual_ready_frames == 3


def test_initial_visual_readiness_waits_for_pending_upload_state():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(config=SimpleNamespace(max_loaded_chunks=100))
    window._initial_chunks_loaded = True
    window._initial_visual_ready = False
    window._initial_visual_ready_frames = 2
    window._initial_visual_ready_visible_chunks = 12
    window._initial_visual_ready_logged = True
    window._chunk_upload_states = {(1, 2, 3): {"next_group_index": 0}}

    visual_stats = window._initial_visual_readiness_stats(
        {
            "loaded_wanted": 27,
            "loaded": 27,
            "pending": 0,
            "ready": 0,
            "wanted": 27,
            "total_available": 1655,
        },
        12,
    )

    assert visual_stats["visual_ready"] is False
    assert window._initial_visual_ready_frames == 0


def test_initial_visual_readiness_waits_for_startup_texture_residency():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(
        config=SimpleNamespace(max_loaded_chunks=100),
        wanted_cells_snapshot=lambda: frozenset({(1, 2, 3)}),
    )
    window.manifest = {
        "chunks": {
            "1_2_3": {
                "materials": ["rock", "silt"],
            },
        },
    }
    window.texture_manager = SimpleNamespace(
        material_to_file={
            "rock": "rock.jpg",
            "silt": "silt.jpg",
        },
        stats=lambda: {
            "unique_files_resident": 1,
            "resident_texture_bytes": 1024,
            "resident_texture_budget_bytes": 4096,
        },
    )
    window._initial_chunks_loaded = True
    window._initial_visual_ready = False
    window._initial_visual_ready_frames = 2
    window._initial_visual_ready_visible_chunks = 1
    window._initial_visual_ready_logged = True
    window._chunk_upload_states = {}

    visual_stats = window._initial_visual_readiness_stats(
        {
            "loaded_wanted": 1,
            "loaded": 1,
            "pending": 0,
            "ready": 0,
            "wanted": 1,
            "total_available": 10,
        },
        1,
    )

    assert visual_stats["visual_ready"] is False
    assert visual_stats["visual_ready_required_textures"] == 2
    assert visual_stats["visual_ready_resident_textures"] == 1
    assert window._initial_visual_ready_frames == 0


def test_initial_visual_readiness_uses_exact_texture_sources_when_available():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(
        config=SimpleNamespace(max_loaded_chunks=100),
        wanted_cells_snapshot=lambda: frozenset({(1, 2, 3)}),
    )
    window.manifest = {
        "chunks": {
            "1_2_3": {
                "materials": ["rock", "silt"],
            },
        },
    }
    window.texture_manager = SimpleNamespace(
        material_to_file={
            "rock": "rock.jpg",
            "silt": "silt.jpg",
        },
        resident_texture_sources=lambda: ("rock.jpg",),
        stats=lambda: {
            "unique_files_resident": 2,
            "resident_texture_bytes": 1024,
            "resident_texture_budget_bytes": 4096,
        },
    )
    window._initial_chunks_loaded = True
    window._initial_visual_ready = False
    window._initial_visual_ready_frames = 2
    window._initial_visual_ready_visible_chunks = 1
    window._initial_visual_ready_logged = True
    window._chunk_upload_states = {}

    visual_stats = window._initial_visual_readiness_stats(
        {
            "loaded_wanted": 1,
            "loaded": 1,
            "pending": 0,
            "ready": 0,
            "wanted": 1,
            "total_available": 10,
        },
        1,
    )

    assert visual_stats["visual_ready"] is False
    assert visual_stats["visual_ready_required_textures"] == 2
    assert visual_stats["visual_ready_resident_textures"] == 2
    assert visual_stats["visual_ready_missing_textures"] == 1
    assert window._initial_visual_ready_frames == 0


def test_startup_visual_readiness_waits_for_frustum_coverage():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(
        config=SimpleNamespace(max_loaded_chunks=100),
        wanted_cells_snapshot=lambda: frozenset({(0, 0, 0), (1, 0, 0)}),
        _failed_cells={},
    )
    window.manifest = {
        "chunks": {
            "0_0_0": {
                "bounds_min": [-0.8, -0.8, -0.8],
                "bounds_max": [-0.2, 0.8, 0.8],
                "materials": [],
            },
            "1_0_0": {
                "bounds_min": [0.2, -0.8, -0.8],
                "bounds_max": [0.8, 0.8, 0.8],
                "materials": [],
            },
        },
    }
    window.texture_manager = SimpleNamespace(
        material_to_file={},
        resident_texture_sources=lambda: (),
        stats=lambda: {"unique_files_resident": 0},
    )
    window._initial_chunks_loaded = True
    window._initial_visual_ready = False
    window._initial_visual_ready_frames = 2
    window._initial_visual_ready_visible_chunks = 1
    window._initial_visual_ready_logged = True
    window._chunk_upload_states = {}

    visual_stats = window._initial_visual_readiness_stats(
        {
            "loaded_wanted": 2,
            "loaded": 2,
            "pending": 0,
            "ready": 0,
            "wanted": 2,
            "total_available": 10,
        },
        1,
        visible_cells=[((0, 0, 0), [(object(), object(), "rock", object())])],
        view=np.eye(4),
        projection=np.eye(4),
    )

    assert visual_stats["visual_ready"] is False
    assert visual_stats["visual_ready_expected_chunks"] == 2
    assert visual_stats["visual_ready_covered_chunks"] == 1
    assert visual_stats["visual_ready_missing_chunks"] == 1
    assert visual_stats["visual_ready_coverage_pct"] == pytest.approx(50.0)
    assert window._initial_visual_ready_frames == 0


def test_loading_render_mode_unlocks_after_initial_chunks_for_texture_settle():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._has_map_loaded = True
    window._initial_chunks_loaded = True
    window.controls_overlay = SimpleNamespace(
        is_active=True,
        is_manual_mode=False,
        is_fading=False,
    )

    assert window._buttons_locked_for_loading() is False


def test_startup_upload_limits_are_boosted_until_initial_load_is_ready():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._upload_chunks_per_frame = 1
    window._upload_groups_per_frame = 1
    window._upload_time_budget_ms = 3.0
    window._initial_chunks_loaded = False
    window.controls_overlay = SimpleNamespace(is_waiting_for_begin=True)

    chunks, operations, budget_ms = window._streaming_upload_limits()

    assert chunks >= 4
    assert operations >= 8
    assert budget_ms >= 12.0

    window._initial_chunks_loaded = True

    assert window._streaming_upload_limits() == (1, 1, 3.0)


def test_upload_limits_boost_while_current_wanted_set_is_incomplete():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._upload_chunks_per_frame = 1
    window._upload_groups_per_frame = 1
    window._upload_time_budget_ms = 3.0
    window._initial_chunks_loaded = True
    window.controls_overlay = SimpleNamespace(is_waiting_for_begin=False)

    chunks, operations, budget_ms = window._streaming_upload_limits(
        {
            "ready": 2,
            "wanted": 10,
            "loaded_wanted": 4,
            "failed_wanted": 0,
        }
    )

    assert chunks >= 2
    assert operations >= 8
    assert budget_ms >= 8.0

    assert window._streaming_upload_limits(
        {
            "ready": 0,
            "wanted": 10,
            "loaded_wanted": 4,
            "failed_wanted": 0,
        }
    ) == (1, 1, 3.0)


def test_drain_streaming_worker_failures_logs_bounded_batch(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._STREAMING_FAILURES_PER_FRAME = 1
    failure = SimpleNamespace(
        fatal=True,
        cell=(1, 0, 0),
        stage="load_chunk_file",
        thread_name="test-worker",
        error_type="ValueError",
        message="bad chunk",
    )
    world = SimpleNamespace(
        drain_worker_failures=lambda *, max_items: [failure][:max_items]
    )
    window.world = world

    window._drain_streaming_worker_failures()

    assert logger.error_messages == [
        "Streaming worker failed for chunk (1, 0, 0) during "
        "load_chunk_file on test-worker: ValueError: bad chunk"
    ]


def test_initial_compilation_completion_is_logged_once(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 12.25)
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._initial_compilation_started_at = 10.0
    window._initial_compilation_logged = False

    stats = {"loaded": 6, "pending": 1, "ready": 0, "wanted": 7}
    window._log_initial_compilation_complete(stats)
    window._log_initial_compilation_complete(stats)

    assert logger.info_messages == [
        "Initial map compilation completed in 2.25s "
        "(loaded=6 pending=1 ready=0 wanted=7)."
    ]


def test_main_thread_stall_log_reports_slow_phases_with_rate_limit(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    ticks = iter([10.0, 10.5, 13.0])
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: next(ticks))
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._main_thread_stall_last_log_at = {}

    window._log_main_thread_stall("streaming update", 0.1)
    window._log_main_thread_stall("streaming update", 0.6, pending=4)
    window._log_main_thread_stall("streaming update", 0.7, pending=5)
    window._log_main_thread_stall("streaming update", 0.8, pending=6)

    assert logger.warning_messages == [
        "Main-thread stall: streaming update took 600ms (pending=4).",
        "Main-thread stall: streaming update took 800ms (pending=6).",
    ]


def test_startup_streaming_radius_prefetches_beyond_revealed_render_distance():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.render_distance_stepper = SimpleNamespace(value=6, max_value=10)
    window.controls_overlay = SimpleNamespace(is_waiting_for_begin=True)
    window._initial_chunks_loaded = False
    window._initial_visual_ready = False

    assert window._target_streaming_load_radius() == 9


def test_streaming_radius_uses_stepper_after_begin_screen_is_dismissed():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.render_distance_stepper = SimpleNamespace(value=6, max_value=10)
    window.controls_overlay = SimpleNamespace(is_waiting_for_begin=False)
    window._initial_chunks_loaded = True
    window._initial_visual_ready = True

    assert window._target_streaming_load_radius() == 6


def test_streaming_cell_priority_prefers_camera_forward_cells():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.world = SimpleNamespace(config=SimpleNamespace(chunk_size=1.0))
    window.wnd = SimpleNamespace(size=(1600, 1000))
    window.camera = SimpleNamespace(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        fov_deg=75.0,
        forward=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )

    priority = window._streaming_cell_priority_key()

    assert priority((5, 0, 0)) < priority((0, 5, 0))
    assert priority((5, 0, 0)) < priority((-5, 0, 0))


class _FakeGpuResource:
    def __init__(self, context=None):
        self._context = context
        self.writes = []
        self.released = False

    def write(self, data):
        byte_count = len(data)
        self.writes.append(byte_count)
        if self._context is not None:
            self._context.buffer_write_sizes.append(byte_count)

    def release(self):
        self.released = True


class _FakeViewerContext:
    def __init__(self):
        self.buffer_sizes = []
        self.buffer_reserves = []
        self.buffer_write_sizes = []

    def buffer(self, data=None, *, reserve=None):
        resource = _FakeGpuResource(self)
        if reserve is not None:
            self.buffer_reserves.append(reserve)
            return resource
        self.buffer_sizes.append(len(data))
        resource.write(data)
        return resource

    def vertex_array(self, *_args):
        return _FakeGpuResource()


class _FakeTextureManager:
    def __init__(self):
        self.acquires = []
        self.releases = []

    def acquire(self, _material_name):
        self.acquires.append(_material_name)
        return _FakeGpuResource()

    def release(self, _material_name):
        self.releases.append(_material_name)


def _drain_chunk_ready(window, chunk_data, *, max_calls=32):
    for _ in range(max_calls):
        if window._on_chunk_ready(chunk_data):
            return True
    return False


def test_chunk_aabbs_are_tracked_only_for_loaded_chunks():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.ctx = _FakeViewerContext()
    window.program = object()
    window.texture_manager = _FakeTextureManager()
    window.render_mode_buttons = SimpleNamespace(smooth_shading_enabled=True)
    window._upload_groups_per_frame = 1
    window._upload_time_budget_ms = 100.0
    window._streaming_frame_timing = None
    window._chunk_gpu_objects = {}
    window._chunk_upload_states = {}
    window._chunk_normal_cache = {}
    window._chunk_aabbs = {}
    cell = (1, 2, 3)
    positions = np.zeros((3, 3), dtype=np.float32)
    uvs = np.zeros((3, 2), dtype=np.float32)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1))
    chunk_data = SimpleNamespace(
        cell=cell,
        bounds_min=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        bounds_max=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        upload_groups=[
            viewer_window.chunker.ChunkUploadGroup(
                material_name="mat",
                positions=positions,
                uvs=uvs,
                smooth_normals=normals,
            )
        ],
    )

    assert _drain_chunk_ready(window, chunk_data)

    assert set(window._chunk_aabbs) == {cell}
    assert window._chunk_aabbs[cell][0].dtype == np.float32

    window._on_chunk_unload(cell)

    assert window._chunk_aabbs == {}


def test_chunk_upload_can_be_split_across_group_frames():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.ctx = _FakeViewerContext()
    window.program = object()
    window.texture_manager = _FakeTextureManager()
    window.render_mode_buttons = SimpleNamespace(smooth_shading_enabled=True)
    window._upload_groups_per_frame = 1
    window._upload_time_budget_ms = 100.0
    window._streaming_frame_timing = None
    window._chunk_gpu_objects = {}
    window._chunk_upload_states = {}
    window._chunk_normal_cache = {}
    window._chunk_aabbs = {}
    cell = (1, 2, 3)
    positions = np.zeros((3, 3), dtype=np.float32)
    uvs = np.zeros((3, 2), dtype=np.float32)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1))
    chunk_data = SimpleNamespace(
        cell=cell,
        bounds_min=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        bounds_max=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        upload_groups=[
            viewer_window.chunker.ChunkUploadGroup(
                material_name="mat_a",
                positions=positions,
                uvs=uvs,
                smooth_normals=normals,
            ),
            viewer_window.chunker.ChunkUploadGroup(
                material_name="mat_b",
                positions=positions,
                uvs=uvs,
                smooth_normals=normals,
            ),
        ],
    )

    assert window._on_chunk_ready(chunk_data) is False

    assert cell not in window._chunk_gpu_objects
    assert cell in window._chunk_upload_states

    assert window._on_chunk_ready(chunk_data) is False
    assert len(window._chunk_gpu_objects[cell]) == 1
    assert len(window._chunk_normal_cache[cell]) == 1
    assert window._chunk_aabbs[cell][0].dtype == np.float32

    assert window._on_chunk_ready(chunk_data) is False
    assert window._on_chunk_ready(chunk_data) is True

    assert cell not in window._chunk_upload_states
    assert len(window._chunk_gpu_objects[cell]) == 2
    assert window._chunk_aabbs[cell][0].dtype == np.float32


def test_chunk_upload_invalidates_visible_chunk_cache_when_residency_changes():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.ctx = _FakeViewerContext()
    window.program = object()
    window.texture_manager = _FakeTextureManager()
    window.render_mode_buttons = SimpleNamespace(smooth_shading_enabled=True)
    window._upload_groups_per_frame = 1
    window._upload_time_budget_ms = 100.0
    window._streaming_frame_timing = None
    window._chunk_gpu_objects = {}
    window._chunk_upload_states = {}
    window._chunk_normal_cache = {}
    window._chunk_aabbs = {}
    window._chunk_visibility_generation = 0
    cell = (1, 2, 3)
    positions = np.zeros((3, 3), dtype=np.float32)
    uvs = np.zeros((3, 2), dtype=np.float32)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1))
    chunk_data = SimpleNamespace(
        cell=cell,
        bounds_min=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        bounds_max=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        upload_groups=[
            viewer_window.chunker.ChunkUploadGroup(
                material_name="mat_a",
                positions=positions,
                uvs=uvs,
                smooth_normals=normals,
            ),
            viewer_window.chunker.ChunkUploadGroup(
                material_name="mat_b",
                positions=positions,
                uvs=uvs,
                smooth_normals=normals,
            ),
        ],
    )

    assert window._on_chunk_ready(chunk_data) is False
    assert window._chunk_visibility_generation == 0

    assert window._on_chunk_ready(chunk_data) is False
    assert len(window._chunk_gpu_objects[cell]) == 1
    assert window._chunk_visibility_generation == 1

    assert window._on_chunk_ready(chunk_data) is False
    assert window._on_chunk_ready(chunk_data) is True
    assert len(window._chunk_gpu_objects[cell]) == 2
    assert window._chunk_visibility_generation == 2

    window._on_chunk_unload(cell)

    assert window._chunk_visibility_generation == 3


def test_partial_chunk_upload_unloads_published_slices_once():
    window = object.__new__(viewer_window.CaveViewerWindow)
    texture_manager = _FakeTextureManager()
    window.ctx = _FakeViewerContext()
    window.program = object()
    window.texture_manager = texture_manager
    window.render_mode_buttons = SimpleNamespace(smooth_shading_enabled=True)
    window._upload_groups_per_frame = 1
    window._upload_time_budget_ms = 100.0
    window._streaming_frame_timing = None
    window._chunk_gpu_objects = {}
    window._chunk_upload_states = {}
    window._chunk_normal_cache = {}
    window._chunk_aabbs = {}
    cell = (1, 2, 3)
    positions = np.zeros((3, 3), dtype=np.float32)
    uvs = np.zeros((3, 2), dtype=np.float32)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1))
    chunk_data = SimpleNamespace(
        cell=cell,
        bounds_min=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        bounds_max=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        upload_groups=[
            viewer_window.chunker.ChunkUploadGroup(
                material_name="mat_a",
                positions=positions,
                uvs=uvs,
                smooth_normals=normals,
            ),
            viewer_window.chunker.ChunkUploadGroup(
                material_name="mat_b",
                positions=positions,
                uvs=uvs,
                smooth_normals=normals,
            ),
        ],
    )

    assert window._on_chunk_ready(chunk_data) is False
    assert window._on_chunk_ready(chunk_data) is False
    assert len(window._chunk_gpu_objects[cell]) == 1

    window._on_chunk_unload(cell)

    assert cell not in window._chunk_upload_states
    assert cell not in window._chunk_gpu_objects
    assert cell not in window._chunk_normal_cache
    assert cell not in window._chunk_aabbs
    assert texture_manager.releases == ["mat_a"]


def test_large_group_upload_is_sliced_into_small_vbos():
    window = object.__new__(viewer_window.CaveViewerWindow)
    context = _FakeViewerContext()
    texture_manager = _FakeTextureManager()
    window.ctx = context
    window.program = object()
    window.texture_manager = texture_manager
    window.render_mode_buttons = SimpleNamespace(smooth_shading_enabled=True)
    window._upload_groups_per_frame = 1
    window._upload_time_budget_ms = 100.0
    window._vbo_upload_slice_bytes = 3 * 8 * np.dtype(np.float32).itemsize
    window._texture_upload_slice_bytes = 1024
    window._streaming_frame_timing = None
    window._chunk_gpu_objects = {}
    window._chunk_upload_states = {}
    window._chunk_normal_cache = {}
    window._chunk_aabbs = {}
    cell = (1, 2, 3)
    positions = np.zeros((9, 3), dtype=np.float32)
    uvs = np.zeros((9, 2), dtype=np.float32)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (9, 1))
    chunk_data = SimpleNamespace(
        cell=cell,
        bounds_min=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        bounds_max=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        upload_groups=[
            viewer_window.chunker.ChunkUploadGroup(
                material_name="mat",
                positions=positions,
                uvs=uvs,
                smooth_normals=normals,
            )
        ],
    )

    assert window._on_chunk_ready(chunk_data) is False
    assert cell not in window._chunk_gpu_objects
    assert window._on_chunk_ready(chunk_data) is False
    assert len(window._chunk_gpu_objects[cell]) == 1
    for _ in range(3):
        assert window._on_chunk_ready(chunk_data) is False
    assert window._on_chunk_ready(chunk_data) is True

    assert context.buffer_sizes == []
    assert context.buffer_reserves == [96, 96, 96]
    assert context.buffer_write_sizes == [96, 96, 96]
    assert len(window._chunk_gpu_objects[cell]) == 3
    assert texture_manager.acquires == ["mat", "mat", "mat"]

    window._on_chunk_unload(cell)

    assert texture_manager.releases == ["mat", "mat", "mat"]


def test_upload_slice_size_shrinks_after_measured_stall():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._upload_time_budget_ms = 3.0
    window._vbo_upload_slice_bytes = 1024 * 1024
    window._texture_upload_slice_bytes = 1024 * 1024
    timing = viewer_window.CaveViewerWindow._new_streaming_frame_timing()

    window._adapt_upload_slice_size(
        kind="texture",
        elapsed_ms=30.0,
        byte_count=1024 * 1024,
        timing=timing,
    )
    window._adapt_upload_slice_size(
        kind="vbo",
        elapsed_ms=30.0,
        byte_count=1024 * 1024,
        timing=timing,
    )

    assert window._texture_upload_slice_bytes < 1024 * 1024
    assert window._vbo_upload_slice_bytes < 1024 * 1024
    assert timing["upload_stalls"] == 2
    assert timing["texture_upload_slice_bytes"] == window._texture_upload_slice_bytes
    assert timing["vbo_upload_slice_bytes"] == window._vbo_upload_slice_bytes


def test_upload_slice_size_uses_current_boosted_time_budget():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._upload_time_budget_ms = 3.0
    window._current_upload_time_budget_ms = 8.0
    window._vbo_upload_slice_bytes = 1024 * 1024
    timing = viewer_window.CaveViewerWindow._new_streaming_frame_timing()

    window._adapt_upload_slice_size(
        kind="vbo",
        elapsed_ms=5.0,
        byte_count=1024 * 1024,
        timing=timing,
    )

    assert window._vbo_upload_slice_bytes == 1024 * 1024
    assert timing["upload_stalls"] == 0


def test_streaming_timing_format_splits_drain_and_upload_details():
    timing = viewer_window.CaveViewerWindow._new_streaming_frame_timing()
    timing.update(
        {
            "drain_ms": 12.0,
            "ready_drain_ms": 9.0,
            "chunk_ready_ms": 5.0,
            "unload_ms": 1.0,
            "failure_drain_ms": 2.0,
            "buffer_alloc_ms": 1.5,
            "buffer_write_ms": 2.5,
            "texture_alloc_ms": 0.5,
            "texture_upload_ms": 3.5,
            "texture_evictions": 2,
            "texture_evicted_bytes": 3 * 1024 * 1024,
            "vbo_upload_slice_bytes": 256 * 1024,
            "texture_upload_slice_bytes": 128 * 1024,
            "upload_stalls": 1,
        }
    )

    detail = viewer_window.CaveViewerWindow._format_streaming_frame_timing(timing)

    assert "ready_drain=9.0ms" in detail
    assert "ready_other=3.0ms" in detail
    assert "failures=2.0ms" in detail
    assert "drain_other=1.0ms" in detail
    assert "buffer_alloc=1.5ms" in detail
    assert "buffer_write=2.5ms" in detail
    assert "tex_alloc=0.5ms" in detail
    assert "tex_upload=3.5ms" in detail
    assert "slices=vbo:256KB/tex:128KB" in detail
    assert "stalls=1" in detail
    assert "tex_evict=2" in detail
    assert "tex_evict_mb=3.0" in detail


def test_uncached_import_holds_desktop_inhibitor_until_import_finishes(
    monkeypatch,
):
    calls = []
    descriptor = {"glb_path": "/maps/cave.glb"}

    def acquire(map_name):
        calls.append(("acquire_inhibitor", map_name))
        return FakeImportInhibitor(calls)

    def start_process(model_descriptor, textures_dir):
        calls.append(("start_process", model_descriptor, textures_dir))
        events = queue.Queue()
        events.put(("progress", "building cache", 0.5))
        events.put(("done", "/cache/cave", "/cache/cave"))
        return SimpleNamespace(process=FakeImportProcess(calls), events=events)

    monkeypatch.setattr(viewer_window, "_acquire_map_import_inhibitor", acquire)
    monkeypatch.setattr(viewer_window.chunker, "cache_is_valid", lambda _path: False)
    monkeypatch.setattr(viewer_window, "start_import_process", start_process)

    window = _import_window()
    window._start_import_async(
        descriptor, "/maps", "cave.glb", is_startup=True
    )
    _wait_for_import_worker(window)

    assert calls == [
        ("acquire_inhibitor", "cave.glb"),
        ("start_process", descriptor, "/maps"),
        ("join_process", 1.0),
        ("close_inhibitor",),
    ]
    assert _queued_import_messages(window) == [
        ("progress", "starting import", 0.0),
        ("progress", "building cache", 0.5),
        ("done", "/cache/cave", "/cache/cave"),
    ]


def test_uncached_import_relays_child_heartbeat(monkeypatch):
    descriptor = {"glb_path": "/maps/cave.glb"}

    def start_process(_model_descriptor, _textures_dir):
        events = queue.Queue()
        events.put(("log", logging.INFO, "ImportProcess", "child import started"))
        events.put(("heartbeat", "building cache", 0.5, 12.0, 3_000, 8_000))
        events.put(("done", "/cache/cave", "/cache/cave"))
        return SimpleNamespace(
            process=FakeImportProcess(),
            events=events,
            cache_dir="/cache/cave",
        )

    monkeypatch.setattr(
        viewer_window,
        "_acquire_map_import_inhibitor",
        lambda _map_name: FakeImportInhibitor([]),
    )
    monkeypatch.setattr(viewer_window.chunker, "cache_is_valid", lambda _path: False)
    monkeypatch.setattr(viewer_window, "start_import_process", start_process)

    window = _import_window()
    window._start_import_async(
        descriptor, "/maps", "cave.glb", is_startup=True
    )
    _wait_for_import_worker(window)

    assert _queued_import_messages(window) == [
        ("progress", "starting import", 0.0),
        ("heartbeat", "building cache", 0.5, 12.0, 3_000, 8_000),
        ("done", "/cache/cave", "/cache/cave"),
    ]


def test_uncached_import_relays_child_keyboard_interrupt_as_cancelled(monkeypatch):
    calls = []
    descriptor = {"glb_path": "/maps/cancelled.glb"}

    def start_process(model_descriptor, textures_dir):
        calls.append(("start_process", model_descriptor, textures_dir))
        events = queue.Queue()
        events.put(("cancelled",))
        return SimpleNamespace(
            process=FakeImportProcess(calls),
            events=events,
            cache_dir="/cache/cancelled",
        )

    monkeypatch.setattr(
        viewer_window,
        "_acquire_map_import_inhibitor",
        lambda _map_name: FakeImportInhibitor(calls),
    )
    monkeypatch.setattr(viewer_window.chunker, "cache_is_valid", lambda _path: False)
    monkeypatch.setattr(viewer_window, "start_import_process", start_process)

    window = _import_window()
    window._start_import_async(
        descriptor, "/maps", "cancelled.glb", is_startup=True
    )
    _wait_for_import_worker(window)

    assert calls == [
        ("start_process", descriptor, "/maps"),
        ("join_process", 1.0),
        ("close_inhibitor",),
    ]
    assert _queued_import_messages(window) == [
        ("progress", "starting import", 0.0),
        ("cancelled",),
    ]


def test_drain_import_queue_heartbeat_updates_visible_progress():
    window = _import_window()
    window._import_queue = queue.Queue()
    window._import_model_format = "obj"
    window._import_progress_stage = "starting import"
    window._import_progress_fraction = 0.0
    window._import_queue.put(
        ("heartbeat", "building cache", 0.5, 12.0, 3_000, 8_000)
    )

    window._drain_import_queue()

    assert window._import_progress_stage == "building cache"
    assert window._import_progress_fraction == 0.5
    assert window._import_progress_title == ""
    assert (
        window._import_progress_note
        == "First-time setup in progress. Next time, this map will open faster."
    )


def test_import_progress_message_switches_for_resume(monkeypatch):
    monkeypatch.setattr(viewer_window.sys, "platform", "linux")
    window = _import_window()
    window._import_model_format = "obj"

    window._update_import_progress_message_for_stage("resuming import")

    assert window._import_resuming_from_checkpoint is True
    assert window._import_progress_title == ""
    assert window._import_progress_note == "Using saved work from the previous session."


def test_pending_import_splash_renders_logo_before_import_starts(monkeypatch):
    rendered = []

    class FakeImportProgressPanel:
        def render(
            self,
            window_size,
            map_name,
            stage,
            fraction,
            *,
            title,
            note,
            progress_session_id=None,
        ):
            rendered.append(
                (
                    window_size,
                    map_name,
                    stage,
                    fraction,
                    title,
                    note,
                    progress_session_id,
                )
            )

    window = _import_window()
    window.wnd = SimpleNamespace(size=(800, 600), buffer_size=(820, 600))
    window.import_progress_panel = FakeImportProgressPanel()

    window._render_pending_import_splash()

    assert rendered == [
        (
            (820, 600),
            "cave.obj",
            "starting import",
            0.0,
            "",
            "First-time setup in progress. Next time, this map will open faster.",
            1,
        )
    ]


def test_present_pending_import_splash_swaps_when_backend_supports_it(monkeypatch):
    rendered = []
    calls = []

    class FakeImportProgressPanel:
        def render(
            self,
            window_size,
            map_name,
            stage,
            fraction,
            *,
            title,
            note,
            progress_session_id=None,
        ):
            rendered.append(
                (
                    window_size,
                    map_name,
                    stage,
                    fraction,
                    title,
                    note,
                    progress_session_id,
                )
            )

    window = _import_window()
    window.wnd = SimpleNamespace(
        size=(800, 600),
        swap_buffers=lambda: calls.append("swap"),
    )
    window.import_progress_panel = FakeImportProgressPanel()

    assert window._present_pending_import_splash_now() is True
    assert calls == ["swap"]
    assert rendered[0][1:5] == (
        "cave.obj",
        "starting import",
        0.0,
        "",
    )


def test_present_pending_import_splash_renders_without_swap_support(monkeypatch):
    rendered = []

    class FakeImportProgressPanel:
        def render(
            self,
            window_size,
            map_name,
            stage,
            fraction,
            *,
            title,
            note,
            progress_session_id=None,
        ):
            rendered.append(
                (
                    window_size,
                    map_name,
                    stage,
                    fraction,
                    title,
                    note,
                    progress_session_id,
                )
            )

    window = _import_window()
    window.wnd = SimpleNamespace(size=(800, 600))
    window.import_progress_panel = FakeImportProgressPanel()

    assert window._present_pending_import_splash_now() is False
    assert rendered[0][1:5] == (
        "cave.obj",
        "starting import",
        0.0,
        "",
    )


def test_startup_render_presents_splash_before_starting_import():
    calls = []
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window.wnd = SimpleNamespace(size=(800, 600), buffer_size=(800, 600))
    window._closing_requested = False
    window._startup_focus_enabled = False
    window._is_iconified = False
    window._has_map_loaded = False
    window._import_active = False
    window._pending_import_started = False
    window._pending_import_splash_rendered = False
    window._sync_render_mode_loading_policy = lambda: None
    window._query_runtime_iconified_state = lambda: False
    window._set_background_pause = lambda _should_pause, _reason: None
    window._render_import_pause_notice_if_active = lambda: False
    window._render_pending_import_splash = lambda: calls.append("splash")
    window._run_pending_import = lambda: calls.append("start import")

    window.on_render(0.0, 0.0)

    assert calls == ["splash"]
    assert window._pending_import_splash_rendered is True
    assert window._pending_import_started is False

    window.on_render(0.0, 0.0)

    assert calls == ["splash", "splash", "start import"]
    assert window._pending_import_started is True


def test_startup_render_starts_import_when_splash_was_already_presented():
    calls = []
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window.wnd = SimpleNamespace(size=(800, 600), buffer_size=(800, 600))
    window._closing_requested = False
    window._startup_focus_enabled = False
    window._is_iconified = False
    window._has_map_loaded = False
    window._import_active = False
    window._pending_import_started = False
    window._pending_import_splash_rendered = True
    window._sync_render_mode_loading_policy = lambda: None
    window._query_runtime_iconified_state = lambda: False
    window._set_background_pause = lambda _should_pause, _reason: None
    window._render_import_pause_notice_if_active = lambda: False
    window._render_pending_import_splash = lambda: calls.append("splash")
    window._run_pending_import = lambda: calls.append("start import")

    window.on_render(0.0, 0.0)

    assert calls == ["splash", "start import"]
    assert window._pending_import_started is True


def test_ready_cache_startup_is_deferred_until_render_loop():
    source = inspect.getsource(viewer_window.CaveViewerWindow.__init__)

    assert "self._startup_map_load_pending =" in source
    assert "self._load_map(" not in source


def test_ready_cache_startup_splash_is_indeterminate():
    clear_calls = []
    rendered = []

    class FakeImportProgressPanel:
        def render(
            self,
            window_size,
            map_name,
            stage,
            fraction,
            *,
            title,
            note,
            progress_session_id=None,
        ):
            rendered.append(
                (
                    window_size,
                    map_name,
                    stage,
                    fraction,
                    title,
                    note,
                    progress_session_id,
                )
            )

    window = object.__new__(viewer_window.CaveViewerWindow)
    window.ctx = SimpleNamespace(clear=lambda *color: clear_calls.append(color))
    window.wnd = SimpleNamespace(size=(800, 600), buffer_size=(820, 600))
    window.import_progress_panel = FakeImportProgressPanel()
    window._startup_map_load_pending = (
        "/cache/devils-eye",
        "/textures/devils-eye",
        {"source_obj": "/maps/devils_eye.obj"},
        "/maps/devils-eye",
    )

    window._render_startup_map_load_splash()

    assert clear_calls == [(0.02, 0.02, 0.03)]
    assert rendered == [
        (
            (820, 600),
            "devils_eye.obj",
            "preparing cave",
            None,
            "",
            "",
            1,
        )
    ]


def test_startup_render_presents_ready_cache_splash_before_loading_map():
    calls = []
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window.wnd = SimpleNamespace(size=(800, 600), buffer_size=(800, 600))
    window._closing_requested = False
    window._startup_focus_enabled = False
    window._is_iconified = False
    window._has_map_loaded = False
    window._import_active = False
    window._startup_map_load_pending = (
        "/cache/devils-eye",
        "/textures/devils-eye",
        {"source_obj": "/maps/devils_eye.obj"},
        "/maps/devils-eye",
    )
    window._startup_map_load_splash_rendered = False
    window._sync_render_mode_loading_policy = lambda: None
    window._query_runtime_iconified_state = lambda: False
    window._set_background_pause = lambda _should_pause, _reason: None
    window._drain_recording_stop_results = lambda: None
    window._render_startup_map_load_splash = lambda: calls.append("splash")
    window._load_map = lambda *args, **kwargs: calls.append(
        ("load", args, kwargs)
    )

    window.on_render(0.0, 0.0)

    assert calls == ["splash"]
    assert window._startup_map_load_splash_rendered is True
    assert window._has_map_loaded is False

    window.on_render(0.0, 0.0)

    assert calls == [
        "splash",
        (
            "load",
            (
                "/cache/devils-eye",
                "/textures/devils-eye",
                {"source_obj": "/maps/devils_eye.obj"},
            ),
            {"map_root": "/maps/devils-eye"},
        ),
    ]
    assert window._startup_map_load_pending is None
    assert window._has_map_loaded is True


def test_texture_validation_queues_without_blocking(monkeypatch):
    window = object.__new__(viewer_window.CaveViewerWindow)
    manager = FakeTextureValidationManager()
    future = FakeFuture(done=False)
    executor = FakeExecutor(future)
    window.texture_manager = manager
    window.cache_dir = "/cache/devils-eye"
    window._texture_validation_executor = None
    window._texture_validation_future = None
    window._texture_validation_manager = None
    window._texture_validation_cache_dir = None
    window._texture_validation_started_at = None

    monkeypatch.setattr(
        viewer_window,
        "ThreadPoolExecutor",
        lambda **_kwargs: executor,
    )

    assert window._start_texture_validation_async() is True

    assert window._texture_validation_future is future
    assert window._texture_validation_executor is executor
    assert window._texture_validation_manager is manager
    assert window._texture_validation_cache_dir == "/cache/devils-eye"
    assert len(executor.submit_calls) == 1
    fn, args, kwargs = executor.submit_calls[0]
    assert fn.__self__ is manager
    assert fn.__name__ == "validate_textures"
    assert args == ()
    assert kwargs == {}
    assert manager.calls == 0


def test_texture_validation_completion_shuts_down_executor():
    window = object.__new__(viewer_window.CaveViewerWindow)
    manager = FakeTextureValidationManager()
    future = FakeFuture({"found": ["texture.jpg"], "missing": []})
    executor = FakeExecutor(future)
    window.texture_manager = manager
    window.cache_dir = "/cache/devils-eye"
    window._texture_validation_executor = executor
    window._texture_validation_future = future
    window._texture_validation_manager = manager
    window._texture_validation_cache_dir = "/cache/devils-eye"
    window._texture_validation_started_at = None

    window._update_texture_validation()

    assert future.result_called is True
    assert executor.shutdown_calls == [
        {"wait": False, "cancel_futures": True}
    ]
    assert window._texture_validation_future is None
    assert window._texture_validation_executor is None


def test_texture_validation_completion_discards_stale_map_result():
    window = object.__new__(viewer_window.CaveViewerWindow)
    old_manager = FakeTextureValidationManager()
    new_manager = FakeTextureValidationManager()
    future = FakeFuture({"found": ["old.jpg"], "missing": []})
    executor = FakeExecutor(future)
    window.texture_manager = new_manager
    window.cache_dir = "/cache/new"
    window._texture_validation_executor = executor
    window._texture_validation_future = future
    window._texture_validation_manager = old_manager
    window._texture_validation_cache_dir = "/cache/old"
    window._texture_validation_started_at = None

    window._update_texture_validation()

    assert future.result_called is False
    assert executor.shutdown_calls == [
        {"wait": False, "cancel_futures": True}
    ]
    assert window._texture_validation_future is None


def test_texture_validation_cancel_shuts_down_executor():
    window = object.__new__(viewer_window.CaveViewerWindow)
    future = FakeFuture(done=False)
    executor = FakeExecutor(future)
    window._texture_validation_executor = executor
    window._texture_validation_future = future
    window._texture_validation_manager = FakeTextureValidationManager()
    window._texture_validation_cache_dir = "/cache/devils-eye"
    window._texture_validation_started_at = None

    assert window._cancel_texture_validation() is True

    assert future.cancelled is True
    assert executor.shutdown_calls == [
        {"wait": False, "cancel_futures": True}
    ]
    assert window._texture_validation_future is None


def test_iconified_render_throttles_polling_without_sleep(monkeypatch):
    ticks = iter([10.0, 10.01, 10.13])
    drains = []
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        viewer_window.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("on_render must not sleep")
        ),
    )

    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window._closing_requested = False
    window._is_iconified = False
    window._query_runtime_iconified_state = lambda: True
    window._set_background_pause = (
        lambda should_pause, _reason: setattr(window, "_is_iconified", should_pause)
    )
    window._drain_recording_stop_results = lambda: drains.append("drain")

    window.on_render(0.0, 0.0)
    window.on_render(0.0, 0.0)
    window.on_render(0.0, 0.0)

    assert drains == ["drain", "drain"]


def test_import_progress_render_draws_every_callback_without_sleep(monkeypatch):
    clear_calls = []
    drain_calls = []
    render_calls = []
    monkeypatch.setattr(
        viewer_window.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("on_render must not sleep")
        ),
    )
    monkeypatch.setattr(viewer_window.bitmap_font, "set_raster_scale", lambda _scale: None)

    class FakeProgressPanel:
        def render(
            self,
            window_size,
            map_name,
            stage,
            fraction,
            *,
            title,
            note,
            progress_session_id=None,
        ):
            render_calls.append(
                (
                    window_size,
                    map_name,
                    stage,
                    fraction,
                    title,
                    note,
                    progress_session_id,
                )
            )

    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window._closing_requested = False
    window.wnd = SimpleNamespace(size=(800, 600), buffer_size=(820, 600))
    window.ctx = SimpleNamespace(clear=lambda *color: clear_calls.append(color))
    window.import_progress_panel = FakeProgressPanel()
    window._startup_focus_enabled = False
    window._is_iconified = False
    window._query_runtime_iconified_state = lambda: False
    window._set_background_pause = (
        lambda should_pause, _reason: setattr(window, "_is_iconified", should_pause)
    )
    window._sync_render_mode_loading_policy = lambda: None
    window._drain_recording_stop_results = lambda: None
    window._drain_import_queue = lambda: drain_calls.append("drain")
    window._import_active = True
    window._import_progress_fraction = 0.5
    window._import_map_name = "cave.obj"
    window._import_progress_stage = "building cache"
    window._import_progress_title = ""
    window._import_progress_note = ""

    window.on_render(0.0, 0.0)
    window.on_render(0.0, 0.0)
    window.on_render(0.0, 0.0)

    assert drain_calls == ["drain", "drain", "drain"]
    assert clear_calls == [
        (0.02, 0.02, 0.03),
        (0.02, 0.02, 0.03),
        (0.02, 0.02, 0.03),
    ]
    assert render_calls == [
        ((820, 600), "cave.obj", "building cache", 0.45, "", "", 1),
        ((820, 600), "cave.obj", "building cache", 0.45, "", "", 1),
        ((820, 600), "cave.obj", "building cache", 0.45, "", "", 1),
    ]


def test_import_pause_notice_uses_framebuffer_surface_size():
    calls = []

    class FakeImportController:
        def render_pause_notice_if_active(self, panel, window, window_size):
            calls.append((panel, window, window_size))
            return True

    window = _import_window()
    window.wnd = SimpleNamespace(size=(800, 600), buffer_size=(820, 600))
    window.import_progress_panel = object()
    window._ensure_import_controller = lambda: FakeImportController()

    assert window._render_import_pause_notice_if_active() is True
    assert calls == [(window.import_progress_panel, window.wnd, (820, 600))]


def test_import_pause_notice_render_is_throttled_without_sleep(monkeypatch):
    ticks = iter([30.0, 30.01, 30.04])
    notice_calls = []
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        viewer_window.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("on_render must not sleep")
        ),
    )
    monkeypatch.setattr(viewer_window.bitmap_font, "set_raster_scale", lambda _scale: None)

    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window._closing_requested = False
    window.wnd = SimpleNamespace(size=(800, 600), buffer_size=(800, 600))
    window._startup_focus_enabled = False
    window._is_iconified = False
    window._has_map_loaded = False
    window._import_active = False
    window._pending_import_started = False
    window._pending_import_splash_rendered = False
    window._import_pause_notice_until = 999.0
    window._query_runtime_iconified_state = lambda: False
    window._set_background_pause = (
        lambda should_pause, _reason: setattr(window, "_is_iconified", should_pause)
    )
    window._sync_render_mode_loading_policy = lambda: None
    window._drain_recording_stop_results = lambda: None
    window._render_import_pause_notice_if_active = (
        lambda: notice_calls.append("notice") or True
    )
    window._render_pending_import_splash = lambda: (_ for _ in ()).throw(
        AssertionError("pause notice should keep startup splash path inactive")
    )

    window.on_render(0.0, 0.0)
    window.on_render(0.0, 0.0)
    window.on_render(0.0, 0.0)

    assert notice_calls == ["notice", "notice"]


def test_render_during_window_setup_returns_before_full_state_exists():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = False

    window.on_render(0.0, 0.0)


def test_key_event_during_window_setup_returns_before_full_state_exists():
    window = object.__new__(viewer_window.CaveViewerWindow)

    window.on_key_event(0, 0, None)


def test_mouse_motion_during_window_setup_returns_before_full_state_exists():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = False

    window.on_mouse_position_event(10, 20, 1, -1)


def test_mouse_callbacks_during_setup_return_before_controls_exist():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = False

    window.on_mouse_press_event(10, 20, 1)
    window.on_mouse_drag_event(10, 20, 1, -1)
    window.on_mouse_release_event(10, 20, 1)


def test_mouse_motion_after_color_picker_release_is_noop():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._window_setup_complete = True
    window._closing_requested = False
    window.color_picker = None
    window._mouse_look_active = False
    window._option_look_active = lambda: False

    window.on_mouse_position_event(10, 20, 1, -1)


def test_map_switch_teardown_uses_bounded_streaming_shutdown():
    timeouts = []
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._has_map_loaded = True
    window._stop_recording = lambda: None
    window.world = SimpleNamespace(
        shutdown=lambda *, timeout=None: timeouts.append(timeout)
    )
    window._chunk_upload_states = {}
    window._chunk_gpu_objects = {}
    window._chunk_normal_cache = {}
    window._chunk_aabbs = {}
    window.texture_manager = None
    window.minimap = None

    window._teardown_current_map()

    assert timeouts == [viewer_window._VIEWER_STREAMING_SHUTDOWN_TIMEOUT_SECONDS]


def test_final_teardown_uses_bounded_streaming_shutdown():
    timeouts = []
    window = object.__new__(viewer_window.CaveViewerWindow)
    window._has_map_loaded = True
    window._stop_recording = lambda: None
    window.world = SimpleNamespace(
        shutdown=lambda *, timeout=None: timeouts.append(timeout)
    )
    window._chunk_upload_states = {}
    window._chunk_gpu_objects = {}
    window._chunk_normal_cache = {}
    window._chunk_aabbs = {}
    window.texture_manager = None
    window.minimap = None

    window._teardown_current_map(final_shutdown=True)

    assert timeouts == [viewer_window._VIEWER_STREAMING_SHUTDOWN_TIMEOUT_SECONDS]


def test_request_import_pause_sends_child_command(monkeypatch):
    logger = FakeLogger()
    commands = queue.Queue()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    window = _import_window()
    window._import_active = True
    window._import_model_format = "obj"
    window._import_command_queue = commands
    window._import_pause_requested = False
    window._import_progress_stage = "building cache"

    window._request_import_pause()

    assert window._import_pause_requested is True
    assert window._import_progress_stage == "pausing import"
    assert window._import_progress_title == ""
    assert window._import_progress_note == "Saving a resume point."
    assert commands.get_nowait() == ("pause",)
    assert "Import pause requested" in logger.info_messages[-1]


def test_request_import_pause_warns_for_non_obj_import(monkeypatch):
    logger = FakeLogger()
    commands = queue.Queue()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    window = _import_window()
    window._import_active = True
    window._import_model_format = "glb"
    window._import_command_queue = commands
    window._import_pause_requested = False

    window._request_import_pause()

    assert window._import_pause_requested is False
    assert commands.empty()
    assert "only for .obj maps" in logger.warning_messages[-1]


def test_drain_import_queue_handles_paused_import(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    window = _import_window()
    window._has_map_loaded = True
    window._import_active = True
    window._import_is_startup = False
    window._import_queue = queue.Queue()
    window._import_thread = object()
    window._import_process = object()
    window._import_command_queue = object()
    window._import_cache_dir = "/cache/cave"
    window._import_stop_event = viewer_window.threading.Event()
    window._import_pause_requested = True
    window._import_model_format = "obj"
    window._import_queue.put(("paused", "/cache/.cave.resume-123"))

    window._drain_import_queue()

    assert window._import_active is False
    assert window._import_queue is None
    assert window._import_command_queue is None
    assert window._import_pause_requested is False
    assert window._import_model_format is None
    assert window._recording_status_message == "Import paused"
    assert (
        window._recording_status_detail
        == "Resume point saved. Open this map again to continue."
    )
    assert any("Resume checkpoint" in message for message in logger.info_messages)


def test_drain_import_queue_paused_startup_sets_visible_notice(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 100.0)
    window = _import_window()
    window._has_map_loaded = False
    window._import_active = True
    window._import_is_startup = True
    window._import_map_name = "cave.obj"
    window._import_queue = queue.Queue()
    window._import_thread = object()
    window._import_process = object()
    window._import_command_queue = object()
    window._import_cache_dir = "/cache/cave"
    window._import_stop_event = viewer_window.threading.Event()
    window._import_pause_requested = True
    window._import_model_format = "obj"
    window._import_queue.put(("paused", "/cache/.cave.resume-123"))

    window._drain_import_queue()

    assert window._import_active is False
    assert window._import_pause_notice_until == 106.0
    assert window._import_pause_notice_close_after is True
    assert window._import_pause_notice_map_name == "cave.obj"
    assert window._import_pause_notice_title == "Import paused"
    assert window._import_pause_notice_stage == "resume point saved"
    assert (
        window._import_pause_notice_note
        == "This window will close shortly; open this map again to continue."
    )


def test_drain_import_queue_loads_manifest_once_on_render_thread(monkeypatch):
    manifest = {"chunks": {}}
    loaded = []

    monkeypatch.setattr(
        viewer_window.chunker,
        "load_manifest",
        lambda path: loaded.append(("manifest", path)) or manifest,
    )

    window = _import_window()
    window._import_active = True
    window._import_is_startup = False
    window._import_queue = queue.Queue()
    window._import_thread = object()
    window._import_process = object()
    window._import_cache_dir = "/cache/cave"
    window._import_controller.source_dir = "/maps/cave"
    window._import_stop_event = viewer_window.threading.Event()

    def load_new_map(cache_dir, textures_dir, loaded_manifest, **kwargs):
        loaded.append(("load", cache_dir, textures_dir, loaded_manifest, kwargs))

    window.load_new_map = load_new_map
    window._import_queue.put(("done", "/cache/cave", "/textures/cave"))

    window._drain_import_queue()

    assert loaded == [
        ("manifest", "/cache/cave"),
        (
            "load",
            "/cache/cave",
            "/textures/cave",
            manifest,
            {"source_dir": "/maps/cave"},
        ),
    ]
    assert window._import_active is False
    assert window._import_queue is None


def test_load_new_map_remembers_source_dir_instead_of_cache_dir(monkeypatch):
    remembered = []
    calls = []

    from caveviewer.gui import map_history

    monkeypatch.setattr(
        map_history,
        "remember_recent_map_path",
        lambda path: remembered.append(path),
    )

    window = object.__new__(viewer_window.CaveViewerWindow)
    window._teardown_current_map = lambda: calls.append("teardown")
    window._load_map = lambda cache_dir, textures_dir, manifest, **kwargs: calls.append(
        ("load", cache_dir, textures_dir, manifest, kwargs)
    )

    window.load_new_map(
        "/cache/Generated-f566598453a9e673",
        "/cache/Generated-f566598453a9e673",
        {"chunks": {}},
        source_dir="/maps/DevilsEyeGoldLine_resized",
    )

    assert calls == [
        "teardown",
        (
            "load",
            "/cache/Generated-f566598453a9e673",
            "/cache/Generated-f566598453a9e673",
            {"chunks": {}},
            {"map_root": "/maps/DevilsEyeGoldLine_resized"},
        ),
    ]
    assert window._has_map_loaded is True
    assert remembered == ["/maps/DevilsEyeGoldLine_resized"]


def test_drain_import_queue_logs_actionable_error_without_traceback(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(viewer_window, "_LOG", logger)

    window = _import_window()
    window._import_active = True
    window._import_is_startup = False
    window._import_queue = queue.Queue()
    window._import_thread = object()
    window._import_process = object()
    window._import_cache_dir = "/cache/cave"
    window._import_stop_event = viewer_window.threading.Event()
    window._import_queue.put(
        (
            "error",
            "Not enough available system RAM to import 'DevilsEye Start.obj'.",
            "",
            "Close memory-heavy applications and retry.",
        )
    )

    window._drain_import_queue()

    assert logger.error_messages == [
        "Import failed: Not enough available system RAM to import "
        "'DevilsEye Start.obj'.",
        "Suggestion: Close memory-heavy applications and retry.",
    ]
    assert window._import_active is False
    assert window._import_queue is None
    assert not any("traceback" in message.lower() for message in logger.error_messages)


def test_cached_import_worker_does_not_request_desktop_inhibitor(monkeypatch):
    descriptor = {"obj_path": "/maps/cave.obj"}

    monkeypatch.setattr(
        viewer_window,
        "_acquire_map_import_inhibitor",
        lambda _map_name: (_ for _ in ()).throw(
            AssertionError("cached map loads should not inhibit the desktop")
        ),
    )
    monkeypatch.setattr(viewer_window.chunker, "cache_is_valid", lambda _path: True)
    monkeypatch.setattr(viewer_window.chunker, "get_cache_dir", lambda _path: "/cache/cave")
    monkeypatch.setattr(
        cache_paths,
        "map_texture_dir",
        lambda _source_path, _cache_dir, _textures_dir: "/textures/cave",
    )

    window = _import_window()
    window._start_import_async(
        descriptor, "/maps", "cave.obj", is_startup=False
    )
    _wait_for_import_worker(window)

    assert _queued_import_messages(window) == [
        ("progress", "loading cached map", 1.0),
        ("done", "/cache/cave", "/textures/cave"),
    ]


def test_uncached_import_releases_desktop_inhibitor_after_failure(monkeypatch):
    calls = []
    descriptor = {"glb_path": "/maps/broken.glb"}

    def acquire(map_name):
        calls.append(("acquire_inhibitor", map_name))
        return FakeImportInhibitor(calls)

    def start_process(model_descriptor, textures_dir):
        calls.append(("start_process", model_descriptor, textures_dir))
        events = queue.Queue()
        events.put(("error", "parse failed", "traceback text"))
        return SimpleNamespace(process=FakeImportProcess(calls), events=events)

    monkeypatch.setattr(viewer_window, "_acquire_map_import_inhibitor", acquire)
    monkeypatch.setattr(viewer_window.chunker, "cache_is_valid", lambda _path: False)
    monkeypatch.setattr(viewer_window, "start_import_process", start_process)

    window = _import_window()
    window._start_import_async(
        descriptor, "/maps", "broken.glb", is_startup=True
    )
    _wait_for_import_worker(window)

    assert calls == [
        ("acquire_inhibitor", "broken.glb"),
        ("start_process", descriptor, "/maps"),
        ("join_process", 1.0),
        ("close_inhibitor",),
    ]
    assert _queued_import_messages(window) == [
        ("progress", "starting import", 0.0),
        ("error", "parse failed", "traceback text"),
    ]


def test_cancel_active_import_uses_zero_timeout_cleanup_when_relay_is_gone(monkeypatch):
    calls = []
    process = object()
    window = _import_window()
    window._import_stop_event = viewer_window.threading.Event()
    window._import_process = process
    window._import_cache_dir = "/cache/cave"
    window._import_thread = None

    monkeypatch.setattr(
        viewer_window,
        "terminate_import_process",
        lambda process, **kwargs: calls.append((process, kwargs)),
    )

    window._cancel_active_import()

    assert window._import_stop_event.is_set()
    assert calls == [(process, {"timeout": 0.0, "cache_dir": "/cache/cave"})]


def test_cancel_active_import_does_not_wait_for_live_import_thread(monkeypatch):
    calls = []
    import_thread = FakeImportThread(alive=True)
    window = _import_window()
    window._import_stop_event = viewer_window.threading.Event()
    window._import_process = None
    window._import_thread = import_thread

    monkeypatch.setattr(
        viewer_window,
        "terminate_import_process",
        lambda *_args, **_kwargs: calls.append("terminate"),
    )

    window._cancel_active_import()

    assert window._import_stop_event.is_set()
    assert import_thread.join_calls == []
    assert import_thread.is_alive()
    assert calls == []


def test_checkpoint_close_hides_the_native_viewer_before_resource_teardown():
    window = object.__new__(viewer_window.CaveViewerWindow)
    window.wnd = SimpleNamespace(visible=True)

    window._hide_window_before_close()

    assert window.wnd.visible is False


def test_on_close_shutdowns_active_import_before_releasing_resources():
    calls = []

    class FakeImportController:
        active = True

        def request_pause_for_close(self):
            return False

        def shutdown(self):
            calls.append("shutdown_import")
            self.active = False

    window = object.__new__(viewer_window.CaveViewerWindow)
    window.__dict__["_import_controller"] = FakeImportController()
    window._closing_requested = False
    window._has_map_loaded = False
    window._release_window_resources = lambda: calls.append("release_resources")
    window.wnd = SimpleNamespace(
        mouse_exclusivity=True,
        close=lambda: calls.append("close_window"),
    )

    window.on_close()

    assert calls == ["shutdown_import", "release_resources", "close_window"]


def test_on_close_keeps_viewer_open_while_active_trace_is_saved():
    calls = []
    recorder = FakeManualDiveTrace()
    window = _recording_window()
    window._closing_requested = False
    window._has_map_loaded = True
    window._manual_dive_trace = recorder
    window._manual_dive_trace_writers = []
    window.camera = _manual_trace_camera()
    window._reset_transient_input_state = lambda reason: calls.append(
        ("reset_input", reason)
    )
    window._teardown_current_map = lambda **_kwargs: calls.append("teardown")
    window._release_window_resources = lambda: calls.append("release_resources")
    window.wnd = SimpleNamespace(
        mouse_exclusivity=True,
        is_closing=True,
        close=lambda: calls.append("close_window"),
    )

    window.on_close()

    assert not window._closing_requested
    assert window._exit_capture_finalization_active()
    assert window.wnd.is_closing is False
    assert recorder.stopped == [
        (ManualDivePose.from_camera(window.camera), "viewer_closed")
    ]
    assert window._recording_status_message == "Finishing dive trace"
    assert window._recording_status_detail == (
        "Saving the final trace. CaveViewer will close automatically."
    )
    assert calls == [("reset_input", "saving capture before close")]

    window.wnd.is_closing = True
    window.on_close()

    assert window.wnd.is_closing is False
    assert len(recorder.stopped) == 1
    assert calls == [("reset_input", "saving capture before close")]


def test_on_close_keeps_viewer_open_while_active_video_is_saved():
    calls = []
    window = _recording_window()
    window._closing_requested = False
    window._has_map_loaded = True
    window._recording_session = object()
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = []
    window._reset_transient_input_state = lambda reason: calls.append(
        ("reset_input", reason)
    )
    window._stop_recording = lambda: calls.append("stop_recording")
    window._teardown_current_map = lambda **_kwargs: calls.append("teardown")
    window._release_window_resources = lambda: calls.append("release_resources")
    window.wnd = SimpleNamespace(
        mouse_exclusivity=True,
        is_closing=True,
        close=lambda: calls.append("close_window"),
    )

    window.on_close()

    assert not window._closing_requested
    assert window._exit_capture_finalization_active()
    assert window.wnd.is_closing is False
    assert window._recording_status_message == "Finishing video"
    assert window._recording_status_detail == (
        "Saving the last frames. CaveViewer will close automatically."
    )
    assert calls == [
        ("reset_input", "saving capture before close"),
        "stop_recording",
    ]


def test_on_close_does_not_replace_escape_cancellation_with_save_finalization():
    window = _recording_window()
    window._closing_requested = False
    window._recording_session = object()
    window._show_capture_status(
        "Canceling video…",
        "Stopping capture and removing partial files. "
        "CaveViewer will close automatically.",
        duration=None,
    )
    window._ensure_capture_workflow().begin_escape_cancellation()
    window.wnd = SimpleNamespace(is_closing=True)

    window.on_close()

    assert window.wnd.is_closing is False
    assert window._escape_capture_cancellation_active()
    assert not window._exit_capture_finalization_active()
    assert window._recording_status_message == "Canceling video…"


def test_escape_cancellation_closes_only_after_cleanup_and_confirmation(monkeypatch):
    current_time = [14.999]
    monkeypatch.setattr(
        viewer_window.time,
        "perf_counter",
        lambda: current_time[0],
    )
    calls = []
    window = _recording_window()
    workflow = window._ensure_capture_workflow()
    workflow.begin_escape_cancellation()
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = []
    window._recording_status_until = 15.0
    window._complete_window_close = lambda: calls.append("close")

    assert window._complete_escape_capture_cancellation_if_ready() is False

    window._recording_stop_thread = object()
    current_time[0] = 20.0
    assert window._complete_escape_capture_cancellation_if_ready() is False

    window._recording_stop_thread = None
    assert window._complete_escape_capture_cancellation_if_ready() is True
    assert calls == ["close"]
    assert not workflow.escape_cancellation_active


def test_escape_cancellation_uses_the_finalizing_capture_frame_phase():
    window = _recording_window()
    window._window_setup_complete = True
    window._closing_requested = False
    window._is_iconified = False
    window._import_active = False
    window._has_map_loaded = True
    window._ensure_capture_workflow().begin_escape_cancellation()

    assert window._frame_phase() is viewer_window.ViewerFramePhase.FINALIZING_CAPTURE


def test_exit_capture_finalization_waits_until_status_is_visible(monkeypatch):
    monkeypatch.setattr(viewer_window.time, "perf_counter", lambda: 10.0)
    calls = []
    window = _recording_window()
    window._closing_requested = False
    workflow = _begin_exit_capture_finalization(window)
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = []
    window._complete_window_close = lambda: calls.append("close")

    assert window._complete_exit_capture_finalization_if_ready() is False

    workflow.mark_exit_status_presented(now=9.5)
    assert window._complete_exit_capture_finalization_if_ready() is False

    workflow.exit_status_presented_at = 9.25
    assert window._complete_exit_capture_finalization_if_ready() is True
    assert calls == ["close"]


def test_exit_finalization_keeps_its_trace_status_and_does_not_reveal_files(tmp_path):
    output_path = tmp_path / "trace.jsonl"
    output_path.write_text('{"record": "trace_completed"}\n', encoding="utf-8")
    recorder = FakeManualDiveTrace()
    recorder.result = ManualDiveTraceResult(
        output_path=str(output_path),
        partial_path=str(tmp_path / ".trace.jsonl.part"),
        completed=True,
        error=None,
    )
    window = _recording_window()
    _begin_exit_capture_finalization(window)
    window._manual_dive_trace = None
    window._manual_dive_trace_writers = [_pending_manual_trace_writer(recorder)]
    window._show_capture_status(
        "Finishing dive trace",
        "Saving the final trace. CaveViewer will close automatically.",
        duration=None,
        now=10.0,
    )

    window._update_manual_dive_trace(now=10.0)

    assert window._manual_dive_trace_writers == []
    assert window._recording_status_message == "Finishing dive trace"
    assert window._recording_status_detail == (
        "Saving the final trace. CaveViewer will close automatically."
    )
    assert window._ensure_artifact_capture_presentation().take_due_reveals(now=20.0) == ()
