"""OpenGL viewer-window lifecycle and render-loop orchestration.

The actual OpenGL window: owns the moderngl context, the free-fly camera,
the StreamingWorld (which decides what to load/unload), and the per-chunk
GPU buffers/textures. This is where the rest of caveviewer.core and caveviewer.gui
gets wired together into a runnable program.

Each loaded chunk becomes a small set of moderngl VAOs, one per material
group within that chunk (so each can be drawn with its own bound texture).
We keep a dict: cell -> list[(vao, texture_material_name)] so unload is a
simple lookup-and-release.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import logging
import math
import os
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window.context.base import KeyModifiers

from caveviewer.branding import BrandingAssets, resolve_branding_assets
from caveviewer.core.chunking import builder as chunker
from caveviewer.core.map import slicing as map_slicing
from caveviewer.core.hardware import gpu_memory, memory_targets, system_memory
from caveviewer.core.diagnostics.logging import get_logger
from caveviewer.core.diagnostics.runtime import (
    record_runtime_exception,
    record_runtime_stage,
)
from caveviewer.core.preferences.runtime_settings import (
    RuntimeSettings,
    ViewerRuntimeSettings,
)
from caveviewer.core.streaming.world import StreamingWorld, StreamingConfig
from caveviewer.gui.chunk_upload import ChunkUploadManager
from caveviewer.gui.recording_capture import RecordingCaptureResources
from caveviewer.gui.texture_manager import TextureManager
from caveviewer.gui.camera import FlyCamera
from caveviewer.gui.minimap import Minimap
from caveviewer.gui.render_mode_buttons import RenderModeButtons
from caveviewer.gui.controls_overlay import ControlsOverlay
from caveviewer.gui.stepper_control import StepperControl
from caveviewer.gui.color_picker import ColorPicker
from caveviewer.gui.import_progress_panel import ImportProgressPanel
from caveviewer.benchmarking.results import BenchmarkController
from caveviewer.gui.import_process import (
    start_import_process,
    terminate_import_process,
)
from caveviewer.gui.import_controller import MapImportController
from caveviewer.gui.map_opening import pick_folder_dialog, resolve_selected_map_folder
from caveviewer.gui.map_opening_progress import (
    MapOpeningProgressFrame,
    MapOpeningProgressSession,
)
from caveviewer.gui import recording
from caveviewer.gui import bitmap_font
from caveviewer.gui import manual_dive_trace
from caveviewer.gui.artifact_capture_controller import (
    ArtifactCapturePresentationController,
    ArtifactCaptureStatus,
)
from caveviewer.gui.manual_dive_trace_controller import ManualDiveTraceStateController
from caveviewer.gui.slice_selection_controller import SliceAnchors, SliceSelectionController
from caveviewer.gui.slice_export_controller import (
    SliceExportCanceled,
    SliceExportController,
    SliceExportFailed,
    SliceExportSucceeded,
)
from caveviewer.gui import recorded_dive
from caveviewer.gui import render_upload
from caveviewer.gui import view_culling
from caveviewer.gui import viewer_input
from caveviewer.gui import viewer_bookmarks
from caveviewer.gui.recording_controller import RecordingStateController
from caveviewer.gui.viewer_action_dispatch import (
    ViewerActionDispatcher,
    ViewerKeyPressActions,
)
from caveviewer.gui.viewer_capture_workflow import (
    CaptureOwner,
    CaptureOverlayMode,
    CaptureOverlayState,
    CaptureOwnershipState,
    ViewerCaptureWorkflow,
)
from caveviewer.gui.viewer_frame_scheduler import (
    ViewerFramePhase,
    ViewerFrameScheduler,
    ViewerFrameState,
)
from caveviewer.gui.viewer_session import (
    PendingImportRequest,
    ViewerBenchmarkConfig,
    ViewerLaunchMode,
    ViewerSession,
    ViewerSessionConfig,
    ViewerSessionOutcome,
)
from caveviewer.gui.viewer_workflow import (
    ViewerRenderRequest,
    ViewerWorkflowCoordinator,
    ViewerWorkflowSnapshot,
)
from caveviewer.gui.viewer_benchmark_composition import (
    environment_size as _benchmark_environment_size,
    streaming_settings_fingerprint as _benchmark_streaming_settings_fingerprint,
    streaming_settings_snapshot as _benchmark_streaming_settings_snapshot,
)
from caveviewer.gui.platform.presentation import (
    PresentationProfile,
    get_presentation_profile,
)
from caveviewer.gui.platform.presentation_actions import (
    PresentationActionsAdapter,
    create_presentation_actions_adapter,
)
from caveviewer.gui.platform.probes.recording import VideoRecordingTarget
from caveviewer.gui.platform.saved_artifact_reveal import (
    SavedArtifactRevealAdapter,
    create_saved_artifact_reveal_adapter,
)
from caveviewer.gui.platform.recording_process import (
    RecordingProcessAdapter,
    create_recording_process_adapter,
)
from caveviewer.gui.platform.recording_preflight import video_recording_preflight
from caveviewer.gui.platform.desktop_inhibition import (
    acquire_idle_suspend_inhibitor,
    release_desktop_inhibitor,
)
from caveviewer.gui.platform import DesktopServiceError, tk_root_options
from caveviewer.gui.platform.viewer_launch import (
    authorized_viewer_launch_target,
    viewer_launch_preflight,
)
from caveviewer.gui.platform.window_backend import (
    ViewerWindowLaunchRequest,
    WindowBackendAdapter,
    create_window_backend_adapter,
)
from caveviewer.resources import resource_path
from caveviewer.version import APP_NAME, APP_VERSION

if TYPE_CHECKING:
    from caveviewer.gui.platform.runtime import PlatformRuntime, VideoRecordingPreflight

_LOG = get_logger("CaveViewer")

_DEFAULT_WINDOW_SIZE = (1600, 1000)
_DESKTOP_WINDOW_SCALE = 0.80
_VIEWER_UI_BASE_WINDOW_SIZE = (1536, 864)
_UI_TEXT_SCALE_ENV = "CAVEVIEWER_UI_TEXT_SCALE"
_VIEWER_UI_SCALE_ENV = "CAVEVIEWER_VIEWER_UI_SCALE"
_VIEWER_UI_SCALE_MAX = 1.45
_TEXTURE_RESIDENT_CACHE_MB_ENV = "CAVEVIEWER_TEXTURE_RESIDENT_CACHE_MB"
_GPU_RESIDENCY_SAFETY_SHARE = 0.05
_RENDER_UPLOAD_INITIAL_SLICE_BYTES = render_upload.RENDER_UPLOAD_INITIAL_SLICE_BYTES
_CATCHUP_UPLOAD_CHUNKS_PER_FRAME = 2
_CATCHUP_UPLOAD_OPERATIONS_PER_CHUNK = 8
_CATCHUP_UPLOAD_TIME_BUDGET_MS = 8.0
_STARTUP_UPLOAD_CHUNKS_PER_FRAME = 4
_STARTUP_UPLOAD_OPERATIONS_PER_CHUNK = 8
_STARTUP_UPLOAD_TIME_BUDGET_MS = 12.0
_VIEWER_STREAMING_SHUTDOWN_TIMEOUT_SECONDS = 2.0
_ICONIFIED_RENDER_POLL_INTERVAL_S = 0.12
_IMPORT_PAUSE_NOTICE_RENDER_INTERVAL_S = 1.0 / 30.0
_MAIN_THREAD_STALL_LOG_THRESHOLD_S = 0.5
_MAIN_THREAD_STALL_LOG_MIN_INTERVAL_S = 2.0
_RECORDED_DIVE_LOOKAHEAD_SECONDS = 10.0
_RECORDED_DIVE_PREFETCH_RADIUS_CELLS = 1
_RECORDED_DIVE_PREFETCH_CELL_CAP = 256
_RecordingStopResult = recording.RecordingStopResult
_RecordingReadbackSlot = recording.RecordingReadbackSlot


@dataclass(frozen=True)
class _PendingManualDiveTraceWriter:
    """One trace writer paired with its user-visible completion policy."""

    recorder: manual_dive_trace.ManualDiveTraceRecorder
    show_completion: bool
    reveal_on_success: bool


def _import_controller_property(attribute_name: str):
    def getter(self):
        return getattr(self._ensure_import_controller(), attribute_name)

    def setter(self, value) -> None:
        setattr(self._ensure_import_controller(), attribute_name, value)

    return property(getter, setter)


def _tk_root_exists(root) -> bool:
    """Return whether a Tk root-like object is still usable."""
    if root is None:
        return False
    try:
        return bool(root.winfo_exists())
    except Exception:
        return False


def _screen_size_from_tk_root(root) -> tuple[int, int] | None:
    """Read a positive desktop size from a Tk root-like object."""
    try:
        desktop_width = int(root.winfo_screenwidth())
        desktop_height = int(root.winfo_screenheight())
    except Exception:
        return None
    if desktop_width <= 0 or desktop_height <= 0:
        return None
    return desktop_width, desktop_height


def _window_size_from_desktop_size(desktop_size: tuple[int, int]) -> tuple[int, int]:
    """Return CaveViewer's default viewer size for a detected desktop."""
    desktop_width, desktop_height = desktop_size
    if desktop_width <= 0 or desktop_height <= 0:
        return _DEFAULT_WINDOW_SIZE

    window_size = (
        max(1, int(round(desktop_width * _DESKTOP_WINDOW_SCALE))),
        max(1, int(round(desktop_height * _DESKTOP_WINDOW_SCALE))),
    )
    _LOG.info(
        "Desktop size %dx%d; opening viewer at %dx%d.",
        desktop_width, desktop_height, *window_size,
    )
    return window_size


def _desktop_relative_window_size(screen_source=None) -> tuple[int, int]:
    """
    Return an 80%-of-screen fallback for non-GLFW desktop backends.

    When a Tk root already exists, reuse it for screen-size measurement instead
    of creating a second Tk application root.  This matters most on macOS,
    where the kept-alive splash root owns process-level Tk app/menu state.
    """
    if screen_source is not None:
        screen_size = _screen_size_from_tk_root(screen_source)
        if screen_size is None:
            _LOG.warning(
                "Could not detect desktop size from existing Tk root; using %dx%d.",
                *_DEFAULT_WINDOW_SIZE,
            )
            return _DEFAULT_WINDOW_SIZE
        return _window_size_from_desktop_size(screen_size)

    root = None
    owns_root = False
    try:
        import tkinter as tk

        default_root = getattr(tk, "_default_root", None)
        if _tk_root_exists(default_root):
            screen_size = _screen_size_from_tk_root(default_root)
            if screen_size is None:
                _LOG.warning(
                    "Could not detect desktop size from existing Tk root; using %dx%d.",
                    *_DEFAULT_WINDOW_SIZE,
                )
                return _DEFAULT_WINDOW_SIZE
            return _window_size_from_desktop_size(screen_size)

        root = tk.Tk(**tk_root_options())
        owns_root = True
        root.withdraw()
        screen_size = _screen_size_from_tk_root(root)
        if screen_size is None:
            return _DEFAULT_WINDOW_SIZE
        return _window_size_from_desktop_size(screen_size)
    except Exception as e:
        _LOG.warning("Could not detect desktop size (%s); using %dx%d.", e, *_DEFAULT_WINDOW_SIZE)
        return _DEFAULT_WINDOW_SIZE
    finally:
        if owns_root and root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _window_pixel_ratio(window) -> float:
    """Return framebuffer pixels per logical window pixel for crisp UI text."""
    try:
        width, height = window.size
        buffer_width, buffer_height = window.buffer_size
        width = max(1, int(width))
        height = max(1, int(height))
        return max(1.0, min(4.0, max(buffer_width / width, buffer_height / height)))
    except Exception:
        return 1.0


def _viewer_overlay_text_scale(
    presentation_profile: PresentationProfile,
    base_scale: float,
    environ: Mapping[str, str] | None = None,
    *,
    configured_scale: float | None = None,
) -> float:
    """Return the startup text scale for FreeType-rendered viewer overlays."""
    if configured_scale is not None:
        return float(configured_scale)
    environment = os.environ if environ is None else environ
    raw_override = str(environment.get(_UI_TEXT_SCALE_ENV, "")).strip()
    if raw_override:
        try:
            return float(raw_override)
        except ValueError:
            pass
    return presentation_profile.viewer_overlay_text_scale(float(base_scale))


def _viewer_ui_surface_size(
    window,
    fallback_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Return the framebuffer-aware surface size used for HUD auto-scaling."""
    fallback = fallback_size or _DEFAULT_WINDOW_SIZE
    try:
        buffer_width, buffer_height = window.buffer_size
        buffer_width = int(buffer_width)
        buffer_height = int(buffer_height)
        if buffer_width > 0 and buffer_height > 0:
            return buffer_width, buffer_height
    except Exception:
        pass
    try:
        width, height = window.size
        width = int(width)
        height = int(height)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return fallback


def _viewer_ui_scale_for_window_size(
    window_size: tuple[int, int] | None,
    environ: Mapping[str, str] | None = None,
    *,
    configured_scale: float | None = None,
) -> float:
    """Return an automatic HUD scale for the current viewer surface.

    The control overlay is rendered inside OpenGL, so it does not inherit GNOME
    titlebar or XWayland desktop scaling.  Keep the old compact size at the
    1536x864 default viewer window, then grow the HUD on larger viewer surfaces.
    The environment override is for development/testing; the normal user path
    is automatic.
    """
    if configured_scale is not None:
        try:
            return max(0.75, min(2.0, float(configured_scale)))
        except (TypeError, ValueError):
            pass
    else:
        environment = os.environ if environ is None else environ
        raw_override = str(environment.get(_VIEWER_UI_SCALE_ENV, "")).strip()
        if raw_override:
            try:
                return max(0.75, min(2.0, float(raw_override)))
            except ValueError:
                pass

    try:
        width, height = window_size or _DEFAULT_WINDOW_SIZE
        width = max(1, int(width))
        height = max(1, int(height))
    except Exception:
        width, height = _DEFAULT_WINDOW_SIZE

    base_width, base_height = _VIEWER_UI_BASE_WINDOW_SIZE
    size_scale = min(width / base_width, height / base_height)
    return max(1.0, min(_VIEWER_UI_SCALE_MAX, size_scale))


def _map_import_inhibit_reason(map_name: str) -> str:
    """Return the desktop-visible reason used while importing a map."""
    display_name = str(map_name or "").strip() or "map"
    return f"Importing {display_name}"


def _acquire_map_import_inhibitor(
    map_name: str,
    *,
    desktop_services=None,
    platform_runtime: PlatformRuntime | None = None,
):
    """Best-effort desktop idle/suspend inhibitor for long map imports."""
    try:
        if desktop_services is None:
            from caveviewer.gui.platform import get_desktop_services

            desktop_services = get_desktop_services()
        return acquire_idle_suspend_inhibitor(
            desktop_services,
            _map_import_inhibit_reason(map_name),
            platform_runtime=platform_runtime,
        )
    except Exception as exc:
        # Legacy desktop-service construction must not block opening maps. The
        # typed acquisition boundary already handles capability and action
        # failures as no-ops once a service exists.
        _LOG.debug(
            "Desktop idle/suspend inhibitor setup skipped: error_type=%s",
            type(exc).__name__,
        )
        return None


def _release_desktop_inhibitor(inhibitor) -> None:
    """Release a desktop inhibitor without affecting import completion."""
    release_desktop_inhibitor(inhibitor)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, float(raw)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_optional_mebibytes(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        _LOG.warning("Ignoring invalid %s=%r; expected a positive MB value.", name, raw)
        return None
    if not math.isfinite(value) or value <= 0.0:
        _LOG.warning("Ignoring invalid %s=%r; expected a positive MB value.", name, raw)
        return None
    return max(1, int(value * 1024 ** 2))


def _map_initial_camera_position(manifest: Mapping[str, Any]) -> np.ndarray:
    """Return a slice entry point or the ordinary render-cache start."""
    slice_metadata = manifest.get(map_slicing.SLICE_MANIFEST_KEY)
    if isinstance(slice_metadata, Mapping):
        try:
            entry_position = np.asarray(
                tuple(float(value) for value in slice_metadata["entry_position"]),
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            entry_position = np.empty(0, dtype=np.float64)
        if entry_position.shape == (3,) and np.isfinite(entry_position).all():
            return entry_position
    position = chunker.first_manifest_chunk_center(manifest.get("chunks"))
    if position is None:
        raise ValueError("map manifest does not contain a valid starting chunk")
    return np.asarray(position, dtype=np.float64)


SHADER_DIR = str(resource_path("shaders"))


def _presentation_profile_for_runtime(
    platform_runtime: PlatformRuntime | None = None,
) -> PresentationProfile:
    """Use the injected static UI profile, with a pure legacy fallback."""
    profile = (
        getattr(platform_runtime, "presentation_profile", None)
        if platform_runtime is not None
        else None
    )
    return profile or get_presentation_profile()


def _presentation_actions_adapter_for_runtime(
    platform_runtime: PlatformRuntime | None = None,
) -> PresentationActionsAdapter:
    """Use injected native presentation actions or a direct fallback."""
    actions = (
        getattr(platform_runtime, "presentation_actions_adapter", None)
        if platform_runtime is not None
        else None
    )
    if actions is not None:
        return actions
    return create_presentation_actions_adapter()


def _window_backend_adapter_for_runtime(
    platform_runtime: PlatformRuntime | None = None,
) -> WindowBackendAdapter:
    """Use the injected viewer-window executor with a legacy direct fallback."""
    if platform_runtime is not None:
        adapter = getattr(platform_runtime, "window_backend_adapter", None)
        if adapter is not None:
            return adapter
    return create_window_backend_adapter()


def _saved_artifact_reveal_adapter_for_runtime(
    platform_runtime: PlatformRuntime | None = None,
) -> SavedArtifactRevealAdapter:
    """Use the injected artifact-reveal action or direct platform fallback."""
    if platform_runtime is not None:
        return platform_runtime.saved_artifact_reveal_adapter
    return create_saved_artifact_reveal_adapter()


def _recording_process_adapter_for_runtime(
    platform_runtime: PlatformRuntime | None = None,
) -> RecordingProcessAdapter:
    """Use the injected process adapter or direct platform fallback."""
    if platform_runtime is not None:
        return platform_runtime.recording_process_adapter
    return create_recording_process_adapter()


def _branding_assets_for_runtime(
    platform_runtime: PlatformRuntime | None,
) -> BrandingAssets:
    """Return the process snapshot, preserving direct viewer test callers."""
    assets = (
        getattr(platform_runtime, "branding_assets", None)
        if platform_runtime is not None
        else None
    )
    return assets or resolve_branding_assets(environ={})


def _runtime_app_icon_path(platform_runtime: PlatformRuntime | None) -> str:
    assets = _branding_assets_for_runtime(platform_runtime)
    platform_name = (
        platform_runtime.profile.platform_name
        if platform_runtime is not None
        else get_presentation_profile().platform_name
    )
    return str(assets.application_icon_for(platform_name))


_UI_PANEL_VERT_SRC = """
#version 330
in vec2 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_color = in_color;
}
"""

_UI_PANEL_FRAG_SRC = """
#version 330
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = v_color;
}
"""


class CaveViewerWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = APP_NAME
    # The launch helpers replace this fallback with an 80%-of-desktop size.
    # Keep aspect_ratio unlocked so manual resizing remains fully flexible.
    window_size = _DEFAULT_WINDOW_SIZE
    resizable = True
    # The launch helpers set this from the immutable runtime snapshot. Direct
    # legacy callers retain an environment-backed fallback at launch time.
    vsync = True
    # Apply hardware anti-aliasing before compositing the cave scene and every
    # OpenGL HUD overlay into the default presentation framebuffer.
    samples = 4
    aspect_ratio = None  # don't letterbox; we recompute from actual window size

    # Global UI text scale for all bitmap_font-rendered labels. This is
    # intentionally configured here so font sizing can be adjusted from
    # one place instead of tuning every overlay module individually.
    UI_TEXT_SCALE = 1.28

    # Shared backplate behind the always-visible right-side HUD controls.
    # This keeps section labels readable over bright cave surfaces without
    # adding a separate background to every individual widget.
    RIGHT_COLUMN_PANEL_SIDE_PAD = 10
    RIGHT_COLUMN_PANEL_TOP_PAD = 8
    RIGHT_COLUMN_PANEL_BOTTOM_PAD = 10
    RIGHT_COLUMN_PANEL_RIGHT_MARGIN = 16
    RIGHT_COLUMN_PANEL_BOTTOM_MARGIN = 16
    RIGHT_COLUMN_PANEL_LABEL_GAP = 8
    RIGHT_COLUMN_PANEL_SCALE = 0.76
    RIGHT_COLUMN_PANEL_TEXT_SCALE = 0.84
    RIGHT_COLUMN_PANEL_LABEL_TEXT_SCALE = 0.98
    RIGHT_COLUMN_PANEL_BUTTON_TEXT_SCALE = 0.70
    RIGHT_COLUMN_PANEL_TEXT_MAX_UI_SCALE = 1.0
    RIGHT_COLUMN_PANEL_MAX_UI_SCALE = _VIEWER_UI_SCALE_MAX
    RIGHT_COLUMN_PANEL_FILL_RGBA = (0.09, 0.12, 0.16, 0.84)
    RIGHT_COLUMN_PANEL_BORDER_RGBA = (0.42, 0.54, 0.72, 0.62)
    RIGHT_COLUMN_PANEL_BORDER_PX = 1.5
    RECORDING_COUNTDOWN_START_NUMBER = 3
    RECORDING_COUNTDOWN_TITLE = "Prepare to record a dive"
    MANUAL_DIVE_TRACE_COUNTDOWN_START_NUMBER = 3
    MANUAL_DIVE_TRACE_COUNTDOWN_TITLE = "Prepare to plan a dive"
    SLICE_COUNTDOWN_START_NUMBER = 3
    SLICE_COUNTDOWN_TITLE = "Prepare to slice a cave"
    SLICE_PADDING = map_slicing.DEFAULT_SLICE_PADDING
    DIVE_STATUS_PANEL_WIDTH_FRACTION = 0.52
    DIVE_STATUS_PANEL_MIN_WIDTH = 520.0
    DIVE_STATUS_PANEL_HEIGHT = 116.0
    DIVE_STATUS_TITLE_PIXEL_SIZE = 2.50
    DIVE_STATUS_NOTE_PIXEL_SIZE = 1.70
    RECORDING_READBACK_BUFFER_COUNT = 3
    RECORDING_READBACK_COMPONENTS = 3
    RECORDING_RAW_PIX_FMT = "rgb24"

    # Startup focus forcing can make bundled macOS app windows appear in a
    # corner first and then jump as the window manager re-places them.
    # Default to disabled for frozen macOS builds; allow override.
    FORCE_STARTUP_FOCUS_ENV = "CAVEVIEWER_FORCE_STARTUP_FOCUS"

    _import_active = _import_controller_property("active")
    _import_is_startup = _import_controller_property("is_startup")
    _import_thread = _import_controller_property("thread")
    _import_process = _import_controller_property("process")
    _import_command_queue = _import_controller_property("command_queue")
    _import_cache_dir = _import_controller_property("cache_dir")
    _import_stop_event = _import_controller_property("stop_event")
    _import_queue = _import_controller_property("event_queue")
    _import_pause_requested = _import_controller_property("pause_requested")
    _import_model_format = _import_controller_property("model_format")
    _import_map_name = _import_controller_property("map_name")
    _import_progress_stage = _import_controller_property("progress_stage")
    _import_progress_fraction = _import_controller_property("progress_fraction")
    _import_progress_title = _import_controller_property("progress_title")
    _import_progress_note = _import_controller_property("progress_note")
    _import_resuming_from_checkpoint = _import_controller_property(
        "resuming_from_checkpoint"
    )
    _import_pause_notice_until = _import_controller_property("pause_notice_until")
    _import_pause_notice_close_after = _import_controller_property(
        "pause_notice_close_after"
    )
    _import_pause_notice_map_name = _import_controller_property("pause_notice_map_name")
    _import_pause_notice_title = _import_controller_property("pause_notice_title")
    _import_pause_notice_stage = _import_controller_property("pause_notice_stage")
    _import_pause_notice_note = _import_controller_property("pause_notice_note")

    def __init__(self, **kwargs):
        session = getattr(type(self), "_viewer_session", None)
        if not isinstance(session, ViewerSession):
            raise RuntimeError(
                "CaveViewerWindow requires a session-bound configuration class"
            )
        self._viewer_session = session
        session_config = session.config
        record_runtime_stage(
            "viewer_config_initialization_begin",
            requested_window_size=getattr(type(self), "window_size", None),
        )
        try:
            super().__init__(**kwargs)
        except BaseException as error:
            record_runtime_exception(
                "viewer_config_initialization_failed",
                error,
            )
            raise
        # moderngl-window closes its default Escape key before forwarding the
        # key callback. CaveViewer owns Escape so capture discard can finish
        # and present its result before the backend window is allowed to close.
        self._claim_backend_escape_key()
        # Pyglet's default close event destroys its native window after the
        # callback returns.  CaveViewer sometimes needs to defer that close
        # briefly (for example, while an OBJ import saves a resume point), so
        # claim the event before moderngl-window's forwarding handler runs.
        self._claim_backend_close_event()
        record_runtime_stage(
            "viewer_config_context_ready",
            context_version=getattr(getattr(self, "ctx", None), "version_code", None),
            window_backend=type(getattr(self, "wnd", None)).__name__,
        )
        self._window_setup_complete = False
        self._platform_runtime = session_config.platform_runtime
        self._branding_assets = _branding_assets_for_runtime(self._platform_runtime)
        self._runtime_settings = (
            session_config.runtime_settings
            or getattr(self._platform_runtime, "runtime_settings", None)
        )
        self._viewer_runtime_settings: ViewerRuntimeSettings | None = (
            self._runtime_settings.viewer_configuration()
            if self._runtime_settings is not None
            else None
        )
        self._presentation_profile = _presentation_profile_for_runtime(
            self._platform_runtime
        )
        self._presentation_actions_adapter = _presentation_actions_adapter_for_runtime(
            self._platform_runtime,
        )
        self._set_runtime_window_icon()

        if self._viewer_runtime_settings is None:
            force_focus_env = os.getenv(self.FORCE_STARTUP_FOCUS_ENV, "").strip().lower()
            force_focus = force_focus_env in {"1", "true", "yes", "on"}
        else:
            force_focus = self._viewer_runtime_settings.force_startup_focus
        self._startup_focus_enabled = True
        if self._presentation_profile.suppress_forced_startup_focus(
            is_frozen=bool(getattr(sys, "frozen", False)),
            force_requested=force_focus,
        ):
            self._startup_focus_enabled = False

        bitmap_font.set_presentation_profile(self._presentation_profile)
        if self._viewer_runtime_settings is None:
            bitmap_font.clear_runtime_style()
        else:
            bitmap_font.configure_runtime_style(
                font_path=self._viewer_runtime_settings.ui_font,
                antialiasing_mode=self._viewer_runtime_settings.text_antialiasing_mode,
            )
        bitmap_font.set_text_scale(
            _viewer_overlay_text_scale(
                self._presentation_profile,
                self.UI_TEXT_SCALE,
                environ={} if self._viewer_runtime_settings is not None else None,
                configured_scale=(
                    self._viewer_runtime_settings.ui_text_scale_override
                    if self._viewer_runtime_settings is not None
                    else None
                ),
            )
        )
        bitmap_font.set_raster_scale(_window_pixel_ratio(getattr(self, "wnd", None)))
        self._viewer_ui_scale = _viewer_ui_scale_for_window_size(
            _viewer_ui_surface_size(getattr(self, "wnd", None), _DEFAULT_WINDOW_SIZE),
            environ={} if self._viewer_runtime_settings is not None else None,
            configured_scale=(
                self._viewer_runtime_settings.viewer_ui_scale
                if self._viewer_runtime_settings is not None
                else None
            ),
        )
        self._right_column_panel_scale = (
            self.RIGHT_COLUMN_PANEL_SCALE * self._viewer_ui_scale
        )
        self._right_column_panel_text_scale = (
            self.RIGHT_COLUMN_PANEL_TEXT_SCALE
            * min(self._viewer_ui_scale, self.RIGHT_COLUMN_PANEL_TEXT_MAX_UI_SCALE)
        )
        self._right_column_panel_label_text_scale = (
            self.RIGHT_COLUMN_PANEL_LABEL_TEXT_SCALE
            * min(self._viewer_ui_scale, self.RIGHT_COLUMN_PANEL_TEXT_MAX_UI_SCALE)
        )
        self._right_column_panel_button_text_scale = (
            self.RIGHT_COLUMN_PANEL_BUTTON_TEXT_SCALE
            * min(self._viewer_ui_scale, self.RIGHT_COLUMN_PANEL_TEXT_MAX_UI_SCALE)
        )

        have_ready_cache = session_config.cache_dir is not None
        have_pending_import = session_config.pending_import is not None

        if not have_ready_cache and not have_pending_import:
            raise RuntimeError(
                "The viewer session has neither a ready cache nor a pending import."
            )

        self._workflow_coordinator = ViewerWorkflowCoordinator(session)
        self.import_progress_panel = None
        self._pending_import_splash_rendered = False
        if have_pending_import:
            self.import_progress_panel = ImportProgressPanel(
                self.ctx,
                branding_assets=self._branding_assets,
            )
            self._pending_import_splash_rendered = (
                self._present_pending_import_splash_now()
            )

        with open(os.path.join(SHADER_DIR, "mesh.vert")) as f:
            vert_src = f.read()
        with open(os.path.join(SHADER_DIR, "mesh.frag")) as f:
            frag_src = f.read()
        self.program = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        # u_model is always the identity matrix -- write it once here rather than
        # allocating and re-uploading a fresh identity matrix every frame.
        self.program["u_model"].write(np.identity(4, dtype=np.float32).tobytes())

        self._hud_panel_program = self.ctx.program(
            vertex_shader=_UI_PANEL_VERT_SRC,
            fragment_shader=_UI_PANEL_FRAG_SRC,
        )
        self._hud_panel_vbo = self.ctx.buffer(reserve=64 * 6 * 4)
        self._hud_panel_vao = self.ctx.vertex_array(
            self._hud_panel_program,
            [(self._hud_panel_vbo, "2f 4f", "in_pos", "in_color")],
        )
        self._status_panel_max_verts = 12000
        self._status_panel_vbo = self.ctx.buffer(reserve=self._status_panel_max_verts * 6 * 4)
        self._status_panel_vao = self.ctx.vertex_array(
            self._hud_panel_program,
            [(self._status_panel_vbo, "2f 4f", "in_pos", "in_color")],
        )

        self._keys_down = set()
        self._last_raw_modifiers = 0
        self._mouse_look_active = False
        self._mouse_look_left_option_active = False
        self._last_mouse_pos = None
        self._frame_count = 0
        self._last_fps_print = time.time()
        self._frame_active_time_s = 0.0
        self._frame_time_history: list[float] = []
        self._last_gpu_draw_ms: float | None = None
        viewer_settings = self._viewer_runtime_settings
        self._gpu_draw_timer_enabled = (
            _env_bool("CAVEVIEWER_GPU_DRAW_TIMER", False)
            if viewer_settings is None
            else viewer_settings.gpu_draw_timer
        )
        self._streaming_frame_timing: dict | None = None
        self._last_input_reset_log = 0.0
        self._layout_cache_size: tuple | None = None
        self._layout_cache_result: dict | None = None
        self._is_iconified = False
        self._is_background_paused = False
        self._closing_requested = False
        self._slice_reveal_before_close = False
        self._slice_reveal_output_path: str | None = None
        self._slice_source_cache_dir: str | None = None
        self._slice_storage_parent: str | None = None
        self._slice_display_base: str | None = None
        self._slice_root_cave_name: str | None = None
        self._startup_focus_requested = False
        self._upload_chunks_per_frame = (
            _env_int("CAVEVIEWER_UPLOAD_CHUNKS_PER_FRAME", 1, 1, 16)
            if viewer_settings is None
            else viewer_settings.streaming.upload_chunks_per_frame
        )
        self._upload_groups_per_frame = (
            _env_int("CAVEVIEWER_UPLOAD_GROUPS_PER_FRAME", 1, 1, 64)
            if viewer_settings is None
            else viewer_settings.streaming.upload_groups_per_frame
        )
        self._upload_time_budget_ms = (
            _env_float("CAVEVIEWER_UPLOAD_TIME_BUDGET_MS", 3.0, 0.5, 50.0)
            if viewer_settings is None
            else viewer_settings.streaming.upload_time_budget_ms
        )
        self._current_upload_operations_per_chunk = self._upload_groups_per_frame
        self._current_upload_time_budget_ms = self._upload_time_budget_ms
        self._vbo_upload_slice_bytes = _RENDER_UPLOAD_INITIAL_SLICE_BYTES
        self._texture_upload_slice_bytes = _RENDER_UPLOAD_INITIAL_SLICE_BYTES
        self._bookmarks_path: str | None = None
        self._bookmarks: viewer_bookmarks.BookmarkSlots = {}
        self._manual_dive_trace: (
            manual_dive_trace.ManualDiveTraceRecorder | None
        ) = None
        self._manual_dive_trace_writers: list[_PendingManualDiveTraceWriter] = []
        self._pending_recorded_dive_trace = (
            session_config.recorded_dive_trace
        )
        self._recorded_dive_trace: recorded_dive.RecordedDiveTrace | None = None
        self._recorded_dive_controller: (
            recorded_dive.RecordedDivePlaybackController | None
        ) = None
        self._recorded_dive_prefetch_cell_set: frozenset[
            tuple[int, int, int]
        ] = frozenset()
        self._recorded_dive_background_paused = False
        if viewer_settings is None:
            self._recording_fps = _env_int("CAVEVIEWER_RECORDING_FPS", 30, 1, 60)
            self._recording_max_height = _env_int(
                recording.RECORDING_MAX_HEIGHT_ENV_VAR,
                recording.RECORDING_DEFAULT_MAX_HEIGHT,
                recording.RECORDING_MIN_OUTPUT_HEIGHT,
                recording.RECORDING_MAX_OUTPUT_HEIGHT,
            )
            self._recording_crf = _env_int("CAVEVIEWER_RECORDING_CRF", 23, 0, 51)
            self._recording_output_dir = os.path.expanduser(
                os.getenv(
                    "CAVEVIEWER_RECORDING_DIR",
                    os.path.join("~", "Movies", "CaveViewer"),
                )
            )
        else:
            self._recording_fps = viewer_settings.recording.fps
            self._recording_max_height = viewer_settings.recording.max_height
            self._recording_crf = viewer_settings.recording.crf
            self._recording_output_dir = os.path.expanduser(
                viewer_settings.recording.directory
            )
        self._workflow_coordinator.recording.frame_interval = (
            1.0 / float(self._recording_fps)
        )
        self._recording_session: recording.RecordingEncoderSession | None = None
        self._recording_output_path: str | None = None
        self._recording_capture: RecordingCaptureResources | None = None
        self._recording_size: tuple[int, int] | None = None
        self._recording_viewport: tuple[int, int, int, int] | None = None
        self._recording_readback_framebuffer: moderngl.Framebuffer | None = None
        self._recording_readback_slots: list[_RecordingReadbackSlot] = []
        self._recording_readback_pending: list[_RecordingReadbackSlot] = []
        self._recording_readback_byte_count = 0
        self._recording_frame_queue: queue.Queue | None = None
        self._recording_stop_results: queue.Queue[_RecordingStopResult] = queue.Queue()
        self._recording_stop_thread: threading.Thread | None = None
        self._recording_stop_cancel_event: threading.Event | None = None

        benchmark_config = session_config.benchmark
        if benchmark_config is not None:
            wnd = getattr(self, "wnd", None)
            actual_window_size = _benchmark_environment_size(
                getattr(wnd, "size", None)
            )
            actual_framebuffer_size = _benchmark_environment_size(
                getattr(wnd, "buffer_size", None)
            )
            ui_surface_size = _benchmark_environment_size(
                _viewer_ui_surface_size(
                    wnd,
                    tuple(actual_window_size)
                    if actual_window_size is not None
                    else _DEFAULT_WINDOW_SIZE,
                )
            )
            benchmark_controller = BenchmarkController(
                scenario=benchmark_config.scenario,
                output_dir=benchmark_config.output_dir,
                logger=_LOG,
                perf_counter=lambda: time.perf_counter(),
                environment=benchmark_config.environment,
            )
            benchmark_controller.update_environment(
                {
                    "gl_vendor": str(self.ctx.info.get("GL_VENDOR", "")),
                    "gl_renderer": str(self.ctx.info.get("GL_RENDERER", "")),
                    "gl_version": str(self.ctx.info.get("GL_VERSION", "")),
                    "window_backend": str(
                        getattr(getattr(self, "wnd", None), "name", "")
                    ),
                    "actual_window_size": actual_window_size,
                    "actual_framebuffer_size": actual_framebuffer_size,
                    "actual_ui_surface_size": ui_surface_size,
                    "vsync": bool(getattr(self, "vsync", False)),
                }
            )
            benchmark_controller.prepare_output()
            self._workflow_coordinator.set_benchmark_controller(
                benchmark_controller
            )

        self._install_backend_modifier_probe()

        # Headlamp brightness control: a -/value/+ stepper, right side of
        # the screen. Replaced a draggable vertical slider -- dragging the
        # handle was unreliable for at least one person testing this
        # (clicking the track worked, grabbing the handle to drag did
        # not), so this sidesteps the whole class of problem by using
        # discrete +/-1 clicks instead of continuous drag-tracking.
        # Range/default unchanged from the old slider (0-10, default 3).
        self.light_stepper = StepperControl(
            self.ctx,
            "BRIGHTNESS",
            initial_value=5,
            min_value=0,
            max_value=10,
            text_scale=self._right_column_text_scale(),
            geometry_scale=self._right_column_geometry_scale(),
            label_text_scale=self._right_column_label_text_scale(),
        )

        # Render distance control: a -/value/+ stepper, left side of the
        # screen, mirroring the brightness control's placement logic but
        # on the opposite side. Directly drives
        # self.world.config.load_radius_cells live, same as the slider it
        # replaced. Range is 1-10 chunk-radius units. Default is 3 for a
        # balanced initial view radius without being overly aggressive on
        # memory usage. StreamingWorld's max_loaded_chunks safety valve
        # (see caveviewer.core.streaming.world) still applies underneath this as
        # a hard backstop regardless of what this is set to.
        self.render_distance_stepper = StepperControl(
            self.ctx,
            "DISTANCE",
            initial_value=3,
            min_value=1,
            max_value=10,
            text_scale=self._right_column_text_scale(),
            geometry_scale=self._right_column_geometry_scale(),
            label_text_scale=self._right_column_label_text_scale(),
        )

        # "Global illumination" control: not actual simulated light
        # bouncing (a much bigger rendering undertaking), but an even
        # ambient fill light across the WHOLE cave, independent of the
        # headlamp -- raising this washes out shadows so the cave reads
        # clearly without the headlamp doing all the work, similar to
        # what people commonly mean by a one-button "GI toggle" in
        # smaller tools. Range 0-10 maps to the shader's u_ambient float
        # (see _AMBIENT_MIN/_AMBIENT_MAX below) -- 0 reproduces the
        # original fixed ambient value this app always used (0.04, a
        # tiny fill so unlit areas aren't pure black), so leaving this at
        # its default changes nothing from before this feature existed.
        self.ambient_stepper = StepperControl(
            self.ctx,
            "GLOBAL LIGHT",
            initial_value=5,
            min_value=0,
            max_value=10,
            text_scale=self._right_column_text_scale(),
            geometry_scale=self._right_column_geometry_scale(),
            label_text_scale=self._right_column_label_text_scale(),
        )

        # Mesh/Texture toggle buttons, stacked just below the brightness
        # slider. Mesh = wireframe overlay on/off; Texture = whether the
        # photo texture is sampled or the surface falls back to plain lit
        # gray. See caveviewer.gui.render_mode_buttons for the four resulting
        # combined display states.
        self.render_mode_buttons = RenderModeButtons(
            self.ctx,
            texture_enabled=True,
            wireframe_enabled=False,
            smooth_shading_enabled=True,
            text_scale=self._right_column_button_text_scale(),
            geometry_scale=self._right_column_geometry_scale(),
        )
        # Loading-policy lock for right-side button effects. While a map
        # is loading, all render-mode toggles are forced off; once
        # loading completes, defaults become Texture ON, Mesh OFF,
        # Shade OFF until explicitly enabled by the user.
        self._render_mode_load_lock_active = False

        # Controls reference / loading overlay -- full-screen right now
        # while the first chunks around the spawn point stream in, and
        # again as a smaller panel any time a minimap click teleports the
        # camera somewhere new (see on_mouse_press_event's minimap-click
        # handling, which calls self.controls_overlay.show_panel()).
        self.controls_overlay = ControlsOverlay(
            self.ctx,
            presentation_profile=self._presentation_profile,
            branding_assets=self._branding_assets,
        )
        self.controls_overlay.show_fullscreen()

        # Background ("void") color picker, toggled via the COLOR button.
        # Defaults to the same near-black the viewer always used, so
        # nothing changes for anyone who never opens it.
        self.color_picker = ColorPicker(self.ctx, initial_color=(0.02, 0.02, 0.03))

        # Shown only while a newly-opened map is being imported/chunked
        # for the first time (see _handle_open_button_click) -- never
        # active during normal viewing, so it has no on/off state of its
        # own the way the other overlays do.
        if self.import_progress_panel is None:
            self.import_progress_panel = ImportProgressPanel(
                self.ctx,
                branding_assets=self._branding_assets,
            )
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        # Map-specific state (world, manifest, camera, minimap, texture manager,
        # chunk GPU objects) lives in its own method, separate
        # from the one-time-per-window setup above, so the exact same
        # logic can run again later when switching to a different map via
        # the OPEN button -- see load_new_map() / _teardown_current_map().
        self.cache_dir = None
        self.textures_dir = None
        self.map_root: str | None = None
        self.manifest = None
        self.world = None
        self.camera = None
        self.minimap = None
        self.texture_manager = None
        self._chunk_upload_manager: ChunkUploadManager | None = None
        self._chunk_gpu_objects: dict[tuple, list] = {}
        self._chunk_upload_states: dict[tuple, dict] = {}
        # Per-chunk, per-material CPU-side data for instant SHADE toggle:
        # each entry holds (mat_name, positions, uvs, smooth_normals, flat_normals)
        # tuples in the same order as _chunk_gpu_objects, so toggling shading
        # can zip the two lists and rewrite each VBO in place via vbo.write().
        self._chunk_normal_cache: dict[tuple, list] = {}
        # Per-cell world-space AABBs for frustum culling, populated as chunks
        # become resident.
        self._chunk_aabbs: dict[tuple, tuple] = {}
        self._view_culling_cache = view_culling.FrustumCullingCache()
        self._chunk_visibility_generation = 0
        self._texture_validation_executor: ThreadPoolExecutor | None = None
        self._texture_validation_future: Future | None = None
        self._texture_validation_manager: TextureManager | None = None
        self._texture_validation_cache_dir: str | None = None
        self._texture_validation_started_at: float | None = None
        self._has_map_loaded = False
        self._pending_import_started = False
        self._initial_chunks_loaded = False
        self._initial_visual_ready = False
        self._initial_visual_ready_frames = 0
        self._initial_visual_ready_visible_chunks = 0
        self._initial_visual_ready_required_textures = 0
        self._initial_visual_ready_resident_textures = 0
        self._initial_visual_ready_visible_textures = 0
        self._initial_visual_ready_missing_textures = 0
        self._initial_visual_ready_expected_chunks = 0
        self._initial_visual_ready_covered_chunks = 0
        self._initial_visual_ready_missing_chunks = 0
        self._initial_visual_ready_coverage_pct = 100.0
        self._initial_route_prefetch_expected_cells = 0
        self._initial_route_prefetch_loaded_cells = 0
        self._initial_route_prefetch_pending_cells = 0
        self._initial_route_prefetch_failed_cells = 0
        self._initial_route_prefetch_missing_cells = 0
        self._initial_route_prefetch_coverage_pct = 100.0
        self._initial_visual_ready_logged = False
        self._initial_compilation_started_at = None
        self._initial_compilation_logged = False
        self._chunk_prep_progress = 0.0
        self._chunk_prep_complete_until = None
        self._chunk_prep_completion_armed = False
        self._main_thread_stall_last_log_at: dict[str, float] = {}
        self._window_resources_released = False

        # Background import state.  Import runs on a worker thread so the
        # render loop stays live (resize, repaint, vsync) the whole time.
        self._import_active: bool = False
        self._import_is_startup: bool = False
        self._import_thread: threading.Thread | None = None
        self._import_process = None
        self._import_command_queue = None
        self._import_stop_event: threading.Event | None = None
        self._import_queue: queue.Queue | None = None
        self._import_pause_requested: bool = False
        self._import_model_format: str | None = None
        self._import_map_name: str = ""
        self._import_progress_stage: str = ""
        self._import_progress_fraction: float = 0.0
        self._import_progress_title: str = ""
        self._import_progress_note: str = ""
        self._import_resuming_from_checkpoint: bool = False
        self._import_pause_notice_until: float | None = None
        self._import_pause_notice_close_after: bool = False
        self._import_pause_notice_map_name: str = ""
        self._import_pause_notice_title: str = "Import paused"
        self._import_pause_notice_stage: str = "resume point saved"
        self._import_pause_notice_note: str = ""
        self._startup_map_load_pending: tuple[
            str,
            str,
            dict,
            str | None,
        ] | None = None
        self._startup_map_load_splash_rendered = False

        if have_ready_cache:
            self._startup_map_load_pending = (
                session_config.cache_dir,
                session_config.textures_dir,
                session_config.manifest,
                session_config.map_root,
            )
        # else: have_pending_import is true instead -- the actual import
        # is deliberately NOT run here, before the window has rendered
        # even one frame. It's triggered from inside on_render() instead
        # (see _run_pending_import), once the window is confirmed to
        # actually be open and able to draw the in-window progress panel
        # -- starting the blocking import here, before super().__init__()
        # has truly finished and the window is on screen, would risk the
        # exact same "nothing to draw into yet" problem this feature
        # exists to avoid.
        self._window_setup_complete = True
        record_runtime_stage(
            "viewer_config_initialization_complete",
            initial_map_mode=(
                "cached" if have_ready_cache else "pending_import"
            ),
        )

    def _active_presentation_profile(self) -> PresentationProfile:
        """Return the immutable UI profile for this viewer instance."""
        profile = getattr(self, "_presentation_profile", None)
        if profile is None:
            profile = _presentation_profile_for_runtime(
                getattr(self, "_platform_runtime", None)
            )
            self._presentation_profile = profile
        return profile

    def _active_presentation_actions_adapter(self) -> PresentationActionsAdapter:
        """Return native presentation actions without reusing static policy."""
        actions = getattr(self, "_presentation_actions_adapter", None)
        if actions is None:
            actions = _presentation_actions_adapter_for_runtime(
                getattr(self, "_platform_runtime", None),
            )
            self._presentation_actions_adapter = actions
        return actions

    def _active_saved_artifact_reveal_adapter(self) -> SavedArtifactRevealAdapter:
        """Return the runtime action adapter or compose a direct fallback."""
        return _saved_artifact_reveal_adapter_for_runtime(
            getattr(self, "_platform_runtime", None)
        )

    def _active_recording_process_adapter(self) -> RecordingProcessAdapter:
        """Return the runtime launch adapter or compose a direct fallback."""
        return _recording_process_adapter_for_runtime(
            getattr(self, "_platform_runtime", None)
        )

    def _active_benchmark_controller(self) -> BenchmarkController | None:
        """Return an injected test controller or the session-owned controller."""
        controller = self.__dict__.get("_benchmark_controller")
        if controller is not None:
            return controller
        workflows = self.__dict__.get("_workflow_coordinator")
        return None if workflows is None else workflows.benchmark_controller

    def _finish_benchmark(self, *, reason: str) -> bool:
        """Finish benchmark output through its session lifecycle owner."""
        workflows = self.__dict__.get("_workflow_coordinator")
        if workflows is not None:
            return workflows.finish_benchmark(reason=reason)
        controller = self.__dict__.get("_benchmark_controller")
        if controller is None or controller.finished:
            return False
        controller.finish(reason=reason)
        return True

    def _acquire_import_inhibitor(self, map_name: str):
        """Use the runtime's shared desktop service for a map-import action."""
        runtime = getattr(self, "_platform_runtime", None)
        if runtime is None:
            return _acquire_map_import_inhibitor(map_name)
        return _acquire_map_import_inhibitor(
            map_name,
            desktop_services=runtime.desktop_services,
            platform_runtime=runtime,
        )

    def _ensure_import_controller(self) -> MapImportController:
        controller = self.__dict__.get("_import_controller")
        if controller is not None:
            return controller

        runtime_settings = getattr(self, "_runtime_settings", None)

        def launch_import_process(model_descriptor: dict, textures_dir: str):
            if runtime_settings is None:
                return start_import_process(model_descriptor, textures_dir)
            return start_import_process(
                model_descriptor,
                textures_dir,
                runtime_settings=runtime_settings.import_configuration(),
            )

        def create_controller() -> MapImportController:
            return MapImportController(
                self,
                logger=lambda: _LOG,
                chunker=lambda: chunker,
                start_import_process=lambda: launch_import_process,
                terminate_import_process=lambda: terminate_import_process,
                acquire_inhibitor=lambda: self._acquire_import_inhibitor,
                release_inhibitor=lambda: _release_desktop_inhibitor,
                perf_counter=lambda: time.perf_counter(),
                monotonic=lambda: time.monotonic(),
                report_startup_failure=self._record_startup_import_failure,
            )

        workflows = self.__dict__.get("_workflow_coordinator")
        if workflows is not None:
            return workflows.ensure_import_controller(create_controller)
        controller = create_controller()
        self.__dict__["_import_controller"] = controller
        return controller

    def _record_startup_import_failure(self, message: str, suggestion: str) -> None:
        """Preserve a recoverable failure across native-window teardown."""
        self._viewer_session.record_outcome(
            kind="import_failed",
            message=message,
            suggestion=suggestion,
        )

    def _ensure_recording_controller(self) -> RecordingStateController:
        controller = self.__dict__.get("_recording_controller")
        if controller is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.recording
            controller = self.__dict__.setdefault(
                "_recording_controller",
                RecordingStateController(),
            )
        return controller

    def _ensure_frame_scheduler(self) -> ViewerFrameScheduler:
        """Return the non-GL frame phase and throttling coordinator."""
        scheduler = self.__dict__.get("_frame_scheduler")
        if scheduler is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.frame_scheduler
            scheduler = self.__dict__.setdefault(
                "_frame_scheduler",
                ViewerFrameScheduler(),
            )
        return scheduler

    def _ensure_capture_workflow(self) -> ViewerCaptureWorkflow:
        """Return the non-GL workflow shared by the capture controllers."""
        workflow = self.__dict__.get("_capture_workflow")
        if workflow is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.capture
            workflow = self.__dict__.setdefault(
                "_capture_workflow",
                ViewerCaptureWorkflow(),
            )
        return workflow

    def _ensure_action_dispatcher(self) -> ViewerActionDispatcher:
        """Return the ordered key-action coordinator for this viewer session."""
        dispatcher = self.__dict__.get("_action_dispatcher")
        if dispatcher is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.actions
            dispatcher = self.__dict__.setdefault(
                "_action_dispatcher",
                ViewerActionDispatcher(),
            )
        return dispatcher

    def _ensure_manual_dive_trace_controller(self) -> ManualDiveTraceStateController:
        controller = self.__dict__.get("_manual_dive_trace_controller")
        if controller is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.manual_dive_trace
            controller = self.__dict__.setdefault(
                "_manual_dive_trace_controller",
                ManualDiveTraceStateController(),
            )
        return controller

    def _ensure_slice_selection_controller(self) -> SliceSelectionController:
        controller = self.__dict__.get("_slice_selection_controller")
        if controller is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.slice_selection
            controller = self.__dict__.setdefault(
                "_slice_selection_controller",
                SliceSelectionController(),
            )
        return controller

    def _ensure_slice_export_controller(self) -> SliceExportController:
        controller = self.__dict__.get("_slice_export_controller")
        if controller is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.slice_export
            controller = self.__dict__.setdefault(
                "_slice_export_controller",
                SliceExportController(),
            )
        return controller

    def _workflow_snapshot(self) -> ViewerWorkflowSnapshot:
        """Adapt render-thread state for the non-GL workflow coordinator."""
        manual_trace = self._ensure_manual_dive_trace_controller()
        slice_selection = self._ensure_slice_selection_controller()
        slice_export = self._ensure_slice_export_controller()
        recording_armed = self._recording_is_armed()
        return ViewerWorkflowSnapshot(
            setup_complete=bool(getattr(self, "_window_setup_complete", False)),
            closing_requested=bool(getattr(self, "_closing_requested", False)),
            iconified=bool(getattr(self, "_is_iconified", False)),
            import_active=bool(getattr(self, "_import_active", False)),
            map_loaded=bool(getattr(self, "_has_map_loaded", False)),
            capture_close_pending=self._capture_close_pending(),
            recording_owned=(
                recording_armed or self._recording_stop_in_progress()
            ),
            recording_armed=recording_armed,
            recording_active=(
                getattr(self, "_recording_session", None) is not None
            ),
            manual_dive_trace_countdown_active=manual_trace.countdown_active,
            manual_dive_trace_active=(
                getattr(self, "_manual_dive_trace", None) is not None
            ),
            manual_dive_trace_finalizing=bool(
                getattr(self, "_manual_dive_trace_writers", None)
            ),
            slice_countdown_active=slice_selection.countdown_active,
            slice_selection_active=slice_selection.selection_active,
            slice_saving=slice_selection.saving,
            slice_export_active=slice_export.active,
        )

    def _workflow_render_request(self) -> ViewerRenderRequest | None:
        """Return aggregate non-GL decisions for a production viewer session."""
        workflows = self.__dict__.get("_workflow_coordinator")
        if workflows is None:
            return None
        return workflows.render_request(self._workflow_snapshot())

    def _slice_work_pending(self) -> bool:
        """Return whether a countdown or child export needs a frame-time poll."""
        request = self._workflow_render_request()
        if request is not None:
            return request.slice_work_pending
        selection = self.__dict__.get("_slice_selection_controller")
        exporter = self.__dict__.get("_slice_export_controller")
        return bool(
            selection is not None and selection.countdown_active
        ) or bool(exporter is not None and exporter.active)

    def _slice_interaction_active(self) -> bool:
        """Return whether slice selection owns the capture interaction surface."""
        request = self._workflow_render_request()
        if request is not None:
            return request.slice_interaction_active
        selection = self.__dict__.get("_slice_selection_controller")
        exporter = self.__dict__.get("_slice_export_controller")
        return bool(
            selection is not None
            and (
                selection.countdown_active
                or selection.selection_active
                or selection.saving
            )
        ) or bool(exporter is not None and exporter.active)

    def _capture_ownership_state(self) -> CaptureOwnershipState:
        """Return all lifecycle owners used to enforce one capture at a time."""
        return CaptureOwnershipState(
            recording_owned=(
                self._recording_is_armed() or self._recording_stop_in_progress()
            ),
            manual_dive_trace_owned=(
                self._ensure_manual_dive_trace_controller().countdown_active
                or getattr(self, "_manual_dive_trace", None) is not None
                or bool(getattr(self, "_manual_dive_trace_writers", None))
            ),
            slice_owned=self._slice_interaction_active(),
        )

    def _capture_owner(self) -> CaptureOwner | None:
        """Return the countdown, active capture, or finalizer owning capture."""
        request = self._workflow_render_request()
        if request is not None:
            return request.capture_owner
        return self._ensure_capture_workflow().owner_for(
            self._capture_ownership_state()
        )

    def _capture_start_blocked(self, requested_owner: CaptureOwner) -> bool:
        """Reject a second capture while the current owner is still cleaning up."""
        owner = self._capture_owner()
        if owner is None:
            return False
        owner_name = {
            CaptureOwner.VIDEO: "video recording",
            CaptureOwner.DIVE_TRACE: "dive trace",
            CaptureOwner.SLICE: "cave slice",
        }[owner]
        requested_name = {
            CaptureOwner.VIDEO: "video recording",
            CaptureOwner.DIVE_TRACE: "dive trace",
            CaptureOwner.SLICE: "cave slice",
        }[requested_owner]
        self._show_capture_status(
            "Capture in progress",
            (
                f"Finish or cancel the current {owner_name} before starting "
                f"a new {requested_name}."
            ),
            kind="info",
            duration=3.0,
        )
        return True

    def _capture_shortcut_is_ignored(self, requested_owner: CaptureOwner) -> bool:
        """Consume a foreign capture shortcut without presentation side effects."""
        return self._ensure_capture_workflow().should_ignore_capture_shortcut(
            active_owner=self._capture_owner(),
            requested_owner=requested_owner,
        )

    def _active_capture_owner(self) -> CaptureOwner | None:
        """Return the owner that is actively collecting a video, trace, or slice."""
        request = self._workflow_render_request()
        if request is not None:
            return request.active_capture_owner
        selection = self.__dict__.get("_slice_selection_controller")
        return self._ensure_capture_workflow().owner_for(
            CaptureOwnershipState(
                recording_owned=(
                    getattr(self, "_recording_session", None) is not None
                ),
                manual_dive_trace_owned=(
                    getattr(self, "_manual_dive_trace", None) is not None
                ),
                slice_owned=bool(
                    selection is not None and selection.selection_active
                ),
            )
        )

    def _render_active_capture_instruction(
        self,
        window_size: tuple[int, int],
    ) -> bool:
        """Render guidance only if capture policy supplies a persistent banner."""
        instruction = self._ensure_capture_workflow().instruction_for(
            self._active_capture_owner(),
            primary_shortcut_label=self._primary_shortcut_label(),
        )
        if instruction is None:
            return False
        self._render_dive_status_prompt(
            window_size,
            title=instruction.title,
            note=instruction.note,
        )
        return True

    def _ensure_artifact_capture_presentation(
        self,
    ) -> ArtifactCapturePresentationController:
        """Return the shared post-save feedback and reveal scheduler."""
        controller = self.__dict__.get("_artifact_capture_presentation")
        if controller is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.artifact_presentation
            controller = self.__dict__.setdefault(
                "_artifact_capture_presentation",
                ArtifactCapturePresentationController(),
            )
        return controller

    def _ensure_recording_capture(self) -> RecordingCaptureResources:
        capture = self.__dict__.get("_recording_capture")
        if capture is None:
            capture = RecordingCaptureResources(
                ctx=getattr(self, "ctx", None),
                buffer_count=self.RECORDING_READBACK_BUFFER_COUNT,
                readback_components=self.RECORDING_READBACK_COMPONENTS,
                logger=_LOG,
                perf_counter=lambda: time.perf_counter(),
            )
            self.__dict__["_recording_capture"] = capture
        capture.ctx = getattr(self, "ctx", None)
        capture.buffer_count = self.RECORDING_READBACK_BUFFER_COUNT
        capture.readback_components = self.RECORDING_READBACK_COMPONENTS
        capture.logger = _LOG
        capture.output_size = getattr(self, "_recording_size", None)
        capture.capture_viewport = getattr(self, "_recording_viewport", None)
        capture.readback_framebuffer = getattr(
            self,
            "_recording_readback_framebuffer",
            None,
        )
        capture.readback_slots = getattr(self, "_recording_readback_slots", [])
        capture.readback_pending = getattr(self, "_recording_readback_pending", [])
        capture.readback_byte_count = int(
            getattr(self, "_recording_readback_byte_count", 0)
        )
        return capture

    def _sync_recording_capture_state_from_manager(self) -> None:
        capture = self.__dict__.get("_recording_capture")
        if capture is None:
            return
        self._recording_size = capture.output_size
        self._recording_viewport = capture.capture_viewport
        self._recording_readback_framebuffer = capture.readback_framebuffer
        self._recording_readback_slots = capture.readback_slots
        self._recording_readback_pending = capture.readback_pending
        self._recording_readback_byte_count = capture.readback_byte_count

    @property
    def _recording_countdown_started_at(self) -> float | None:
        return self._ensure_recording_controller().countdown_started_at

    @_recording_countdown_started_at.setter
    def _recording_countdown_started_at(self, value: float | None) -> None:
        self._ensure_recording_controller().countdown_started_at = value

    @property
    def _recording_countdown_until(self) -> float | None:
        return self._ensure_recording_controller().countdown_until

    @_recording_countdown_until.setter
    def _recording_countdown_until(self, value: float | None) -> None:
        self._ensure_recording_controller().countdown_until = value

    @property
    def _recording_last_stage_ms(self) -> float:
        return self._ensure_recording_controller().last_stage_ms

    @_recording_last_stage_ms.setter
    def _recording_last_stage_ms(self, value: float) -> None:
        self._ensure_recording_controller().last_stage_ms = value

    @property
    def _recording_last_drain_ms(self) -> float:
        return self._ensure_recording_controller().last_drain_ms

    @_recording_last_drain_ms.setter
    def _recording_last_drain_ms(self, value: float) -> None:
        self._ensure_recording_controller().last_drain_ms = value

    @property
    def _recording_next_frame_time(self) -> float | None:
        return self._ensure_recording_controller().next_frame_time

    @_recording_next_frame_time.setter
    def _recording_next_frame_time(self, value: float | None) -> None:
        self._ensure_recording_controller().next_frame_time = value

    @property
    def _recording_frame_interval(self) -> float:
        return self._ensure_recording_controller().frame_interval

    @_recording_frame_interval.setter
    def _recording_frame_interval(self, value: float) -> None:
        self._ensure_recording_controller().frame_interval = value

    @property
    def _recording_dropped_frames(self) -> int:
        return self._ensure_recording_controller().dropped_frames

    @_recording_dropped_frames.setter
    def _recording_dropped_frames(self, value: int) -> None:
        self._ensure_recording_controller().dropped_frames = value

    @property
    def _recording_status_message(self) -> str | None:
        return self._ensure_recording_controller().status_message

    @_recording_status_message.setter
    def _recording_status_message(self, value: str | None) -> None:
        self._ensure_recording_controller().status_message = value

    @property
    def _recording_status_detail(self) -> str | None:
        return self._ensure_recording_controller().status_detail

    @_recording_status_detail.setter
    def _recording_status_detail(self, value: str | None) -> None:
        self._ensure_recording_controller().status_detail = value

    @property
    def _recording_status_kind(self) -> str | None:
        return self._ensure_recording_controller().status_kind

    @_recording_status_kind.setter
    def _recording_status_kind(self, value: str | None) -> None:
        self._ensure_recording_controller().status_kind = value

    @property
    def _recording_status_until(self) -> float | None:
        return self._ensure_recording_controller().status_until

    @_recording_status_until.setter
    def _recording_status_until(self, value: float | None) -> None:
        self._ensure_recording_controller().status_until = value

    def _claim_backend_escape_key(self) -> None:
        """Disable the backend's preemptive Escape close callback."""
        self.wnd.exit_key = None

    def _claim_backend_close_event(self) -> None:
        """Route Pyglet close requests through CaveViewer's deferred workflow.

        Returning ``True`` is Pyglet's ``EVENT_HANDLED`` sentinel.  Without it,
        Pyglet invokes its default close handler after our callback and sets
        ``has_exit`` even when :meth:`on_close` has deferred shutdown.
        """
        backend = getattr(self, "wnd", None)
        native_window = getattr(backend, "_window", None)
        push_handlers = getattr(native_window, "push_handlers", None)
        if getattr(backend, "name", None) != "pyglet" or not callable(push_handlers):
            return

        def on_close() -> bool:
            self.on_close()
            return True

        push_handlers(on_close=on_close)

    def _set_runtime_window_icon(self) -> None:
        """Set the native viewer-window icon when the backend exposes one."""
        viewer_settings = getattr(self, "_viewer_runtime_settings", None)
        icon_path = (
            viewer_settings.app_icon
            if viewer_settings is not None and viewer_settings.app_icon
            else _runtime_app_icon_path(getattr(self, "_platform_runtime", None))
        )
        if not os.path.exists(icon_path):
            _LOG.warning(f"viewer window icon asset not found: {icon_path}")
            return

        targets = []
        for target in (getattr(self, "wnd", None), getattr(getattr(self, "wnd", None), "_window", None)):
            if target is not None and target not in targets:
                targets.append(target)

        for target in targets:
            set_icon = getattr(target, "set_icon", None)
            if not callable(set_icon):
                continue
            try:
                # Try passing the path directly first — some pyglet versions
                # (and some backends) expect a filename/Path rather than a
                # pre-loaded ImageData object and will call .is_absolute() on
                # the argument, which fails on ImageData.
                set_icon(icon_path)
                _LOG.info("Set viewer window icon.")
                return
            except Exception:
                pass
            try:
                import pyglet
                icon = pyglet.image.load(icon_path)
                set_icon(icon)
                _LOG.info("Set viewer window icon.")
                return
            except Exception as e:
                _LOG.warning(f"could not set viewer window icon ({e}); continuing without it.")
                return

        _LOG.debug("viewer backend does not expose a set_icon() hook.")

    def _load_map(
        self,
        cache_dir: str,
        textures_dir: str,
        manifest: dict,
        *,
        map_root: str | os.PathLike[str] | None = None,
    ) -> None:
        """
        Sets up everything specific to ONE map: the texture manager, the
        streaming world, the starting camera position, and the minimap.
        Called once from __init__ for the map the program launched with,
        and called again from load_new_map() when switching to a
        different map via the OPEN button -- _teardown_current_map() must
        be called first in that second case, to cleanly release the
        previous map's GPU/thread resources before this builds new ones.
        """
        load_started_at = time.perf_counter()
        self.cache_dir = cache_dir
        self.textures_dir = textures_dir
        self.map_root = _normalize_map_root(map_root)
        self.manifest = manifest
        pending_recorded_dive = getattr(
            self,
            "_pending_recorded_dive_trace",
            None,
        )
        if pending_recorded_dive is not None:
            recorded_dive.validate_recorded_dive_manifest(
                pending_recorded_dive,
                manifest,
            )
        self._initial_compilation_started_at = time.perf_counter()
        self._initial_compilation_logged = False

        viewer_settings = self._viewer_runtime_settings
        streaming_settings = (
            viewer_settings.streaming if viewer_settings is not None else None
        )
        gpu_vendor = str(self.ctx.info.get("GL_VENDOR", ""))
        gpu_memory_bytes = gpu_memory.detect_total_gpu_memory_bytes(
            gpu_vendor,
            logger=_LOG,
            environment=(
                (
                    {"CAVEVIEWER_GPU_MEMORY_GB": str(streaming_settings.gpu_memory_gb)}
                    if streaming_settings is not None
                    and streaming_settings.gpu_memory_gb is not None
                    else {}
                )
                if streaming_settings is not None
                else None
            ),
        )
        gpu_target_fraction = (
            memory_targets.parse_gpu_target_fraction(
                os.environ.get("CAVEVIEWER_GPU_MEMORY_UTILIZATION_TARGET")
            )
            if streaming_settings is None
            else max(
                0.01,
                min(
                    0.80,
                    float(streaming_settings.gpu_memory_target_percent) / 100.0,
                ),
            )
        )
        max_texture_dimension = TextureManager.recommend_max_texture_dimension(
            self.manifest["mtl_materials"],
            gpu_memory_bytes,
            gpu_target_fraction,
            configured_limit=(
                viewer_settings.max_texture_dimension
                if viewer_settings is not None
                else None
            ),
            use_environment_override=viewer_settings is None,
        )
        ram_snapshot = system_memory.detect_ram_snapshot()
        max_decoded_cache_bytes = TextureManager.recommend_decoded_cache_bytes(
            ram_snapshot.available_bytes if ram_snapshot is not None else None
        )
        max_resident_texture_bytes = (
            TextureManager.recommend_resident_texture_cache_bytes(
                gpu_memory_bytes,
                gpu_target_fraction,
            )
        )
        resident_texture_cap_bytes = (
            _env_optional_mebibytes(_TEXTURE_RESIDENT_CACHE_MB_ENV)
            if streaming_settings is None
            else (
                max(1, int(streaming_settings.texture_resident_cache_mb * 1024 ** 2))
                if streaming_settings.texture_resident_cache_mb is not None
                else None
            )
        )
        if resident_texture_cap_bytes is not None:
            max_resident_texture_bytes = min(
                max_resident_texture_bytes,
                resident_texture_cap_bytes,
            )
            if streaming_settings is None:
                _LOG.info(
                    "Texture resident GPU LRU cache capped by %s=%s MB: %.1f MB.",
                    _TEXTURE_RESIDENT_CACHE_MB_ENV,
                    os.environ.get(_TEXTURE_RESIDENT_CACHE_MB_ENV, "").strip(),
                    max_resident_texture_bytes / (1024 ** 2),
                )
            else:
                _LOG.info(
                    "Texture resident GPU LRU cache capped by composed runtime settings: %.1f MB.",
                    max_resident_texture_bytes / (1024 ** 2),
                )
        gpu_geometry_budget_bytes = None
        if gpu_memory_bytes is not None and gpu_memory_bytes > 0:
            total_gpu_residency_budget_bytes = int(
                gpu_memory_bytes * gpu_target_fraction
            )
            max_resident_texture_bytes = min(
                max_resident_texture_bytes,
                total_gpu_residency_budget_bytes,
            )
            gpu_residency_safety_bytes = min(
                max(0, total_gpu_residency_budget_bytes - max_resident_texture_bytes),
                int(total_gpu_residency_budget_bytes * _GPU_RESIDENCY_SAFETY_SHARE),
            )
            gpu_geometry_budget_bytes = max(
                0,
                total_gpu_residency_budget_bytes
                - max_resident_texture_bytes
                - gpu_residency_safety_bytes,
            )
            _LOG.info(
                "GPU residency budget split: target %.1f MB (%.0f%% of %.1f GB); "
                "textures %.1f MB, geometry %.1f MB, safety %.1f MB.",
                total_gpu_residency_budget_bytes / (1024 ** 2),
                gpu_target_fraction * 100.0,
                gpu_memory_bytes / (1024 ** 3),
                max_resident_texture_bytes / (1024 ** 2),
                gpu_geometry_budget_bytes / (1024 ** 2),
                gpu_residency_safety_bytes / (1024 ** 2),
            )
        texture_setup_started_at = time.perf_counter()
        self.texture_manager = TextureManager(
            self.ctx,
            self.textures_dir,
            self.manifest["mtl_materials"],
            max_texture_dimension=max_texture_dimension,
            max_decoded_cache_bytes=max_decoded_cache_bytes,
            max_resident_texture_bytes=max_resident_texture_bytes,
        )
        self._log_main_thread_stall(
            "texture manager setup",
            time.perf_counter() - texture_setup_started_at,
            materials=len(self.manifest.get("mtl_materials", {})),
        )
        self._start_texture_validation_async()

        def predecode_textures_for_chunk(chunk_data):
            # Called from a background worker thread (see StreamingWorld) --
            # decodes JPEGs for every material this chunk uses, ahead of
            # time, so the eventual main-thread GPU upload can use
            # already-decoded pixels rather than doing a slow
            # decode-and-upload combination.
            for group in chunk_data.groups.values():
                self.texture_manager.decode_for_material(group.material_name)

        chunk_size = chunker.manifest_chunk_size(self.manifest)
        if chunk_size is None:
            raise ValueError(
                "Map cache manifest is missing a valid chunk_size. "
                "Rebuild this map's reported cache directory with this version "
                "of CaveViewer."
            )
        configured_chunk_size = (
            chunker.configured_chunk_size()
            if self._runtime_settings is None
            else float(self._runtime_settings["chunk_size_meters"])
        )
        _LOG.info(f"Opening map cache with manifest chunk size: {chunk_size:g}.")
        if abs(chunk_size - configured_chunk_size) > 1e-6:
            _LOG.info(
                f"Current {chunker.CHUNK_SIZE_ENV_VAR} setting is {configured_chunk_size:g}, "
                "but existing/prebuilt caches stream using the chunk size recorded in manifest.json."
            )
        benchmark_controller = self._active_benchmark_controller()
        if benchmark_controller is not None:
            benchmark_radius = int(benchmark_controller.scenario.render_distance)
            clamped_radius = max(
                self.render_distance_stepper.min_value,
                min(self.render_distance_stepper.max_value, benchmark_radius),
            )
            if clamped_radius != benchmark_radius:
                _LOG.warning(
                    "Benchmark render_distance=%d exceeds viewer control range; "
                    "using %d.",
                    benchmark_radius,
                    clamped_radius,
                )
            self.render_distance_stepper.value = clamped_radius
        config = StreamingConfig(
            chunk_size=chunk_size,
            load_radius_cells=self.render_distance_stepper.value,
            unload_radius_margin=1,
        )
        world_setup_started_at = time.perf_counter()
        self.world = StreamingWorld(
            self.cache_dir,
            config,
            on_decode_textures=predecode_textures_for_chunk,
            prepack_smooth_shading=bool(
                self.render_mode_buttons.smooth_shading_enabled
            ),
            gpu_vendor=gpu_vendor,
            textures_dir=self.textures_dir,
            total_gpu_memory_bytes=gpu_memory_bytes,
            texture_gpu_budget_bytes=max_resident_texture_bytes,
            gpu_geometry_budget_bytes=gpu_geometry_budget_bytes,
            manifest=self.manifest,
            estimate_texture_gpu_bytes=False,
            runtime_settings=streaming_settings,
        )
        self._log_main_thread_stall(
            "streaming world setup",
            time.perf_counter() - world_setup_started_at,
            chunks=len(self.manifest.get("chunks", {})),
        )

        # A Recorded Dive owns its exact first camera pose. Ordinary map opens
        # start at the render-cache position and remain under manual control.
        if pending_recorded_dive is not None:
            start_pos = np.asarray(
                pending_recorded_dive.initial_pose.position,
                dtype=np.float64,
            )
        else:
            start_pos = _map_initial_camera_position(self.manifest)
        self.camera = FlyCamera(position=tuple(start_pos))
        self._benchmark_route_prefetch_cells = frozenset()
        if benchmark_controller is not None:
            benchmark_controller.set_position_origin(start_pos)
            benchmark_controller.apply_initial_camera(self.camera)
            self._configure_benchmark_route_prefetch(start_pos)
        elif pending_recorded_dive is not None:
            controller = recorded_dive.RecordedDivePlaybackController(
                pending_recorded_dive
            )
            controller.start(self.camera, now=time.perf_counter())
            self._recorded_dive_trace = pending_recorded_dive
            self._recorded_dive_controller = controller
            self._pending_recorded_dive_trace = None
            self._refresh_recorded_dive_prefetch()
        self._bookmarks_path = os.path.join(self.cache_dir, "camera_bookmarks.json")
        self._load_bookmarks()

        # Bottom-left minimap: a crude top-down outline of the whole cave's
        # footprint with a live red dot for current position. Built once
        # from the manifest's chunk bounding boxes -- no extra rendering
        # pass or GPU cost beyond this tiny 2D overlay.
        minimap_started_at = time.perf_counter()
        self.minimap = Minimap(self.ctx, self.manifest)
        self._log_main_thread_stall(
            "minimap setup",
            time.perf_counter() - minimap_started_at,
            chunks=len(self.manifest.get("chunks", {})),
        )

        # One-time texture diagnostic: print material/texture summary to
        # console so atlas feasibility can be judged without guessing.
        self._print_texture_diagnostics(manifest, textures_dir)

        # Keep GPU upload state scoped to the active map.  Large maps can have
        # tens or hundreds of thousands of manifest cells, so frustum-culling
        # bounds are populated only as chunks become resident.
        self._chunk_gpu_objects = {}
        self._chunk_upload_states = {}
        self._chunk_normal_cache = {}
        self._chunk_aabbs = {}
        self._view_culling_cache = view_culling.FrustumCullingCache()
        self._chunk_visibility_generation = 0
        self._chunk_upload_manager = ChunkUploadManager(
            ctx=self.ctx,
            program=self.program,
            texture_manager=self.texture_manager,
            smooth_shading_enabled=lambda: bool(
                self.render_mode_buttons.smooth_shading_enabled
            ),
            gpu_objects=self._chunk_gpu_objects,
            upload_states=self._chunk_upload_states,
            normal_cache=self._chunk_normal_cache,
            aabbs=self._chunk_aabbs,
            upload_operations_per_chunk=self._current_upload_operations_per_chunk,
            upload_time_budget_ms=self._current_upload_time_budget_ms,
            vbo_upload_slice_bytes=self._vbo_upload_slice_bytes,
            texture_upload_slice_bytes=self._texture_upload_slice_bytes,
        )

        # Render-distance slider's current value should drive the new
        # map's streaming config immediately, rather than resetting back
        # to the control's own default -- if someone already turned it up
        # for a previous large map, opening another large map shouldn't
        # silently reset that preference. (On first launch, from
        # __init__, this just re-applies the control's own initial value,
        # a harmless no-op.)
        if hasattr(self, "render_distance_stepper"):
            self.world.config.load_radius_cells = self.render_distance_stepper.value

        self.controls_overlay.show_fullscreen()
        # Reset on each map load; set True when the initial view has enough
        # uploaded chunks to be usable, not merely when the first chunk arrives.
        self._reset_initial_chunk_loading_state()
        self._record_benchmark_streaming_environment()
        self._log_main_thread_stall(
            "map load",
            time.perf_counter() - load_started_at,
            chunks=len(self.manifest.get("chunks", {})),
        )

    def _start_texture_validation_async(self) -> bool:
        """
        Start CPU/disk-only texture validation off the render thread.

        Texture validation opens texture headers and checks paths for every
        referenced material.  That is useful diagnostics, but doing it inside
        _load_map() can keep the window event loop from responding long enough
        for the desktop shell to report "application not responding."
        """
        texture_manager = getattr(self, "texture_manager", None)
        if texture_manager is None:
            return False

        self._cancel_texture_validation()
        executor: ThreadPoolExecutor | None = None
        try:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="caveviewer-texture-validate",
            )
            future = executor.submit(texture_manager.validate_textures)
        except Exception as exc:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            _LOG.warning(
                "Could not start background texture validation: %s", exc
            )
            return False

        self._texture_validation_executor = executor
        self._texture_validation_future = future
        self._texture_validation_manager = texture_manager
        self._texture_validation_cache_dir = self.cache_dir
        self._texture_validation_started_at = time.perf_counter()
        return True

    def _update_texture_validation(self) -> None:
        future = getattr(self, "_texture_validation_future", None)
        if future is None or not future.done():
            return

        executor = getattr(self, "_texture_validation_executor", None)
        texture_manager = getattr(self, "_texture_validation_manager", None)
        cache_dir = getattr(self, "_texture_validation_cache_dir", None)
        started_at = getattr(self, "_texture_validation_started_at", None)

        self._clear_texture_validation_state(shutdown_executor=False)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        if texture_manager is not getattr(self, "texture_manager", None):
            return
        if cache_dir != getattr(self, "cache_dir", None):
            return

        elapsed_s = (
            max(0.0, time.perf_counter() - started_at)
            if started_at is not None
            else None
        )
        try:
            result = future.result()
        except Exception as exc:
            _LOG.warning("Background texture validation failed: %s", exc)
            return

        found = len(result.get("found", ())) if isinstance(result, dict) else None
        missing = len(result.get("missing", ())) if isinstance(result, dict) else None
        if elapsed_s is None:
            _LOG.info(
                "Background texture validation completed "
                "(found=%s missing=%s).",
                found,
                missing,
            )
        else:
            _LOG.info(
                "Background texture validation completed in %.2fs "
                "(found=%s missing=%s).",
                elapsed_s,
                found,
                missing,
            )

    def _clear_texture_validation_state(self, *, shutdown_executor: bool) -> None:
        executor = getattr(self, "_texture_validation_executor", None)
        if shutdown_executor and executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self._texture_validation_executor = None
        self._texture_validation_future = None
        self._texture_validation_manager = None
        self._texture_validation_cache_dir = None
        self._texture_validation_started_at = None

    def _cancel_texture_validation(self) -> bool:
        future = getattr(self, "_texture_validation_future", None)
        if future is None:
            return False
        try:
            future.cancel()
        finally:
            self._clear_texture_validation_state(shutdown_executor=True)
        return True

    def _configure_benchmark_route_prefetch(self, origin: np.ndarray) -> None:
        """Ask streaming to keep the benchmark route tube wanted during startup."""
        benchmark_controller = self._active_benchmark_controller()
        world = getattr(self, "world", None)
        if benchmark_controller is None or world is None:
            return

        radius = max(1, int(getattr(world.config, "load_radius_cells", 1)))
        route_cells: set[tuple[int, int, int]] = set()
        route_positions = tuple(
            self._benchmark_route_sample_positions(
                benchmark_controller.scenario,
                origin,
            )
        )
        for position in route_positions:
            route_cell = world.cell_for_position(np.asarray(position, dtype=np.float32))
            route_cells.update(world.available_cells_in_radius(route_cell, radius))

        self._benchmark_route_prefetch_cells = frozenset(route_cells)
        set_prefetch = getattr(world, "set_prefetch_wanted_cells", None)
        if callable(set_prefetch):
            set_prefetch(route_cells)
        _LOG.info(
            "Benchmark route prefetch enabled: %d cells from %d sampled route "
            "position(s), radius=%d.",
            len(route_cells),
            len(route_positions),
            radius,
        )
        benchmark_controller.update_environment(
            {
                "benchmark_route_prefetch_cells": len(route_cells),
                "benchmark_route_prefetch_sample_positions": len(route_positions),
                "benchmark_route_prefetch_radius_chunks": radius,
            }
        )

    def _benchmark_route_sample_positions(
        self,
        scenario,
        origin: np.ndarray,
    ) -> Iterable[np.ndarray]:
        """Yield absolute route positions densely enough to prefetch the route."""
        world = getattr(self, "world", None)
        chunk_size = max(
            1e-6,
            float(getattr(getattr(world, "config", None), "chunk_size", 1.0)),
        )
        origin_array = np.asarray(origin, dtype=np.float64)
        route = tuple(getattr(scenario, "route", ()))
        if not route:
            return

        absolute_positions = [
            self._benchmark_absolute_route_position(scenario, keyframe, origin_array)
            for keyframe in route
        ]
        previous = absolute_positions[0]
        yield previous
        for current in absolute_positions[1:]:
            segment = current - previous
            distance = float(np.linalg.norm(segment))
            steps = max(1, int(math.ceil(distance / chunk_size)))
            for step in range(1, steps + 1):
                t = step / steps
                yield previous + segment * t
            previous = current

    @staticmethod
    def _benchmark_absolute_route_position(
        scenario,
        keyframe,
        origin: np.ndarray,
    ) -> np.ndarray:
        position = np.asarray(keyframe.position, dtype=np.float64)
        if getattr(scenario, "position_mode", "absolute") == "first_chunk_center_offset":
            return origin + position
        return position

    def _record_benchmark_streaming_environment(self) -> None:
        """Persist effective Streaming/texture settings for benchmark artifacts."""
        benchmark_controller = self._active_benchmark_controller()
        world = getattr(self, "world", None)
        if benchmark_controller is None or world is None:
            return

        config = getattr(world, "config", None)
        texture_manager = getattr(self, "texture_manager", None)
        ready_backlog_capacity = getattr(world, "_ready_backlog_capacity", None)
        worker_target = getattr(world, "_worker_pool_size", None)
        active_workers = getattr(world, "_workers", ())
        update = {
            "effective_render_distance_chunks": int(
                getattr(config, "load_radius_cells", 0) or 0
            ),
            "streaming_chunk_size_m": float(
                getattr(config, "chunk_size", 0.0) or 0.0
            ),
            "streaming_unload_radius_margin": int(
                getattr(config, "unload_radius_margin", 0) or 0
            ),
            "streaming_max_loaded_chunks": int(
                getattr(config, "max_loaded_chunks", 0) or 0
            ),
            "streaming_ready_backlog_capacity": (
                None
                if ready_backlog_capacity is None
                else int(ready_backlog_capacity)
            ),
            "streaming_worker_target": (
                None if worker_target is None else int(worker_target)
            ),
            "streaming_active_workers_at_load": len(tuple(active_workers or ())),
            "benchmark_route_prefetch_cells": int(
                len(getattr(self, "_benchmark_route_prefetch_cells", ()))
            ),
            "upload_chunks_per_frame_effective": int(self._upload_chunks_per_frame),
            "upload_groups_per_frame_effective": int(self._upload_groups_per_frame),
            "upload_time_budget_ms_effective": float(self._upload_time_budget_ms),
            "startup_upload_chunks_per_frame": max(
                self._upload_chunks_per_frame,
                _STARTUP_UPLOAD_CHUNKS_PER_FRAME,
            ),
            "startup_upload_groups_per_frame": max(
                self._upload_groups_per_frame,
                _STARTUP_UPLOAD_OPERATIONS_PER_CHUNK,
            ),
            "startup_upload_time_budget_ms": max(
                self._upload_time_budget_ms,
                _STARTUP_UPLOAD_TIME_BUDGET_MS,
            ),
            "catchup_upload_chunks_per_frame": max(
                self._upload_chunks_per_frame,
                _CATCHUP_UPLOAD_CHUNKS_PER_FRAME,
            ),
            "catchup_upload_groups_per_frame": max(
                self._upload_groups_per_frame,
                _CATCHUP_UPLOAD_OPERATIONS_PER_CHUNK,
            ),
            "catchup_upload_time_budget_ms": max(
                self._upload_time_budget_ms,
                _CATCHUP_UPLOAD_TIME_BUDGET_MS,
            ),
            "texture_max_dimension": (
                None
                if texture_manager is None
                else texture_manager.max_texture_dimension
            ),
            "texture_resident_budget_bytes": (
                None
                if texture_manager is None
                else texture_manager.max_resident_texture_bytes
            ),
            "texture_decoded_cache_budget_bytes": (
                None
                if texture_manager is None
                else texture_manager.max_decoded_cache_bytes
            ),
        }
        benchmark_controller.update_environment(update)

    def _move_camera(self, forward_amt: float, right_amt: float, up_amt: float,
                     dt: float, speed_multiplier: float) -> None:
        """Move the camera freely without constraining it to map geometry."""
        if self.camera is None:
            return
        self.camera.move(forward_amt, right_amt, up_amt, dt, speed_multiplier)

    def _recording_is_armed(self) -> bool:
        return self._ensure_recording_controller().is_armed(
            process_active=getattr(self, "_recording_session", None) is not None
        )

    def _ensure_recording_stop_state(self) -> None:
        if not hasattr(self, "_recording_stop_results"):
            self._recording_stop_results = queue.Queue()
        if not hasattr(self, "_recording_stop_thread"):
            self._recording_stop_thread = None
        if not hasattr(self, "_recording_stop_cancel_event"):
            self._recording_stop_cancel_event = None

    def _recording_stop_in_progress(self) -> bool:
        self._ensure_recording_stop_state()
        return self._recording_stop_thread is not None

    def _exit_capture_artifacts(self) -> tuple[str, ...]:
        """Return user artifacts that must finish before the viewer may close."""
        artifacts: list[str] = []
        if (
            getattr(self, "_recording_session", None) is not None
            or self._recording_stop_in_progress()
        ):
            artifacts.append("Video")
        if (
            getattr(self, "_manual_dive_trace", None) is not None
            or getattr(self, "_manual_dive_trace_writers", None)
        ):
            artifacts.append("Dive trace")
        if (
            self._ensure_slice_selection_controller().selection_active
            or self._ensure_slice_export_controller().active
        ):
            artifacts.append("Slice")
        return tuple(artifacts)

    def _exit_capture_finalization_active(self) -> bool:
        """Return whether shutdown is waiting for a user artifact writer."""
        return self._ensure_capture_workflow().exit_finalization_active

    def _escape_capture_cancellation_active(self) -> bool:
        """Return whether Escape is canceling a capture before viewer close."""
        return self._ensure_capture_workflow().escape_cancellation_active

    def _capture_close_pending(self) -> bool:
        """Return whether capture cleanup currently owns viewer shutdown."""
        return self._ensure_capture_workflow().close_pending

    def _defer_backend_close_request(self) -> None:
        """Keep the GLFW window alive after its close callback has fired."""
        wnd = getattr(self, "wnd", None)
        if wnd is None:
            return
        try:
            wnd.is_closing = False
        except Exception:
            # A lightweight or future backend may not expose a cancellable
            # native-close flag. Its normal close path remains unchanged.
            pass

    def _begin_exit_capture_finalization(
        self,
        artifact_names: tuple[str, ...],
    ) -> None:
        """Stop active capture cleanly and show progress before shutdown."""
        self._ensure_capture_workflow().begin_exit_finalization()
        self._defer_backend_close_request()
        self._reset_transient_input_state("saving capture before close")

        presentation = self._ensure_artifact_capture_presentation()
        # A user who is closing the viewer did not ask to open a file browser.
        presentation.discard_pending_reveals()

        if getattr(self, "_recording_session", None) is not None:
            self._stop_recording()
        if getattr(self, "_manual_dive_trace", None) is not None:
            self._stop_manual_dive_trace(reason="viewer_closed")
        if self._ensure_slice_selection_controller().selection_active:
            self._finish_active_slice(closing=True)

        remaining_artifact_names = self._exit_capture_artifacts()
        if remaining_artifact_names:
            self._show_artifact_capture_status(
                presentation.exit_saving_status(remaining_artifact_names)
            )
        _LOG.info(
            "Waiting for %s before closing the viewer.",
            " and ".join(name.lower() for name in remaining_artifact_names)
            or "final capture cleanup",
        )

    def _begin_escape_capture_cancellation(self) -> bool:
        """Discard the active capture and close after its result is readable."""
        owner = self._capture_owner()
        if owner is None:
            return False

        workflow = self._ensure_capture_workflow()
        workflow.begin_escape_cancellation()
        self._defer_backend_close_request()
        self._reset_transient_input_state("canceling capture before close")

        # A discarded capture must never reveal a file that was queued by an
        # earlier publication attempt while the cancellation takes ownership.
        self._ensure_artifact_capture_presentation().discard_pending_reveals()
        handled = self._cancel_active_capture()
        owner_after_request = self._capture_owner()
        cancellation_rejected = bool(
            owner_after_request is not None
            and getattr(self, "_recording_status_kind", None) == "error"
        )
        if not handled or cancellation_rejected:
            # If cleanup could not start, keep the viewer open so the owned
            # writer is not silently converted back into save-on-shutdown.
            workflow.complete_escape_cancellation()
            if owner_after_request is None:
                self.on_close()
            return True

        _LOG.info(
            "Waiting for capture cancellation feedback before closing the viewer."
        )
        return True

    def _complete_exit_capture_finalization_if_ready(
        self,
        *,
        allow_unpresented_status: bool = False,
    ) -> bool:
        """Close once all exit-time capture writers have published their files."""
        workflow = self._ensure_capture_workflow()
        if not workflow.can_complete_exit_finalization(
            artifacts_pending=bool(self._exit_capture_artifacts()),
            now=time.perf_counter(),
            allow_unpresented_status=allow_unpresented_status,
        ):
            return False

        workflow.complete_exit_finalization()
        if getattr(self, "_slice_reveal_before_close", False):
            output_path = getattr(self, "_slice_reveal_output_path", None)
            if output_path:
                self._reveal_saved_output(output_path, output_kind="slice")
            self._slice_reveal_before_close = False
            self._slice_reveal_output_path = None
        self._complete_window_close()
        return True

    def _complete_escape_capture_cancellation_if_ready(self) -> bool:
        """Close after cancellation cleanup and its three-second result pause."""
        workflow = self._ensure_capture_workflow()
        if not workflow.can_complete_escape_cancellation(
            artifacts_pending=bool(self._exit_capture_artifacts()),
            confirmation_until=self._recording_status_until,
            now=time.perf_counter(),
        ):
            return False

        workflow.complete_escape_cancellation()
        self._complete_window_close()
        return True

    def _input_is_suppressed(self) -> bool:
        """Return whether viewer controls should ignore late input callbacks."""
        return bool(
            getattr(self, "_closing_requested", False)
            or self._capture_close_pending()
        )

    def _recording_hides_hud(self) -> bool:
        return self._recording_is_armed()

    def _toggle_recording(self) -> None:
        self._drain_recording_stop_results()
        if self._recording_stop_in_progress():
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().saving_status(
                    "Video",
                    cancelable=True,
                )
            )
            return

        if self._recording_session is not None:
            self._stop_recording(show_message=True, reveal_on_success=True)
            return

        if self._recording_countdown_until is not None:
            now = time.perf_counter()
            self._ensure_recording_controller().clear_countdown()
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().canceled_status("Video"),
                now=now,
            )
            _LOG.info("Recording countdown canceled.")
            return

        self._start_recording_countdown()

    def _cancel_recording_capture(self) -> bool:
        """Cancel recording countdown, capture, or pending output publication."""
        if self._exit_capture_finalization_active():
            return False
        self._ensure_recording_stop_state()
        self._drain_recording_stop_results()
        controller = self._ensure_recording_controller()
        if controller.countdown_until is not None:
            controller.clear_countdown()
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().canceled_status(
                    "Video",
                    after_escape=True,
                ),
                now=time.perf_counter(),
            )
            _LOG.info("Recording countdown canceled with Escape.")
            return True
        if getattr(self, "_recording_session", None) is not None:
            self._stop_recording(show_message=True, cancel_output=True)
            _LOG.info("Recording cancellation requested with Escape.")
            return True
        cancel_event = self._recording_stop_cancel_event
        if self._recording_stop_in_progress() and cancel_event is not None:
            cancel_event.set()
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().canceling_status(
                    "Video"
                )
            )
            _LOG.info("Pending recording publication canceled with Escape.")
            return True
        return False

    def _start_recording_countdown(self) -> None:
        if not self._has_map_loaded:
            return
        if self._capture_start_blocked(CaptureOwner.VIDEO):
            return
        if self._recording_target_if_available() is None:
            return

        self.color_picker.hide()
        if self.controls_overlay.is_manual_mode:
            self.controls_overlay.hide_help()
        now = time.perf_counter()
        self._ensure_recording_controller().start_countdown(
            now=now,
            start_number=self.RECORDING_COUNTDOWN_START_NUMBER,
        )
        _LOG.info(
            "Recording countdown started. Press %s+R to stop or Escape to cancel.",
            self._primary_shortcut_label(),
        )

    def _resolve_ffmpeg_path(self) -> str | None:
        viewer_settings = getattr(self, "_viewer_runtime_settings", None)
        if viewer_settings is None:
            return recording.resolve_ffmpeg_path()
        configured_path = viewer_settings.recording.ffmpeg_path
        if configured_path:
            return configured_path
        return recording.resolve_ffmpeg_path(environ={})

    def _recording_preflight(self) -> VideoRecordingPreflight:
        """Return one fresh recording probe paired with its policy decision."""
        return video_recording_preflight(
            self._recording_output_dir,
            ffmpeg_resolver=self._resolve_ffmpeg_path,
            platform_runtime=getattr(self, "_platform_runtime", None),
        )

    def _recording_target_if_available(self) -> VideoRecordingTarget | None:
        """Return a freshly-probed ffmpeg target or show the policy explanation."""
        preflight = self._recording_preflight()
        capability = preflight.capability
        decision = preflight.decision
        if not decision.allows_execution or capability.value is None:
            self._recording_unavailable(decision.explanation)
            return None
        return capability.value

    def _recording_unavailable(self, reason: str) -> None:
        message = f"Cannot start recording: {reason}"
        _LOG.warning(message)
        self._show_capture_status(
            "Recording unavailable",
            reason,
            kind="error",
            duration=3.4,
        )

    def _recording_capture_viewport(self) -> tuple[int, int, int, int]:
        for viewport in (
            getattr(self.ctx, "viewport", None),
            getattr(self.ctx.screen, "viewport", None),
        ):
            if viewport and len(viewport) >= 4:
                x, y, width, height = (int(v) for v in viewport[:4])
                if width > 0 and height > 0:
                    return x, y, width, height

        screen_size = getattr(self.ctx.screen, "size", None)
        if screen_size:
            width, height = screen_size
            return 0, 0, int(width), int(height)

        width, height = self.wnd.size
        return 0, 0, int(width), int(height)

    def _recording_framebuffer_size(self) -> tuple[int, int]:
        _x, _y, width, height = self._recording_capture_viewport()
        return width, height

    def _recording_output_size(self, width: int, height: int) -> tuple[int, int]:
        return recording.recording_output_size(
            width,
            height,
            self._recording_max_height,
        )

    def _release_recording_readback_framebuffer(self) -> None:
        self._ensure_recording_capture().release_framebuffer()
        self._sync_recording_capture_state_from_manager()

    def _discard_recording_staged_frames(self) -> int:
        dropped = self._ensure_recording_capture().discard_staged_frames()
        self._sync_recording_capture_state_from_manager()
        return dropped

    def _release_recording_readback_buffers(self) -> None:
        self._ensure_recording_capture().release_buffers()
        self._sync_recording_capture_state_from_manager()
        self._ensure_recording_controller().reset_frame_timings()

    def _create_recording_readback_framebuffer(
        self,
        capture_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> moderngl.Framebuffer | None:
        framebuffer = self._ensure_recording_capture().create_framebuffer(
            capture_size,
            output_size,
        )
        self._sync_recording_capture_state_from_manager()
        return framebuffer

    def _create_recording_readback_buffers(self, output_size: tuple[int, int]) -> None:
        self._ensure_recording_capture().create_buffers(output_size)
        self._sync_recording_capture_state_from_manager()

    def _start_recording_encoder(self) -> bool:
        recording_target = self._recording_target_if_available()
        if recording_target is None:
            self._ensure_recording_controller().clear_countdown()
            return False

        viewport = self._recording_capture_viewport()
        width, height = viewport[2], viewport[3]
        if width <= 0 or height <= 0:
            self._ensure_recording_controller().clear_countdown()
            return False

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(
            recording_target.output_directory,
            f"CaveViewerDive-{timestamp}.mp4",
        )
        output_width, output_height = self._recording_output_size(width, height)
        output_size = (output_width, output_height)
        try:
            readback_framebuffer = self._create_recording_readback_framebuffer(
                (width, height),
                output_size,
            )
            self._create_recording_readback_buffers(output_size)
        except Exception as exc:
            self._release_recording_readback_framebuffer()
            self._release_recording_readback_buffers()
            _LOG.warning(f"Cannot start recording: failed to create recording readback resources: {exc}")
            self._ensure_recording_controller().clear_countdown()
            self._show_capture_status(
                "Recording unavailable",
                "Could not prepare the recording framebuffer.",
                kind="error",
                duration=3.4,
            )
            return False

        # Recheck the on-demand gate immediately before launching ffmpeg. The
        # countdown and framebuffer setup can take long enough for a removable
        # drive or folder permission to change underneath the viewer.
        recording_target = self._recording_target_if_available()
        if recording_target is None:
            self._release_recording_readback_framebuffer()
            self._release_recording_readback_buffers()
            self._ensure_recording_controller().clear_countdown()
            return False

        output_path = os.path.join(
            recording_target.output_directory,
            f"CaveViewerDive-{timestamp}.mp4",
        )

        try:
            session = recording.start_encoder_session(
                ffmpeg_path=recording_target.ffmpeg_path,
                output_path=output_path,
                output_size=output_size,
                viewport=viewport,
                fps=self._recording_fps,
                crf=self._recording_crf,
                raw_pix_fmt=self.RECORDING_RAW_PIX_FMT,
                popen_startup_kwargs=(
                    self._active_recording_process_adapter().encoder_popen_kwargs()
                ),
            )
        except (OSError, RuntimeError) as exc:
            _LOG.warning(f"Cannot start recording: {exc}")
            self._release_recording_readback_framebuffer()
            self._release_recording_readback_buffers()
            self._ensure_recording_controller().clear_countdown()
            return False

        self._recording_session = session
        self._recording_frame_queue = session.frame_queue
        self._recording_output_path = session.output_path
        self._recording_size = session.output_size
        self._recording_viewport = session.viewport
        self._recording_readback_framebuffer = readback_framebuffer
        now = time.perf_counter()
        self._ensure_recording_controller().mark_encoder_started(now=now)
        _LOG.info(
            f"Recording started: {output_path} "
            f"capture_viewport={viewport} readback_size={output_width}x{output_height} "
            f"output_size={output_width}x{output_height} "
            f"raw_pix_fmt={self.RECORDING_RAW_PIX_FMT} "
            f"readback_buffers={len(self._recording_readback_slots)}"
        )
        return True

    def _recording_signal_writer_stop(self, frame_queue: queue.Queue | None) -> None:
        recording.signal_writer_stop(frame_queue)

    def _recording_drop_frames(self, count: int = 1) -> None:
        if self._ensure_recording_controller().drop_frames(count):
            _LOG.warning("Recording encoder is falling behind; dropping video frames.")

    def _recording_due_frame_slots(self, now: float, next_frame_time: float | None) -> int:
        return self._ensure_recording_controller().due_frame_slots(
            now=now,
            next_frame_time=next_frame_time,
        )

    def _recording_enqueue_frame(self, frame: bytes) -> bool:
        frame_queue = self._recording_frame_queue
        if frame_queue is None:
            self._stop_recording()
            return False

        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            self._recording_drop_frames()
            return False
        return True

    def _recording_display_path(self, path: str | None) -> str | None:
        return recording.recording_display_path(path)

    def _show_capture_status(
        self,
        message: str,
        detail: str | None = None,
        *,
        kind: str = "info",
        duration: float | None = 2.8,
        now: float | None = None,
    ) -> None:
        self._ensure_recording_controller().show_status(
            message,
            detail=detail,
            kind=kind,
            duration=duration,
            now=time.perf_counter() if now is None else now,
        )

    def _show_artifact_capture_status(
        self,
        status: ArtifactCaptureStatus,
        *,
        now: float | None = None,
    ) -> None:
        """Present one shared artifact-capture status through the HUD."""
        self._show_capture_status(
            status.message,
            status.detail,
            kind=status.kind,
            duration=status.duration,
            now=now,
        )

    def _stop_recording(
        self,
        *,
        show_message: bool = False,
        reveal_on_success: bool = False,
        cancel_output: bool = False,
    ) -> None:
        self._ensure_recording_stop_state()
        self._drain_recording_stop_results()
        if self._recording_stop_in_progress():
            if cancel_output and self._recording_stop_cancel_event is not None:
                self._recording_stop_cancel_event.set()
            return

        session = self._recording_session

        self._ensure_recording_controller().clear_countdown()
        self._recording_session = None
        self._recording_output_path = None
        self._recording_size = None
        self._recording_viewport = None
        self._release_recording_readback_buffers()
        self._release_recording_readback_framebuffer()
        self._recording_next_frame_time = None
        self._recording_frame_queue = None

        if session is None:
            return

        cancel_event = threading.Event()
        if cancel_output:
            cancel_event.set()
        self._recording_stop_cancel_event = cancel_event
        session.signal_writer_stop(discard_pending=cancel_output)
        work = session.stop_work(
            show_message=show_message,
            reveal_on_success=reveal_on_success,
            cancel_event=cancel_event,
        )
        self._recording_stop_thread = recording.start_stop_finalizer(
            work,
            result_queue=self._recording_stop_results,
            stderr_text=session.stderr_text,
            writer_error=lambda: session.writer_error,
            dropped_frames=lambda: self._recording_dropped_frames,
            logger=_LOG,
        )
        if show_message:
            self._show_artifact_capture_status(
                (
                    self._ensure_artifact_capture_presentation().canceling_status(
                        "Video"
                    )
                    if cancel_output
                    else self._ensure_artifact_capture_presentation().saving_status(
                        "Video",
                        cancelable=True,
                    )
                )
            )

    def _drain_recording_stop_results(self) -> None:
        self._ensure_recording_stop_state()
        while True:
            try:
                result = self._recording_stop_results.get_nowait()
            except queue.Empty:
                break
            self._apply_recording_stop_result(result)
            self._recording_stop_thread = None
            self._recording_stop_cancel_event = None

    def _reveal_saved_output(
        self,
        output_path: str | None,
        *,
        output_kind: str,
    ) -> None:
        """Best-effort native reveal after a writer has published final output."""
        if not output_path:
            return
        try:
            self._active_saved_artifact_reveal_adapter().reveal_saved_artifact(
                output_path
            )
        except Exception as exc:
            _LOG.warning(
                "Could not reveal saved %s %s: %s",
                output_kind,
                output_path,
                exc,
            )

    def _drain_due_saved_artifact_reveals(self, *, now: float | None = None) -> None:
        """Reveal artifacts only after their shared success message has been visible."""
        controller = self._ensure_artifact_capture_presentation()
        if not controller.has_pending_reveals:
            return
        current_time = time.perf_counter() if now is None else now
        for request in controller.take_due_reveals(
            now=current_time
        ):
            self._reveal_saved_output(
                request.output_path,
                output_kind=request.artifact_name.lower(),
            )

    def _apply_recording_stop_result(self, result: _RecordingStopResult) -> None:
        self._ensure_recording_controller().reset_after_stop_result()

        if result.canceled:
            if result.cleanup_error:
                _LOG.warning(
                    "Canceled recording cleanup failed: %s",
                    result.cleanup_error,
                )
                if result.show_message and not self._exit_capture_finalization_active():
                    self._show_artifact_capture_status(
                        self._ensure_artifact_capture_presentation().cancellation_failed_status(
                            "Video",
                            "The partial video could not be removed.",
                        )
                    )
            else:
                _LOG.info("Recording canceled and partial output removed.")
                if result.show_message and not self._exit_capture_finalization_active():
                    self._show_artifact_capture_status(
                        self._ensure_artifact_capture_presentation().canceled_status(
                            "Video",
                            after_escape=True,
                        )
                    )
            return

        if result.returncode == 0:
            _LOG.info(f"Recording saved: {result.output_path}")
            if result.dropped_frames:
                _LOG.warning(f"Recording saved after dropping {result.dropped_frames} frame(s).")
            if result.show_message and not self._exit_capture_finalization_active():
                now = time.perf_counter()
                status = self._ensure_artifact_capture_presentation().saved_status(
                    "Video",
                    result.output_path,
                    now=now,
                    reveal=result.reveal_on_success,
                )
                self._show_artifact_capture_status(status, now=now)
        else:
            if result.stderr_text and result.writer_error:
                detail = f": {result.stderr_text}; writer_error={result.writer_error}"
            elif result.stderr_text:
                detail = f": {result.stderr_text}"
            elif result.writer_error:
                detail = f": writer_error={result.writer_error}"
            else:
                detail = ""
            _LOG.warning(f"Recording encoder exited with code {result.returncode}{detail}")
            if result.show_message and not self._exit_capture_finalization_active():
                self._show_artifact_capture_status(
                    self._ensure_artifact_capture_presentation().failed_status(
                        "Video",
                        self._recording_failure_detail(result.stderr_text),
                    )
                )

    def _recording_failure_detail(self, stderr_text: str) -> str:
        return recording.recording_failure_detail(stderr_text)

    def _recording_capture_state(self) -> tuple[tuple[int, int], tuple[int, int, int, int], int]:
        return self._ensure_recording_capture().capture_state()

    def _recording_free_readback_slot(self) -> _RecordingReadbackSlot | None:
        return self._ensure_recording_capture().free_readback_slot()

    def _recording_copy_to_readback_framebuffer(
        self,
        readback_framebuffer: moderngl.Framebuffer,
        output_size: tuple[int, int],
        capture_viewport: tuple[int, int, int, int],
    ) -> None:
        self._ensure_recording_capture().copy_to_readback_framebuffer(
            readback_framebuffer,
            output_size,
            capture_viewport,
        )

    def _recording_stage_frame(
        self,
        render_frame: Callable[[moderngl.Framebuffer, tuple[int, int]], None] | None = None,
    ) -> bool:
        try:
            return self._ensure_recording_capture().stage_frame(
                render_frame=render_frame,
            )
        finally:
            self._sync_recording_capture_state_from_manager()

    def _recording_drain_staged_frames(self) -> float:
        try:
            return self._ensure_recording_capture().drain_staged_frames(
                frame_queue=self._recording_frame_queue,
                enqueue_frame=self._recording_enqueue_frame,
                stop_recording=self._stop_recording,
            )
        finally:
            self._sync_recording_capture_state_from_manager()

    def _recording_update_after_scene(
        self,
        now: float,
        *,
        render_frame: Callable[[moderngl.Framebuffer, tuple[int, int]], None] | None = None,
    ) -> float:
        controller = self._ensure_recording_controller()
        controller.reset_frame_timings()

        if controller.countdown_until is not None:
            if not controller.countdown_ready(now=now):
                return 0.0
            if not self._start_recording_encoder():
                return 0.0

        session = self._recording_session
        if session is None:
            return 0.0

        if session.stopped_before_finalization():
            _LOG.warning("Recording encoder stopped before recording was finalized.")
            self._stop_recording(show_message=True)
            return 0.0

        if self._recording_viewport != self._recording_capture_viewport():
            _LOG.warning("Recording stopped because the window size changed.")
            self._stop_recording()
            return 0.0

        read_ms = 0.0
        try:
            drain_ms = self._recording_drain_staged_frames()
            self._recording_last_drain_ms = drain_ms
            read_ms += drain_ms
        except (OSError, moderngl.Error) as exc:
            _LOG.warning(f"Recording stopped because frame capture failed: {exc}")
            self._stop_recording(show_message=True)
            return read_ms
        if self._recording_session is None:
            return read_ms

        next_frame_time = self._recording_next_frame_time
        if next_frame_time is not None and now < next_frame_time:
            return read_ms

        frame_slots = self._recording_due_frame_slots(now, next_frame_time)
        frame_queue = self._recording_frame_queue
        if frame_queue is None:
            self._stop_recording(show_message=True)
            return read_ms

        if frame_queue.full():
            staged_frames = self._discard_recording_staged_frames()
            self._recording_drop_frames(frame_slots + staged_frames)
            controller.advance_next_frame_time(now=now, frame_slots=frame_slots)
            return read_ms

        try:
            self._recording_drop_frames(frame_slots - 1)

            t_stage = time.perf_counter()
            if not self._recording_stage_frame(render_frame=render_frame):
                self._recording_drop_frames()
            stage_ms = (time.perf_counter() - t_stage) * 1000.0
            self._recording_last_stage_ms = stage_ms
            read_ms += stage_ms
            controller.advance_next_frame_time(now=now, frame_slots=frame_slots)
            return read_ms
        except (OSError, moderngl.Error) as exc:
            _LOG.warning(f"Recording stopped because frame capture failed: {exc}")
            self._stop_recording(show_message=True)
            return read_ms

    def _render_countdown_overlay(
        self,
        *,
        now: float,
        controller: (
            RecordingStateController
            | ManualDiveTraceStateController
            | SliceSelectionController
        ),
        start_number: int,
        title: str,
        note: str,
    ) -> None:
        """Render the shared import-style countdown used before capture begins."""
        display = controller.countdown_display(
            now=now,
            start_number=start_number,
        )
        self._render_recording_countdown_scrim(self.wnd.size)
        self.import_progress_panel.draw_countdown_number(
            center_x=self.wnd.size[0] / 2.0,
            center_y=self.wnd.size[1] / 2.0,
            window_size=self.wnd.size,
            number=display.number,
            progress=display.progress,
            fixed_text_scale=self.UI_TEXT_SCALE,
            stage=title,
            note=note,
        )

    def _countdown_cancel_note(self, shortcut_key: str) -> str:
        """Show both the capture toggle and normalized Escape cancellation."""
        return (
            f"Press {self._primary_shortcut_label()}+{shortcut_key} again to stop. "
            "Press Esc to cancel."
        )

    def _print_texture_diagnostics(self, manifest: dict, textures_dir: str) -> None:
        """Print a one-time texture summary to console on map load."""
        from PIL import Image
        import io as _io

        mats = manifest.get("mtl_materials", {})
        _LOG.info(f"Texture diagnostics: {len(mats)} materials, "
              f"{len(manifest.get('chunks', {}))} total chunks")

        # Deduplicate: multiple material names can share one file/bytes blob.
        seen: dict[object, tuple[str, tuple[int, int]]] = {}  # key -> (first_mat, size)
        missing = 0
        embedded = 0

        for mat_name, file_or_bytes in mats.items():
            if file_or_bytes is None:
                missing += 1
                continue
            if file_or_bytes in seen:
                continue
            if isinstance(file_or_bytes, bytes):
                embedded += 1
                try:
                    img = Image.open(_io.BytesIO(file_or_bytes))
                    seen[file_or_bytes] = (mat_name, img.size)
                except Exception:
                    seen[file_or_bytes] = (mat_name, (0, 0))
            else:
                import os as _os
                path = _os.path.join(textures_dir, file_or_bytes)
                try:
                    with Image.open(path) as img:
                        seen[file_or_bytes] = (mat_name, img.size)
                except Exception:
                    seen[file_or_bytes] = (mat_name, (0, 0))

        sizes = [sz for _, sz in seen.values() if sz != (0, 0)]
        unique_files = len(seen)
        total_px = sum(w * h for w, h in sizes)
        total_mb = total_px * 3 / (1024 * 1024)  # RGB uncompressed

        size_counts: dict[tuple, int] = {}
        for sz in sizes:
            size_counts[sz] = size_counts.get(sz, 0) + 1

        _LOG.info(f"  Unique texture files : {unique_files}"
                  + (f" ({embedded} embedded)" if embedded else ""))
        if missing:
            _LOG.info(f"  Materials with no texture: {missing}")
        for sz, count in sorted(size_counts.items(), key=lambda x: -x[1]):
            _LOG.info(f"  {sz[0]}x{sz[1]} : {count} texture(s)")
        _LOG.info(f"  Uncompressed RGB total  : {total_mb:.0f} MB")
        max_dim = max((max(w, h) for w, h in sizes), default=0)
        # Rough atlas fit: next power-of-2 square that holds total_px
        import math as _math
        atlas_side = 2 ** _math.ceil(_math.log2(_math.sqrt(total_px))) if total_px > 0 else 0
        _LOG.info(f"  Estimated atlas needed  : {atlas_side}x{atlas_side} px "
              f"({atlas_side*atlas_side*3/1024/1024:.0f} MB)")

    def _recorded_dive_is_active(self) -> bool:
        controller = getattr(self, "_recorded_dive_controller", None)
        return controller is not None and controller.active

    def _recorded_dive_is_paused(self) -> bool:
        """Report whether an active dive is in orientation-only inspection mode."""
        controller = getattr(self, "_recorded_dive_controller", None)
        return bool(
            controller is not None
            and controller.active
            and getattr(controller, "state", None)
            is recorded_dive.RecordedDivePlaybackState.PAUSED
        )

    def _recorded_dive_prefetch_cells(self) -> frozenset[tuple[int, int, int]]:
        """Build a bounded, chronological chunk tube ahead of trace time."""
        controller = getattr(self, "_recorded_dive_controller", None)
        world = getattr(self, "world", None)
        if controller is None or world is None or not controller.active:
            return frozenset()

        poses = controller.lookahead_poses(_RECORDED_DIVE_LOOKAHEAD_SECONDS)
        if not poses:
            return frozenset()
        chunk_size = max(1e-6, float(world.config.chunk_size))
        sample_step_m = max(0.25, chunk_size * 0.5)
        centers: list[tuple[int, int, int]] = []
        previous_pose = poses[0]
        centers.append(
            world.cell_for_position(
                np.asarray(previous_pose.position, dtype=np.float32)
            )
        )
        for pose in poses[1:]:
            start = np.asarray(previous_pose.position, dtype=np.float64)
            end = np.asarray(pose.position, dtype=np.float64)
            segment = end - start
            distance = float(np.linalg.norm(segment))
            steps = 1
            if pose.record_kind != "discontinuity":
                steps = max(1, int(math.ceil(distance / sample_step_m)))
            for step in range(1, steps + 1):
                position = end if steps == 1 else start + segment * (step / steps)
                center = world.cell_for_position(
                    np.asarray(position, dtype=np.float32)
                )
                if not centers or center != centers[-1]:
                    centers.append(center)
            previous_pose = pose

        wanted: set[tuple[int, int, int]] = set()
        for center in centers:
            nearby = sorted(
                world.available_cells_in_radius(
                    center,
                    _RECORDED_DIVE_PREFETCH_RADIUS_CELLS,
                ),
                key=lambda cell: (
                    (cell[0] - center[0]) ** 2
                    + (cell[1] - center[1]) ** 2
                    + (cell[2] - center[2]) ** 2,
                    cell,
                ),
            )
            for cell in nearby:
                wanted.add(cell)
                if len(wanted) >= _RECORDED_DIVE_PREFETCH_CELL_CAP:
                    return frozenset(wanted)
        return frozenset(wanted)

    def _refresh_recorded_dive_prefetch(self) -> None:
        world = getattr(self, "world", None)
        if world is None:
            return
        cells = self._recorded_dive_prefetch_cells()
        if cells == getattr(self, "_recorded_dive_prefetch_cell_set", frozenset()):
            return
        self._recorded_dive_prefetch_cell_set = cells
        world.set_prefetch_wanted_cells(cells)

    def _recorded_dive_chunks_ready(self, *, now: float) -> bool:
        """Require GPU-resident geometry around the next authoritative pose."""
        controller = getattr(self, "_recorded_dive_controller", None)
        world = getattr(self, "world", None)
        if controller is None or world is None:
            return False
        if (
            not getattr(self, "_initial_chunks_loaded", False)
            or not getattr(self, "_initial_visual_ready", False)
        ):
            return False

        candidate_pose = controller.trace.pose_at(
            controller.candidate_elapsed(now=now)
        )
        center = world.cell_for_position(
            np.asarray(candidate_pose.position, dtype=np.float32)
        )
        required = set(
            world.available_cells_in_radius(
                center,
                _RECORDED_DIVE_PREFETCH_RADIUS_CELLS,
            )
        )
        if not required:
            required = set(
                world.available_cells_in_radius(
                    center,
                    max(1, int(world.config.load_radius_cells)),
                )
            )

        lock = getattr(world, "_lock", None)
        if lock is None:
            loaded = set(getattr(world, "loaded_cells", ()))
            failed = set(getattr(world, "_failed_cells", {}))
        else:
            with lock:
                loaded = set(getattr(world, "loaded_cells", ()))
                failed = set(getattr(world, "_failed_cells", {}))
        failed_required = required & failed
        if failed_required:
            _LOG.error(
                "Recorded Dive stopped because %d required map chunk(s) failed to load.",
                len(failed_required),
            )
            self._stop_recorded_dive(reason="chunk_load_failed")
            return False
        return required.issubset(loaded)

    def _update_recorded_dive(self, *, now: float) -> None:
        controller = getattr(self, "_recorded_dive_controller", None)
        if controller is None or not controller.active:
            return
        self._refresh_recorded_dive_prefetch()
        previous_state = controller.state
        chunks_ready = (
            True
            if previous_state is recorded_dive.RecordedDivePlaybackState.PAUSED
            else self._recorded_dive_chunks_ready(now=now)
        )
        current_state = controller.update(
            self.camera,
            now=now,
            chunks_ready=chunks_ready,
        )
        if current_state is recorded_dive.RecordedDivePlaybackState.BUFFERING:
            if previous_state is not current_state:
                _LOG.info("Recorded Dive paused its clock while map chunks load.")
            return
        if (
            previous_state is recorded_dive.RecordedDivePlaybackState.BUFFERING
            and current_state is recorded_dive.RecordedDivePlaybackState.PLAYING
        ):
            if self.controls_overlay.is_waiting_for_begin:
                self.controls_overlay.dismiss_begin_screen()
            _LOG.info("Recorded Dive playback started/resumed.")
        if current_state is recorded_dive.RecordedDivePlaybackState.FINISHED:
            if self.controls_overlay.is_waiting_for_begin:
                self.controls_overlay.dismiss_begin_screen()
            self._recorded_dive_prefetch_cell_set = frozenset()
            self.world.set_prefetch_wanted_cells(())
            _LOG.info(
                "Recorded Dive completed: %.1f seconds, %d recorded poses.",
                controller.trace.duration_s,
                len(controller.trace.poses),
            )

    def _stop_recorded_dive(self, *, reason: str) -> bool:
        controller = getattr(self, "_recorded_dive_controller", None)
        if controller is None:
            return False
        was_active = controller.active
        controller.stop()
        world = getattr(self, "world", None)
        if world is not None:
            world.set_prefetch_wanted_cells(())
        self._recorded_dive_prefetch_cell_set = frozenset()
        if was_active:
            _LOG.info("Recorded Dive stopped: %s.", reason)
        return was_active

    def _toggle_recorded_dive_pause(self) -> bool:
        controller = getattr(self, "_recorded_dive_controller", None)
        if controller is None or not controller.active:
            return False
        now = time.perf_counter()
        if controller.state is recorded_dive.RecordedDivePlaybackState.PAUSED:
            return controller.resume(self.camera, now=now)
        return controller.pause(now=now)

    def _render_dive_status(self, window_size: tuple[int, int]) -> None:
        """Render status for Recorded Dive playback."""
        self._render_recorded_dive_progress(window_size)

    @staticmethod
    def _recorded_dive_time_label(elapsed_s: float) -> str:
        total_seconds = max(0, int(round(float(elapsed_s))))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:d}:{seconds:02d}"

    def _render_recorded_dive_progress(
        self,
        window_size: tuple[int, int],
    ) -> bool:
        controller = getattr(self, "_recorded_dive_controller", None)
        if controller is None or not controller.active:
            return False
        elapsed = self._recorded_dive_time_label(controller.elapsed_s)
        duration = self._recorded_dive_time_label(controller.trace.duration_s)
        if controller.state is recorded_dive.RecordedDivePlaybackState.BUFFERING:
            note = f"Loading nearby cave chunks… {elapsed} / {duration}"
        elif controller.state is recorded_dive.RecordedDivePlaybackState.PAUSED:
            note = (
                f"Paused for inspection at {elapsed} / {duration}. "
                "Look around; Space resumes."
            )
        else:
            note = f"{elapsed} / {duration}. Space pauses; movement takes control."
        self._render_dive_status_prompt(
            window_size,
            title="Recorded Dive",
            note=note,
        )
        return True

    def _render_dive_status_prompt(
        self,
        window_size: tuple[int, int],
        *,
        title: str,
        note: str,
    ) -> None:
        """Draw the small top prompt for Recorded Dive playback state."""
        w, h = window_size
        panel_w = min(
            max(
                self.DIVE_STATUS_PANEL_MIN_WIDTH,
                w * self.DIVE_STATUS_PANEL_WIDTH_FRACTION,
            ),
            w - 48.0,
        )
        panel_h = self.DIVE_STATUS_PANEL_HEIGHT
        x0 = (w - panel_w) / 2.0
        y0 = 30.0
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        verts = []

        def px_to_ndc(x: float, y: float) -> tuple[float, float]:
            return (x / w) * 2.0 - 1.0, 1.0 - (y / h) * 2.0

        def add_quad_px(
            qx0: float,
            qy0: float,
            qx1: float,
            qy1: float,
            rgba: tuple[float, float, float, float],
        ) -> None:
            nx0, ny0 = px_to_ndc(qx0, qy0)
            nx1, ny1 = px_to_ndc(qx1, qy1)
            top, bottom = max(ny0, ny1), min(ny0, ny1)
            left, right = min(nx0, nx1), max(nx0, nx1)
            quad = [
                (left, bottom), (right, bottom), (right, top),
                (left, bottom), (right, top), (left, top),
            ]
            for vx, vy in quad:
                verts.append((vx, vy, *rgba))

        def add_centered_text(
            text: str,
            y: float,
            pixel_size: float,
            rgba: tuple[float, float, float, float],
            *,
            max_width: float,
        ) -> float:
            text = " ".join(str(text or "").split())
            if not text:
                return 0.0
            min_pixel_size = bitmap_font.pixel_size_at_text_scale(
                1.20,
                self.UI_TEXT_SCALE,
            )
            pixel_size = bitmap_font.pixel_size_at_text_scale(
                pixel_size,
                self.UI_TEXT_SCALE,
            )
            bounds = bitmap_font.text_bounds_px(text, pixel_size)
            text_w = bounds[2] - bounds[0]
            if text_w > max_width:
                pixel_size = max(min_pixel_size, pixel_size * max_width / text_w)
                bounds = bitmap_font.text_bounds_px(text, pixel_size)
                text_w = bounds[2] - bounds[0]
            text_h = bounds[3] - bounds[1]
            origin_x = (w - text_w) / 2.0 - bounds[0]
            origin_y = y - bounds[1]
            r, g, b, a = rgba
            for glyph in bitmap_font.iter_text_pixels(text, origin_x, origin_y, pixel_size):
                px0, py0, px1, py1 = glyph[0], glyph[1], glyph[2], glyph[3]
                glyph_alpha = glyph[4] if len(glyph) > 4 else 1.0
                add_quad_px(px0, py0, px1, py1, (r, g, b, a * glyph_alpha))
            return text_h

        add_quad_px(x0, y0, x1, y1, (0.025, 0.028, 0.040, 0.86))
        border = 2.0
        border_color = (0.8980, 0.6314, 0.1216, 0.95)
        add_quad_px(x0, y0, x1, y0 + border, border_color)
        add_quad_px(x0, y1 - border, x1, y1, border_color)
        add_quad_px(x0, y0, x0 + border, y1, border_color)
        add_quad_px(x1 - border, y0, x1, y1, border_color)

        title_h = add_centered_text(
            title,
            y0 + 24.0,
            self.DIVE_STATUS_TITLE_PIXEL_SIZE,
            (0.9490, 0.8510, 0.5490, 1.0),
            max_width=panel_w - 48.0,
        )
        add_centered_text(
            note,
            y0 + 24.0 + title_h + 18.0,
            self.DIVE_STATUS_NOTE_PIXEL_SIZE,
            (0.835, 0.855, 0.86, 0.92),
            max_width=panel_w - 48.0,
        )

        data = np.array(verts, dtype=np.float32)
        if len(verts) > self._status_panel_max_verts:
            self._status_panel_vbo.release()
            self._status_panel_max_verts = max(self._status_panel_max_verts * 2, len(verts))
            self._status_panel_vbo = self.ctx.buffer(reserve=self._status_panel_max_verts * 6 * 4)
            self._status_panel_vao = self.ctx.vertex_array(
                self._hud_panel_program,
                [(self._status_panel_vbo, "2f 4f", "in_pos", "in_color")],
            )
        self._status_panel_vbo.write(data.tobytes())
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self._status_panel_vao.render(moderngl.TRIANGLES, vertices=len(verts))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def _teardown_current_map(self, *, final_shutdown: bool = False) -> None:
        """
        Cleanly releases everything specific to the CURRENTLY loaded map
        before _load_map() builds a new one -- stops StreamingWorld's
        background threads, then
        releases every currently-resident chunk's GPU buffers/VAOs and
        decrements the texture manager's reference counts via the exact
        same _on_chunk_unload() path used during normal streaming (so
        there's no separate cleanup logic to keep in sync with the
        regular unload path). The texture manager itself is then simply
        discarded -- a fresh one is constructed for the new map rather
        than trying to partially reuse the old one.

        Safe to call even if no map was ever loaded yet (e.g. the very
        first import, triggered from _run_pending_import, completing for
        the first time rather than switching away from an existing map)
        -- there's nothing to tear down in that case, so this just
        returns immediately rather than crashing on self.world not
        existing yet.

        Shutdown uses a finite worker-join timeout even during final window
        close.  Streaming workers are CPU/I/O-only and never issue OpenGL
        commands; if one is stuck in external I/O, StreamingWorld records and
        logs the unjoined worker instead of letting the viewer close callback
        block forever.
        """
        self._cancel_texture_validation()
        if not self._has_map_loaded:
            return

        self._stop_manual_dive_trace(
            reason="viewer_closed" if final_shutdown else "map_changed"
        )
        slice_selection = self._ensure_slice_selection_controller()
        if slice_selection.countdown_active:
            slice_selection.cancel_countdown()
            self._clear_slice_context()
        elif slice_selection.selection_active:
            slice_selection.cancel_selection()
            self._clear_slice_context()
        self._stop_recorded_dive(
            reason="viewer_closed" if final_shutdown else "map_changed"
        )
        self._stop_recording()
        # Keep this callback bounded: on_close() runs inside the window/render
        # event path, and an unbounded join here can leave the viewer visually
        # frozen if a streaming worker is stuck in disk or callback code.
        self.world.shutdown(timeout=_VIEWER_STREAMING_SHUTDOWN_TIMEOUT_SECONDS)

        upload_manager = getattr(self, "_chunk_upload_manager", None)
        if upload_manager is not None:
            upload_manager.unload_all()
            self._sync_chunk_upload_state_from_manager(upload_manager)
        else:
            for cell in list(getattr(self, "_chunk_upload_states", {}).keys()):
                self._on_chunk_unload(cell)

            for cell in list(self._chunk_gpu_objects.keys()):
                self._on_chunk_unload(cell)

        # belt-and-suspenders: if anything was somehow left behind (it
        # shouldn't be, given the loop above), don't carry it into the
        # next map's state
        self._chunk_gpu_objects.clear()
        self._chunk_upload_states.clear()
        self._chunk_normal_cache.clear()
        self._chunk_aabbs.clear()
        self._chunk_upload_manager = None
        self._invalidate_visible_chunk_cache()
        self._recorded_dive_trace = None
        self._recorded_dive_controller = None

        if hasattr(self, "texture_manager") and self.texture_manager is not None:
            self.texture_manager.shutdown()

        if self.minimap is not None:
            try:
                self.minimap.release()
            except Exception:
                pass

        self._has_map_loaded = False
        self.world = None
        self.camera = None
        self.minimap = None
        self.texture_manager = None

    def _release_window_resources(self) -> None:
        """Release non-map GPU/UI resources when closing the viewer window."""
        if self._window_resources_released:
            return
        self._window_resources_released = True

        self._cancel_texture_validation()
        self._stop_recording()
        self._keys_down.clear()
        self._mouse_look_active = False
        self._mouse_look_left_option_active = False
        self._last_mouse_pos = None

        # on_close() asks the import controller to stop any active import before
        # resource teardown. Drop remaining refs here so detached fallback
        # messages cannot be applied after the window closes.
        self._import_active = False
        self._import_queue = None
        self._import_thread = None
        self._import_command_queue = None

        def _release_attr(obj, attr_name: str) -> None:
            resource = getattr(obj, attr_name, None)
            if resource is None:
                return
            if hasattr(resource, "release"):
                try:
                    resource.release()
                except Exception:
                    pass
            try:
                setattr(obj, attr_name, None)
            except Exception:
                pass

        components = (
            "light_stepper",
            "render_distance_stepper",
            "ambient_stepper",
            "render_mode_buttons",
            "controls_overlay",
            "color_picker",
            "import_progress_panel",
            "minimap",
        )
        for name in components:
            obj = getattr(self, name, None)
            if obj is None:
                continue
            _release_attr(obj, "_vao")
            _release_attr(obj, "_vbo")
            _release_attr(obj, "program")
            if hasattr(obj, "release"):
                try:
                    obj.release()
                except Exception:
                    pass
            setattr(self, name, None)

        _release_attr(self, "program")
        _release_attr(self, "_hud_panel_vao")
        _release_attr(self, "_hud_panel_vbo")
        _release_attr(self, "_status_panel_vao")
        _release_attr(self, "_status_panel_vbo")
        _release_attr(self, "_hud_panel_program")

    def load_new_map(
        self,
        cache_dir: str,
        textures_dir: str,
        manifest: dict,
        *,
        source_dir: str | None = None,
    ) -> None:
        """
        Switches the viewer to a different map without closing the
        window -- called by the OPEN button's click handler once a new
        folder has been picked and imported/cached (see
        caveviewer.app's find_input_files/import_and_cache, reused as-is
        rather than duplicated here).

        Order matters: tear down the OLD map's GPU/thread state fully
        before constructing any NEW state, rather than interleaving the
        two -- this guarantees the old map's resources are genuinely
        released (not just about to be overwritten by Python references
        moving on, which would leak the GPU-side buffers/textures since
        those aren't cleaned up by garbage collection alone).
        """
        self._teardown_current_map()
        self._load_map(
            cache_dir,
            textures_dir,
            manifest,
            map_root=source_dir,
        )
        self._has_map_loaded = True
        try:
            from caveviewer.gui.map_history import remember_recent_map_path

            remember_recent_map_path(source_dir or textures_dir)
        except Exception:
            pass

    def _handle_open_button_click(self) -> None:
        """
        Full OPEN button flow: shows the folder-browse dialog (same one
        used at startup), detects which supported format (.obj or
        .glb) the selected folder contains, imports/caches it if there's
        no valid cache yet (showing the progress panel while that one-
        time work runs), and finally calls load_new_map() to actually
        switch.

        Any failure along the way (cancelled dialog, no supported model
        file found, import error) prints a clear message and leaves the
        CURRENTLY loaded map running untouched -- a failed attempt to
        open a different map should never take down the map you already
        had open and were presumably still looking at.
        """
        slice_selection = self._ensure_slice_selection_controller()
        if (
            slice_selection.countdown_active
            or slice_selection.selection_active
            or self._ensure_slice_export_controller().active
        ):
            self._show_capture_status(
                "Slice in progress",
                "Finish or cancel the slice before opening another map.",
                kind="info",
                duration=3.0,
            )
            return
        try:
            folder = pick_folder_dialog(
                platform_runtime=getattr(self, "_platform_runtime", None)
            )
        except DesktopServiceError as exc:
            _LOG.warning("Map folder selection unavailable: %s", exc)
            return
        if not folder:
            _LOG.info("Open cancelled -- no folder selected.")
            return

        _LOG.info(f"Opening new map from: {os.path.abspath(folder)}")

        try:
            open_target = resolve_selected_map_folder(folder)
        except FileNotFoundError as e:
            _LOG.warning(f"Could not open this folder: {e}")
            return
        except Exception as manifest_err:
            _LOG.error(f"Failed to load the selected prebuilt map: {manifest_err}")
            return

        if open_target.is_prebuilt_cache:
            _LOG.info(f"Found cache manifest in selected directory: {open_target.cache_dir}")
            _LOG.info(f"Switching to prebuilt map: {open_target.map_name}")
            _LOG.info(f"Using cache directory: {open_target.cache_dir}")
            self._ensure_map_opening_progress_session().begin_cached(
                open_target.map_name,
                new_operation=True,
            )
            self.load_new_map(
                open_target.cache_dir,
                open_target.textures_dir,
                open_target.manifest,
                source_dir=open_target.source_dir,
            )
            _LOG.info(f"Now viewing: {open_target.map_name}")
            return

        self._start_import_async(
            open_target.model_descriptor,
            open_target.textures_dir,
            open_target.map_name,
            is_startup=False,
        )

    def _import_model_format_from_descriptor(self, model_descriptor: dict) -> str | None:
        return self._ensure_import_controller().import_model_format_from_descriptor(
            model_descriptor
        )

    def _default_import_progress_note(self) -> str:
        return self._ensure_import_controller().default_progress_note()

    def _set_import_progress_message(self, title: str, note: str) -> None:
        self._ensure_import_controller().set_progress_message(title, note)

    def _update_import_progress_message_for_stage(self, stage: str) -> None:
        self._ensure_import_controller().update_progress_message_for_stage(stage)

    def _show_import_pause_notice(
        self,
        map_name: str,
        *,
        close_after: bool = False,
        duration: float = 6.0,
    ) -> None:
        self._ensure_import_controller().show_pause_notice(
            map_name,
            close_after=close_after,
            duration=duration,
        )

    def _clear_import_pause_notice(self) -> bool:
        return self._ensure_import_controller().clear_pause_notice()

    def _render_import_pause_notice_if_active(self) -> bool:
        return self._ensure_import_controller().render_pause_notice_if_active(
            self.import_progress_panel,
            self.wnd,
            _viewer_ui_surface_size(self.wnd),
        )

    def _render_pending_import_splash(self) -> None:
        pending = self._viewer_session.config.pending_import
        pending_payload = (
            {
                "model_descriptor": pending.model_descriptor,
                "textures_dir": pending.textures_dir,
            }
            if pending is not None
            else None
        )
        self._ensure_import_controller().render_pending_import_splash(
            pending_payload,
            self.import_progress_panel,
            _viewer_ui_surface_size(self.wnd),
            opening_session=self._ensure_map_opening_progress_session(),
        )

    def _ensure_map_opening_progress_session(self) -> MapOpeningProgressSession:
        """Return the GUI-only presentation state for the active map open."""
        session = getattr(self, "_map_opening_progress_session", None)
        if session is None:
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                return workflows.map_opening
            session = MapOpeningProgressSession()
            self._map_opening_progress_session = session
        return session

    def _render_map_opening_progress(
        self,
        frame: MapOpeningProgressFrame,
    ) -> None:
        """Render one opening frame without giving lifecycle ownership to the panel."""
        self.import_progress_panel.render(
            _viewer_ui_surface_size(self.wnd),
            frame.map_name,
            frame.stage,
            frame.fraction,
            title=frame.title,
            note=frame.note,
            progress_session_id=frame.session_id,
        )

    def _abandon_map_opening_progress(self) -> None:
        """End the presentation session after cancellation, failure, or pause."""
        self._ensure_map_opening_progress_session().abandon()

    def _render_startup_map_load_splash(self) -> None:
        pending = getattr(self, "_startup_map_load_pending", None)
        manifest = pending[2] if pending is not None else {}
        map_name = os.path.basename(str(manifest.get("source_obj", "map")))
        self.ctx.clear(0.02, 0.02, 0.03)
        frame = self._ensure_map_opening_progress_session().begin_cached(map_name)
        self._render_map_opening_progress(frame)

    def _load_startup_map_after_splash(self) -> None:
        pending = getattr(self, "_startup_map_load_pending", None)
        if pending is None:
            return
        if not getattr(self, "_startup_map_load_splash_rendered", False):
            self._render_startup_map_load_splash()
            self._startup_map_load_splash_rendered = True
            return

        self._startup_map_load_pending = None
        cache_dir, textures_dir, manifest, map_root = pending
        self._load_map(
            cache_dir,
            textures_dir,
            manifest,
            map_root=map_root,
        )
        self._has_map_loaded = True

    def _present_pending_import_splash_now(self) -> bool:
        """Best-effort immediate splash presentation during window setup."""
        try:
            self._render_pending_import_splash()
        except Exception as exc:
            _LOG.debug("Could not render early import splash: %s", exc)
            return False

        for target in (
            getattr(self, "wnd", None),
            getattr(getattr(self, "wnd", None), "_window", None),
        ):
            if target is None:
                continue
            for method_name in ("swap_buffers", "flip", "swap"):
                swap = getattr(target, method_name, None)
                if not callable(swap):
                    continue
                try:
                    swap()
                    return True
                except Exception as exc:
                    _LOG.debug(
                        "Could not present early import splash with %s.%s: %s",
                        type(target).__name__,
                        method_name,
                        exc,
                    )
        return False

    def _start_import_async(
        self,
        model_descriptor: dict,
        textures_dir: str,
        map_name: str,
        is_startup: bool = False,
    ) -> None:
        controller = self._ensure_import_controller()
        if not controller.active:
            self._ensure_map_opening_progress_session().begin_import(
                map_name,
                new_operation=not is_startup,
            )
        controller.start_async(
            model_descriptor,
            textures_dir,
            map_name,
            is_startup=is_startup,
        )

    def _drain_import_queue(self) -> None:
        self._ensure_import_controller().drain_queue()

    def _run_pending_import(self) -> None:
        """
        Runs the FIRST-TIME import for the map the program was launched
        with, when the viewer session carries a pending import instead
        of an already-built cache (see run_viewer_with_pending_import()
        at the bottom of this file, and main()'s use of it in
        caveviewer.app). Called once, from on_render()'s first frame --
        see the _has_map_loaded branch there for why it's deferred to
        that point rather than running before the window even opens.

        Format-agnostic: works the same regardless of whether the
        pending import is an .obj or .glb (see
        caveviewer.app's find_model_file()/import_and_cache_any(), which
        this delegates the actual format-specific parsing to) -- this
        method only deals with the progress-panel/window-lifecycle side
        of things, not anything about the source format itself.

        Shares the exact same import-with-progress-panel approach as
        _handle_open_button_click() (the OPEN button's mid-session
        equivalent of this), just sourced from the pending-import details
        already resolved by main() rather than a fresh folder-browse
        dialog + find_model_file() call.

        Unlike the OPEN button's failure handling (which can safely leave
        a previously-loaded map running untouched), a failure HERE means
        there was never a map to fall back to at all -- so this prints a
        clear error and closes the window instead, rather than leaving
        the person staring at a permanently blank screen with no map and
        no way to get one without restarting the program anyway.
        """
        pending = self._viewer_session.config.pending_import
        if pending is None:
            raise RuntimeError("The viewer session has no pending import")
        model_descriptor = dict(pending.model_descriptor)
        textures_dir = pending.textures_dir
        source_path = model_descriptor.get("obj_path") or model_descriptor.get("glb_path")
        map_name = os.path.basename(source_path)
        self._start_import_async(model_descriptor, textures_dir, map_name, is_startup=True)

    # -- chunk GPU lifecycle ------------------------------------------------

    @staticmethod
    def _new_streaming_frame_timing() -> dict:
        return render_upload.new_streaming_frame_timing()

    @staticmethod
    def _format_optional_ms(value: float | None) -> str:
        return render_upload.format_optional_ms(value)

    @staticmethod
    def _format_streaming_frame_timing(timing: dict) -> str:
        return render_upload.format_streaming_frame_timing(timing)

    def _ensure_chunk_upload_manager(self) -> ChunkUploadManager:
        """Return the render-thread chunk upload owner for the active window."""
        gpu_objects = getattr(self, "_chunk_gpu_objects", None)
        if gpu_objects is None:
            gpu_objects = {}
            self._chunk_gpu_objects = gpu_objects
        upload_states = getattr(self, "_chunk_upload_states", None)
        if upload_states is None:
            upload_states = {}
            self._chunk_upload_states = upload_states
        normal_cache = getattr(self, "_chunk_normal_cache", None)
        if normal_cache is None:
            normal_cache = {}
            self._chunk_normal_cache = normal_cache
        aabbs = getattr(self, "_chunk_aabbs", None)
        if aabbs is None:
            aabbs = {}
            self._chunk_aabbs = aabbs

        ctx = getattr(self, "ctx", None)
        program = getattr(self, "program", None)
        texture_manager = getattr(self, "texture_manager", None)

        def smooth_shading_enabled() -> bool:
            render_mode_buttons = getattr(self, "render_mode_buttons", None)
            return bool(getattr(render_mode_buttons, "smooth_shading_enabled", False))

        manager = getattr(self, "_chunk_upload_manager", None)
        current_time_budget = getattr(
            self,
            "_current_upload_time_budget_ms",
            getattr(self, "_upload_time_budget_ms", 3.0),
        )
        current_operations = getattr(
            self,
            "_current_upload_operations_per_chunk",
            getattr(self, "_upload_groups_per_frame", 1),
        )
        if (
            manager is None
            or manager.ctx is not ctx
            or manager.program is not program
            or manager.texture_manager is not texture_manager
            or manager.gpu_objects is not gpu_objects
            or manager.upload_states is not upload_states
            or manager.normal_cache is not normal_cache
            or manager.aabbs is not aabbs
        ):
            manager = ChunkUploadManager(
                ctx=ctx,
                program=program,
                texture_manager=texture_manager,
                smooth_shading_enabled=smooth_shading_enabled,
                gpu_objects=gpu_objects,
                upload_states=upload_states,
                normal_cache=normal_cache,
                aabbs=aabbs,
                upload_operations_per_chunk=current_operations,
                upload_time_budget_ms=current_time_budget,
                vbo_upload_slice_bytes=getattr(
                    self,
                    "_vbo_upload_slice_bytes",
                    _RENDER_UPLOAD_INITIAL_SLICE_BYTES,
                ),
                texture_upload_slice_bytes=getattr(
                    self,
                    "_texture_upload_slice_bytes",
                    _RENDER_UPLOAD_INITIAL_SLICE_BYTES,
                ),
            )
            self._chunk_upload_manager = manager
        else:
            manager.upload_operations_per_chunk = max(1, int(current_operations))
            manager.upload_time_budget_ms = max(0.5, float(current_time_budget))
        return manager

    def _sync_chunk_upload_state_from_manager(
        self,
        manager: ChunkUploadManager | None = None,
    ) -> None:
        """Mirror manager-owned upload state for legacy private callers/tests."""
        manager = self._ensure_chunk_upload_manager() if manager is None else manager
        self._vbo_upload_slice_bytes = manager.vbo_upload_slice_bytes
        self._texture_upload_slice_bytes = manager.texture_upload_slice_bytes
        self._chunk_gpu_objects = manager.gpu_objects
        self._chunk_upload_states = manager.upload_states
        self._chunk_normal_cache = manager.normal_cache
        self._chunk_aabbs = manager.aabbs

    def _ensure_view_culling_cache(self) -> view_culling.FrustumCullingCache:
        cache = getattr(self, "_view_culling_cache", None)
        if cache is None:
            cache = view_culling.FrustumCullingCache()
            self._view_culling_cache = cache
        if not hasattr(self, "_chunk_visibility_generation"):
            self._chunk_visibility_generation = 0
        return cache

    def _invalidate_visible_chunk_cache(self) -> None:
        self._chunk_visibility_generation = int(
            getattr(self, "_chunk_visibility_generation", 0)
        ) + 1
        cache = getattr(self, "_view_culling_cache", None)
        if cache is not None:
            cache.invalidate()

    @staticmethod
    def _resident_chunk_signature(
        manager: ChunkUploadManager,
        cell,
    ) -> tuple[int, int, bool]:
        vao_list = manager.gpu_objects.get(cell)
        aabb = manager.aabbs.get(cell)
        return (
            id(vao_list) if vao_list is not None else 0,
            len(vao_list) if vao_list is not None else 0,
            aabb is not None,
        )

    def _visible_chunk_gpu_objects(
        self,
        view: np.ndarray,
        projection: np.ndarray,
    ) -> list[tuple[tuple, list]]:
        cache = self._ensure_view_culling_cache()
        return cache.visible_chunks(
            view=view,
            projection=projection,
            chunk_gpu_objects=self._chunk_gpu_objects,
            chunk_aabbs=self._chunk_aabbs,
            generation=self._chunk_visibility_generation,
        )

    def _render_upload_slice_vertices(self) -> int:
        return render_upload.render_upload_slice_vertices(
            getattr(
                self,
                "_vbo_upload_slice_bytes",
                _RENDER_UPLOAD_INITIAL_SLICE_BYTES,
            )
        )

    @staticmethod
    def _min_vbo_upload_slice_bytes() -> int:
        return render_upload.min_vbo_upload_slice_bytes()

    def _record_upload_slice_sizes(self, timing: dict | None) -> None:
        manager = self._ensure_chunk_upload_manager()
        manager.record_upload_slice_sizes(timing)
        self._sync_chunk_upload_state_from_manager(manager)

    def _adapt_upload_slice_size(
        self,
        *,
        kind: str,
        elapsed_ms: float,
        byte_count: int,
        timing: dict | None,
    ) -> None:
        """Delegate adaptive upload-slice policy to the chunk upload manager."""
        manager = self._ensure_chunk_upload_manager()
        manager.adapt_upload_slice_size(
            kind=kind,
            elapsed_ms=elapsed_ms,
            byte_count=byte_count,
            timing=timing,
        )
        self._sync_chunk_upload_state_from_manager(manager)

    def _on_chunk_ready(self, chunk_data):
        manager = self._ensure_chunk_upload_manager()
        before_signature = self._resident_chunk_signature(manager, chunk_data.cell)
        manager.set_frame_limits(
            operations_per_chunk=getattr(
                self,
                "_current_upload_operations_per_chunk",
                getattr(self, "_upload_groups_per_frame", 1),
            ),
            time_budget_ms=getattr(
                self,
                "_current_upload_time_budget_ms",
                getattr(self, "_upload_time_budget_ms", 3.0),
            ),
            timing=getattr(self, "_streaming_frame_timing", None),
        )
        try:
            return manager.on_chunk_ready(chunk_data)
        finally:
            after_signature = self._resident_chunk_signature(manager, chunk_data.cell)
            if after_signature != before_signature:
                self._invalidate_visible_chunk_cache()
            self._sync_chunk_upload_state_from_manager(manager)

    def _on_chunk_unload(self, cell):
        manager = self._ensure_chunk_upload_manager()
        before_signature = self._resident_chunk_signature(manager, cell)
        manager.set_frame_limits(
            operations_per_chunk=getattr(
                self,
                "_current_upload_operations_per_chunk",
                getattr(self, "_upload_groups_per_frame", 1),
            ),
            time_budget_ms=getattr(
                self,
                "_current_upload_time_budget_ms",
                getattr(self, "_upload_time_budget_ms", 3.0),
            ),
            timing=getattr(self, "_streaming_frame_timing", None),
        )
        try:
            manager.on_chunk_unload(cell)
        finally:
            if before_signature != self._resident_chunk_signature(manager, cell):
                self._invalidate_visible_chunk_cache()
            self._sync_chunk_upload_state_from_manager(manager)

    def _apply_shading_toggle_to_cell(self, cell) -> None:
        manager = self._ensure_chunk_upload_manager()
        manager.apply_shading_toggle_to_cell(cell)
        self._sync_chunk_upload_state_from_manager(manager)

    def _apply_shading_toggle(self) -> None:
        """Rewrite loaded VBO normals through the chunk upload manager."""
        manager = self._ensure_chunk_upload_manager()
        manager.apply_shading_toggle(world=getattr(self, "world", None))
        self._sync_chunk_upload_state_from_manager(manager)

    def _buttons_locked_for_loading(self) -> bool:
        """True while map loading should disable the right-side button block."""
        if not self._has_map_loaded:
            return True
        if not self._initial_chunks_loaded:
            return True
        # Once the initial chunks are resident, release the loading-time render
        # mode lock while the startup help screen is still covering the view.
        # That lets Texture turn back on and gives the renderer real textured
        # frames to settle before the user dismisses the overlay.
        return False

    def _sync_render_mode_loading_policy(self) -> None:
        """Apply loading-time button policy and post-load defaults exactly on transitions."""
        locked = self._buttons_locked_for_loading()

        if locked:
            if self._render_mode_load_lock_active:
                return
            self.render_mode_buttons.texture_enabled = False
            self.render_mode_buttons.wireframe_enabled = False
            if self.render_mode_buttons.smooth_shading_enabled:
                self.render_mode_buttons.smooth_shading_enabled = False
                if self._has_map_loaded:
                    self._apply_shading_toggle()
            self._render_mode_load_lock_active = True
            return

        # Just unlocked after loading: enable only Texture.
        if self._render_mode_load_lock_active:
            self.render_mode_buttons.texture_enabled = True
            self.render_mode_buttons.wireframe_enabled = False
            if self.render_mode_buttons.smooth_shading_enabled:
                self.render_mode_buttons.smooth_shading_enabled = False
                if self._has_map_loaded:
                    self._apply_shading_toggle()
            self._render_mode_load_lock_active = False

    # -- moderngl_window hooks ------------------------------------------------
    #
    # moderngl-window renamed its per-frame/event hooks across major versions
    # (older releases used bare names like render()/key_event(), 3.x renamed
    # them to on_render()/on_key_event() etc). To work across versions without
    # guessing which exact release someone has installed, each hook below is
    # implemented under the new on_* name and aliased to the old bare name.

    # Right-side column layout: brightness stepper, then render-distance
    # stepper, then the Mesh/Texture/Help/Color/Open button block, all
    # stacked vertically and anchored as ONE group to the bottom-right
    # corner of the window (moved here from separate top-anchored
    # positions per request). Computed in this single method, used
    # identically by render() and the mouse-press handler, so the
    # clickable areas can never drift out of sync with what's actually
    # drawn -- the same reasoning the old per-control anchor helpers
    # already followed, just now covering the whole column at once since
    # a bottom anchor means every piece's position depends on the total
    # height of everything below the WINDOW bottom margin, not just its
    # own height.
    RIGHT_COLUMN_BOTTOM_MARGIN = 18
    RIGHT_COLUMN_GAP = 10  # vertical gap between the right-side HUD blocks
    RIGHT_COLUMN_BUTTON_GROUP_GAP = 20  # extra gap before the Mesh/Texture/Shade group

    # Keyboard look fallback (especially useful on macOS hardware where
    # right-button drag can be awkward/unavailable). Interpreted as
    # virtual mouse pixels per second and passed through camera.look().
    _KEY_LOOK_PIXELS_PER_SECOND = 700.0

    # Maps the GLOBAL LIGHT stepper's 0-10 integer range onto the
    # shader's actual u_ambient float. 0 -> _AMBIENT_MIN reproduces the
    # exact fixed ambient value this app always used before this feature
    # existed (a tiny fill so unlit areas aren't pure black, not truly
    # zero) -- so the default stepper value of 0 changes nothing for
    # anyone who never touches this control. 10 -> _AMBIENT_MAX is a
    # strong, even fill bright enough to read the whole cave clearly
    # without the headlamp doing any of the work, without fully blowing
    # out texture detail into flat white.
    _AMBIENT_MIN = 0.04
    _AMBIENT_MAX = 0.9
    _INITIAL_LOAD_MIN_CHUNKS = 6
    _INITIAL_VISUAL_READY_SETTLE_FRAMES = 3
    _STARTUP_VISUAL_RADIUS_EXTRA_CHUNKS = 3
    _STARTUP_VISUAL_RADIUS_MAX_CHUNKS = 10
    _CHUNK_PREP_MAX_FRACTION = 0.97
    _CHUNK_PREP_COMPLETE_HOLD_SECONDS = 0.85
    _STREAMING_FAILURES_PER_FRAME = 8

    def _reset_initial_chunk_loading_state(self) -> None:
        """Reset ordinary map-load readiness before streaming a new map."""
        self._initial_chunks_loaded = False
        self._initial_visual_ready = False
        self._initial_visual_ready_frames = 0
        self._initial_visual_ready_visible_chunks = 0
        self._initial_visual_ready_required_textures = 0
        self._initial_visual_ready_resident_textures = 0
        self._initial_visual_ready_visible_textures = 0
        self._initial_visual_ready_missing_textures = 0
        self._initial_visual_ready_expected_chunks = 0
        self._initial_visual_ready_covered_chunks = 0
        self._initial_visual_ready_missing_chunks = 0
        self._initial_visual_ready_coverage_pct = 100.0
        self._initial_visual_ready_logged = False
        self._chunk_prep_progress = 0.0
        self._chunk_prep_complete_until = None
        self._chunk_prep_completion_armed = False
        manifest = getattr(self, "manifest", {})
        source_obj = (
            manifest.get("source_obj", "map")
            if isinstance(manifest, Mapping)
            else "map"
        )
        map_name = os.path.basename(str(source_obj or "map"))
        self._ensure_map_opening_progress_session().begin_streaming(map_name)

    def _startup_visual_prefetch_is_active(self) -> bool:
        overlay = getattr(self, "controls_overlay", None)
        return (
            overlay is not None
            and bool(getattr(overlay, "is_waiting_for_begin", False))
            and not getattr(self, "_initial_visual_ready", False)
        )

    def _target_streaming_load_radius(self) -> int:
        base_radius = max(1, int(self.render_distance_stepper.value))
        if not self._startup_visual_prefetch_is_active():
            return base_radius
        max_radius = max(
            base_radius,
            min(
                int(
                    getattr(
                        self.render_distance_stepper,
                        "max_value",
                        self._STARTUP_VISUAL_RADIUS_MAX_CHUNKS,
                    )
                ),
                self._STARTUP_VISUAL_RADIUS_MAX_CHUNKS,
            ),
        )
        return min(
            max_radius,
            base_radius + self._STARTUP_VISUAL_RADIUS_EXTRA_CHUNKS,
        )


    def _streaming_cell_priority_key(
        self,
    ) -> Callable[[tuple[int, int, int]], tuple[int, int, float, float, float]]:
        """Rank streaming cells by current camera view, then by distance.

        Render distance answers "how much cave should be eligible to load";
        this priority answers "which eligible cells should consume the next
        limited worker/upload slots."  A distance-only priority can spend that
        budget on nearby side/behind cells while the screen-facing corridor is
        still empty, which makes high distance values look ineffective.
        """
        world = getattr(self, "world", None)
        world_config = getattr(world, "config", None)
        chunk_size = max(1e-6, float(getattr(world_config, "chunk_size", 1.0)))
        position = np.asarray(self.camera.position, dtype=np.float64)
        forward = np.asarray(self.camera.forward(), dtype=np.float64)
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-9:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            forward = forward / forward_norm

        wnd = getattr(self, "wnd", None)
        window_size = getattr(wnd, "size", _DEFAULT_WINDOW_SIZE)
        width, height = window_size
        aspect = max(1.0, float(width) / max(1.0, float(height)))
        fov_deg = float(getattr(self.camera, "fov_deg", 75.0))
        half_fov_rad = math.radians(max(1.0, min(179.0, fov_deg)) * 0.5)
        visible_cone_tan = math.tan(half_fov_rad) * aspect * 1.25
        chunk_size_sq = chunk_size * chunk_size

        camera_x = float(position[0])
        camera_y = float(position[1])
        camera_z = float(position[2])
        forward_x = float(forward[0])
        forward_y = float(forward[1])
        forward_z = float(forward[2])

        def priority(cell: tuple[int, int, int]) -> tuple[int, int, float, float, float]:
            center_x = (cell[0] + 0.5) * chunk_size
            center_y = (cell[1] + 0.5) * chunk_size
            center_z = (cell[2] + 0.5) * chunk_size
            rel_x = center_x - camera_x
            rel_y = center_y - camera_y
            rel_z = center_z - camera_z
            depth = rel_x * forward_x + rel_y * forward_y + rel_z * forward_z
            distance_sq = rel_x * rel_x + rel_y * rel_y + rel_z * rel_z
            lateral_sq = max(0.0, distance_sq - depth * depth)
            front_penalty = 0 if depth >= -chunk_size else 1
            cone_depth = max(chunk_size, depth)
            visible_radius = cone_depth * visible_cone_tan + chunk_size
            visible_penalty = (
                0
                if front_penalty == 0 and lateral_sq <= visible_radius * visible_radius
                else 1
            )
            angular_sq = lateral_sq / max(chunk_size_sq, depth * depth)
            depth_cells = max(0.0, depth / chunk_size)
            distance_cells_sq = distance_sq / chunk_size_sq
            return (
                front_penalty,
                visible_penalty,
                depth_cells,
                angular_sq,
                distance_cells_sq,
            )

        return priority

    def _startup_upload_boost_is_active(self) -> bool:
        overlay = getattr(self, "controls_overlay", None)
        return (
            overlay is not None
            and overlay.is_waiting_for_begin
            and not getattr(self, "_initial_chunks_loaded", False)
        )

    def _streaming_upload_limits(self, stats: dict | None = None) -> tuple[int, int, float]:
        """Return chunk/operation/time upload limits for the current frame."""
        if self._startup_upload_boost_is_active():
            return (
                max(self._upload_chunks_per_frame, _STARTUP_UPLOAD_CHUNKS_PER_FRAME),
                max(
                    self._upload_groups_per_frame,
                    _STARTUP_UPLOAD_OPERATIONS_PER_CHUNK,
                ),
                max(self._upload_time_budget_ms, _STARTUP_UPLOAD_TIME_BUDGET_MS),
            )
        if stats is not None:
            ready = max(0, int(stats.get("ready", 0)))
            wanted = max(0, int(stats.get("wanted", 0)))
            loaded_wanted = max(
                0,
                int(stats.get("loaded_wanted", stats.get("loaded", 0))),
            )
            failed_wanted = max(0, int(stats.get("failed_wanted", 0)))
            missing_wanted = max(0, wanted - loaded_wanted - failed_wanted)
            if ready > 0 and missing_wanted > 0:
                return (
                    max(
                        self._upload_chunks_per_frame,
                        _CATCHUP_UPLOAD_CHUNKS_PER_FRAME,
                    ),
                    max(
                        self._upload_groups_per_frame,
                        _CATCHUP_UPLOAD_OPERATIONS_PER_CHUNK,
                    ),
                    max(
                        self._upload_time_budget_ms,
                        _CATCHUP_UPLOAD_TIME_BUDGET_MS,
                    ),
                )
        return (
            self._upload_chunks_per_frame,
            self._upload_groups_per_frame,
            self._upload_time_budget_ms,
        )

    @staticmethod
    def _initial_chunk_load_needed(
        stats: dict,
        max_loaded_chunks: int,
    ) -> int:
        total_available = max(1, int(stats.get("total_available", 1)))
        wanted = max(1, int(stats.get("wanted", CaveViewerWindow._INITIAL_LOAD_MIN_CHUNKS)))
        # Startup streams the same radius the viewer will reveal. Require that
        # current wanted set to settle before revealing the begin prompt;
        # otherwise the first visible frame can have missing chunk rectangles
        # beyond the old startup-only radius.
        return min(total_available, max(1, int(max_loaded_chunks)), wanted)

    def _initial_chunk_load_is_ready(self, stats: dict) -> bool:
        loaded = max(0, int(stats.get("loaded_wanted", stats.get("loaded", 0))))
        failed_wanted = max(0, int(stats.get("failed_wanted", 0)))
        max_loaded = max(1, int(getattr(self.world.config, "max_loaded_chunks", self._INITIAL_LOAD_MIN_CHUNKS)))
        needed = self._initial_chunk_load_needed(stats, max_loaded)
        return loaded + failed_wanted >= needed

    def _initial_visual_readiness_is_settled(
        self,
        stats: dict,
        texture_readiness: Mapping[str, object] | None = None,
        visual_coverage: Mapping[str, object] | None = None,
        route_prefetch: Mapping[str, object] | None = None,
    ) -> bool:
        if not getattr(self, "_initial_chunks_loaded", False):
            return False
        if not self._initial_chunk_load_is_ready(stats):
            return False
        if max(0, int(stats.get("pending", 0))) > 0:
            return False
        if max(0, int(stats.get("ready", 0))) > 0:
            return False
        upload_states = getattr(self, "_chunk_upload_states", {})
        if len(upload_states) > 0:
            return False
        if texture_readiness is not None and not bool(
            texture_readiness.get("textures_ready", True)
        ):
            return False
        if visual_coverage is not None and not bool(
            visual_coverage.get("coverage_ready", True)
        ):
            return False
        if route_prefetch is not None and not bool(
            route_prefetch.get("ready", True)
        ):
            return False
        return True

    @staticmethod
    def _texture_source_key(source: object) -> object:
        try:
            hash(source)
        except TypeError:
            return id(source)
        return source

    def _current_wanted_cells_snapshot(self) -> frozenset[tuple[int, int, int]]:
        world = getattr(self, "world", None)
        snapshot = getattr(world, "wanted_cells_snapshot", None)
        if callable(snapshot):
            return frozenset(snapshot())
        return frozenset(getattr(world, "_last_wanted_cells", ()))

    def _texture_sources_for_cells(
        self,
        cells: Iterable[tuple[int, int, int]],
    ) -> set[object]:
        texture_manager = getattr(self, "texture_manager", None)
        material_to_file = getattr(texture_manager, "material_to_file", {})
        if not isinstance(material_to_file, Mapping):
            return set()
        manifest = getattr(self, "manifest", {})
        chunks = manifest.get("chunks", {}) if isinstance(manifest, Mapping) else {}
        if not isinstance(chunks, Mapping):
            return set()

        sources: set[object] = set()
        for cell in cells:
            chunk_info = chunks.get(f"{cell[0]}_{cell[1]}_{cell[2]}")
            if not isinstance(chunk_info, Mapping):
                continue
            materials = chunk_info.get("materials", ())
            if not isinstance(materials, Iterable) or isinstance(materials, str):
                materials = ()
            for material in materials:
                source = material_to_file.get(str(material))
                if source:
                    sources.add(self._texture_source_key(source))
        return sources

    def _texture_sources_for_visible_cells(
        self,
        visible_cells: Iterable[tuple[tuple, list]] | None,
    ) -> set[object]:
        texture_manager = getattr(self, "texture_manager", None)
        material_to_file = getattr(texture_manager, "material_to_file", {})
        if visible_cells is None or not isinstance(material_to_file, Mapping):
            return set()
        sources: set[object] = set()
        for _cell, vao_list in visible_cells:
            for _vao, _vbo, material_name, _texture in vao_list:
                source = material_to_file.get(str(material_name))
                if source:
                    sources.add(self._texture_source_key(source))
        return sources

    def _resident_texture_source_keys(self) -> tuple[set[object], bool]:
        texture_manager = getattr(self, "texture_manager", None)
        resident_sources = None
        known_exact = False
        if texture_manager is not None:
            exact_sources = getattr(texture_manager, "resident_texture_sources", None)
            if callable(exact_sources):
                resident_sources = exact_sources()
                known_exact = True
        if resident_sources is None:
            return set(), known_exact
        return {
            self._texture_source_key(source)
            for source in resident_sources
            if source
        }, known_exact

    def _benchmark_route_prefetch_stats(self) -> dict[str, object]:
        prefetch_cells = set(getattr(self, "_benchmark_route_prefetch_cells", ()))
        if not prefetch_cells:
            return {
                "active": False,
                "ready": True,
                "expected_cells": 0,
                "loaded_cells": 0,
                "pending_cells": 0,
                "failed_cells": 0,
                "missing_cells": 0,
                "coverage_pct": 100.0,
            }

        world = getattr(self, "world", None)
        if world is None:
            return {
                "active": True,
                "ready": False,
                "expected_cells": len(prefetch_cells),
                "loaded_cells": 0,
                "pending_cells": 0,
                "failed_cells": 0,
                "missing_cells": len(prefetch_cells),
                "coverage_pct": 0.0,
            }

        lock = getattr(world, "_lock", None)
        if lock is None:
            loaded_cells = set(getattr(world, "loaded_cells", set()))
            pending_cells = set(getattr(world, "_pending", set()))
            failed_cells = set(getattr(world, "_failed_cells", {}))
        else:
            with lock:
                loaded_cells = set(getattr(world, "loaded_cells", set()))
                pending_cells = set(getattr(world, "_pending", set()))
                failed_cells = set(getattr(world, "_failed_cells", {}))

        loaded_prefetch = prefetch_cells & loaded_cells
        pending_prefetch = prefetch_cells & pending_cells
        failed_prefetch = prefetch_cells & failed_cells
        covered_count = len(loaded_prefetch | failed_prefetch)
        missing_count = max(0, len(prefetch_cells) - covered_count)
        coverage_pct = 100.0 * covered_count / max(1, len(prefetch_cells))
        return {
            "active": True,
            "ready": missing_count == 0,
            "expected_cells": len(prefetch_cells),
            "loaded_cells": len(loaded_prefetch),
            "pending_cells": len(pending_prefetch),
            "failed_cells": len(failed_prefetch),
            "missing_cells": missing_count,
            "coverage_pct": coverage_pct,
        }

    def _initial_texture_readiness_stats(
        self,
        visible_cells: Iterable[tuple[tuple, list]] | None,
    ) -> dict[str, object]:
        texture_manager = getattr(self, "texture_manager", None)
        manager_stats = (
            texture_manager.stats()
            if texture_manager is not None and hasattr(texture_manager, "stats")
            else {}
        )
        wanted_sources = self._texture_sources_for_cells(
            self._current_wanted_cells_snapshot()
        )
        visible_sources = self._texture_sources_for_visible_cells(visible_cells)
        required_sources = wanted_sources if wanted_sources else visible_sources
        required_textures = len(required_sources)
        resident_textures = max(
            0,
            int(manager_stats.get("unique_files_resident", required_textures)),
        )
        resident_sources, exact_sources_known = self._resident_texture_source_keys()
        missing_sources = (
            required_sources - resident_sources
            if exact_sources_known and required_sources
            else set()
        )
        textures_ready = (
            not missing_sources
            if exact_sources_known and required_sources
            else required_textures <= 0 or resident_textures >= required_textures
        )
        return {
            "textures_ready": textures_ready,
            "required_textures": required_textures,
            "resident_textures": resident_textures,
            "missing_textures": len(missing_sources),
            "visible_textures": len(visible_sources),
            "resident_texture_bytes": int(
                manager_stats.get("resident_texture_bytes", 0)
            ),
            "resident_texture_budget_bytes": int(
                manager_stats.get("resident_texture_budget_bytes", 0)
            ),
        }

    def _manifest_chunk_bounds(
        self,
        cell: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        manifest = getattr(self, "manifest", {})
        chunks = manifest.get("chunks", {}) if isinstance(manifest, Mapping) else {}
        chunk_info = chunks.get(f"{cell[0]}_{cell[1]}_{cell[2]}")
        if not isinstance(chunk_info, Mapping):
            return None
        try:
            bounds_min = np.asarray(chunk_info["bounds_min"], dtype=np.float64)
            bounds_max = np.asarray(chunk_info["bounds_max"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None
        if bounds_min.shape != (3,) or bounds_max.shape != (3,):
            return None
        return bounds_min, bounds_max

    def _failed_cells_snapshot(self) -> frozenset[tuple[int, int, int]]:
        world = getattr(self, "world", None)
        failed_cells = getattr(world, "_failed_cells", {})
        if isinstance(failed_cells, Mapping):
            return frozenset(failed_cells.keys())
        return frozenset(failed_cells or ())

    def _startup_visual_coverage_stats(
        self,
        visible_cells: Iterable[tuple[tuple, list]] | None,
        view: np.ndarray | None,
        projection: np.ndarray | None,
    ) -> dict[str, object]:
        if view is None or projection is None:
            return {
                "coverage_ready": True,
                "expected_chunks": 0,
                "covered_chunks": 0,
                "missing_chunks": 0,
                "coverage_pct": 100.0,
            }

        wanted_cells = self._current_wanted_cells_snapshot()
        if not wanted_cells:
            return {
                "coverage_ready": True,
                "expected_chunks": 0,
                "covered_chunks": 0,
                "missing_chunks": 0,
                "coverage_pct": 100.0,
            }

        planes = view_culling.frustum_planes(
            np.asarray(view, dtype=np.float64),
            np.asarray(projection, dtype=np.float64),
        )
        expected_cells = set()
        for cell in wanted_cells:
            bounds = self._manifest_chunk_bounds(cell)
            if bounds is None:
                continue
            if view_culling.aabb_inside_frustum(planes, bounds[0], bounds[1]):
                expected_cells.add(cell)

        visible_loaded_cells = {
            tuple(cell)
            for cell, _vao_list in (visible_cells or ())
        }
        terminal_cells = visible_loaded_cells | self._failed_cells_snapshot()
        covered_chunks = len(expected_cells & terminal_cells)
        missing_chunks = max(0, len(expected_cells) - covered_chunks)
        coverage_pct = (
            100.0
            if not expected_cells
            else 100.0 * covered_chunks / len(expected_cells)
        )
        return {
            "coverage_ready": missing_chunks == 0,
            "expected_chunks": len(expected_cells),
            "covered_chunks": covered_chunks,
            "missing_chunks": missing_chunks,
            "coverage_pct": coverage_pct,
        }

    def _initial_visual_readiness_stats(
        self,
        stats: dict,
        visible_chunk_count: int,
        visible_cells: Iterable[tuple[tuple, list]] | None = None,
        view: np.ndarray | None = None,
        projection: np.ndarray | None = None,
    ) -> dict:
        """Update and return startup stats augmented with visual-ready state."""
        texture_readiness = self._initial_texture_readiness_stats(visible_cells)
        visual_coverage = self._startup_visual_coverage_stats(
            visible_cells,
            view,
            projection,
        )
        route_prefetch = self._benchmark_route_prefetch_stats()
        if getattr(self, "_initial_visual_ready", False):
            visual_stats = dict(stats)
            visual_stats["visual_ready"] = True
            visual_stats["visual_ready_frames"] = int(
                getattr(self, "_initial_visual_ready_frames", 0)
            )
            visual_stats["visual_ready_visible_chunks"] = int(
                getattr(self, "_initial_visual_ready_visible_chunks", 0)
            )
            visual_stats["visual_ready_required_textures"] = int(
                getattr(self, "_initial_visual_ready_required_textures", 0)
            )
            visual_stats["visual_ready_resident_textures"] = int(
                getattr(self, "_initial_visual_ready_resident_textures", 0)
            )
            visual_stats["visual_ready_visible_textures"] = int(
                getattr(self, "_initial_visual_ready_visible_textures", 0)
            )
            visual_stats["visual_ready_missing_textures"] = int(
                getattr(self, "_initial_visual_ready_missing_textures", 0)
            )
            visual_stats["visual_ready_expected_chunks"] = int(
                getattr(self, "_initial_visual_ready_expected_chunks", 0)
            )
            visual_stats["visual_ready_covered_chunks"] = int(
                getattr(self, "_initial_visual_ready_covered_chunks", 0)
            )
            visual_stats["visual_ready_missing_chunks"] = int(
                getattr(self, "_initial_visual_ready_missing_chunks", 0)
            )
            visual_stats["visual_ready_coverage_pct"] = float(
                getattr(self, "_initial_visual_ready_coverage_pct", 100.0)
            )
            visual_stats["route_prefetch_expected_cells"] = int(
                getattr(self, "_initial_route_prefetch_expected_cells", 0)
            )
            visual_stats["route_prefetch_loaded_cells"] = int(
                getattr(self, "_initial_route_prefetch_loaded_cells", 0)
            )
            visual_stats["route_prefetch_pending_cells"] = int(
                getattr(self, "_initial_route_prefetch_pending_cells", 0)
            )
            visual_stats["route_prefetch_failed_cells"] = int(
                getattr(self, "_initial_route_prefetch_failed_cells", 0)
            )
            visual_stats["route_prefetch_missing_cells"] = int(
                getattr(self, "_initial_route_prefetch_missing_cells", 0)
            )
            visual_stats["route_prefetch_coverage_pct"] = float(
                getattr(self, "_initial_route_prefetch_coverage_pct", 100.0)
            )
            return visual_stats

        if self._initial_visual_readiness_is_settled(
            stats,
            texture_readiness,
            visual_coverage,
            route_prefetch,
        ):
            self._initial_visual_ready_frames = (
                int(getattr(self, "_initial_visual_ready_frames", 0)) + 1
            )
            self._initial_visual_ready_visible_chunks = int(visible_chunk_count)
            self._initial_visual_ready_required_textures = int(
                texture_readiness["required_textures"]
            )
            self._initial_visual_ready_resident_textures = int(
                texture_readiness["resident_textures"]
            )
            self._initial_visual_ready_visible_textures = int(
                texture_readiness["visible_textures"]
            )
            self._initial_visual_ready_missing_textures = int(
                texture_readiness["missing_textures"]
            )
            self._initial_visual_ready_expected_chunks = int(
                visual_coverage["expected_chunks"]
            )
            self._initial_visual_ready_covered_chunks = int(
                visual_coverage["covered_chunks"]
            )
            self._initial_visual_ready_missing_chunks = int(
                visual_coverage["missing_chunks"]
            )
            self._initial_visual_ready_coverage_pct = float(
                visual_coverage["coverage_pct"]
            )
            self._record_initial_route_prefetch_stats(route_prefetch)
            if (
                self._initial_visual_ready_frames
                >= self._INITIAL_VISUAL_READY_SETTLE_FRAMES
            ):
                self._initial_visual_ready = True
                self._log_initial_visual_ready_complete(
                    stats,
                    visible_chunk_count=visible_chunk_count,
                )
        else:
            self._initial_visual_ready_frames = 0
            self._initial_visual_ready_visible_chunks = 0
            self._initial_visual_ready_required_textures = 0
            self._initial_visual_ready_resident_textures = int(
                texture_readiness["resident_textures"]
            )
            self._initial_visual_ready_visible_textures = int(
                texture_readiness["visible_textures"]
            )
            self._initial_visual_ready_missing_textures = int(
                texture_readiness["missing_textures"]
            )
            self._initial_visual_ready_expected_chunks = int(
                visual_coverage["expected_chunks"]
            )
            self._initial_visual_ready_covered_chunks = int(
                visual_coverage["covered_chunks"]
            )
            self._initial_visual_ready_missing_chunks = int(
                visual_coverage["missing_chunks"]
            )
            self._initial_visual_ready_coverage_pct = float(
                visual_coverage["coverage_pct"]
            )
            self._record_initial_route_prefetch_stats(route_prefetch)

        visual_stats = dict(stats)
        visual_stats["visual_ready"] = bool(
            getattr(self, "_initial_visual_ready", False)
        )
        visual_stats["visual_ready_frames"] = int(
            getattr(self, "_initial_visual_ready_frames", 0)
        )
        visual_stats["visual_ready_visible_chunks"] = int(visible_chunk_count)
        visual_stats["visual_ready_required_textures"] = int(
            texture_readiness["required_textures"]
        )
        visual_stats["visual_ready_resident_textures"] = int(
            texture_readiness["resident_textures"]
        )
        visual_stats["visual_ready_visible_textures"] = int(
            texture_readiness["visible_textures"]
        )
        visual_stats["visual_ready_missing_textures"] = int(
            texture_readiness["missing_textures"]
        )
        visual_stats["visual_ready_expected_chunks"] = int(
            visual_coverage["expected_chunks"]
        )
        visual_stats["visual_ready_covered_chunks"] = int(
            visual_coverage["covered_chunks"]
        )
        visual_stats["visual_ready_missing_chunks"] = int(
            visual_coverage["missing_chunks"]
        )
        visual_stats["visual_ready_coverage_pct"] = round(
            float(visual_coverage["coverage_pct"]),
            3,
        )
        visual_stats["route_prefetch_expected_cells"] = int(
            route_prefetch["expected_cells"]
        )
        visual_stats["route_prefetch_loaded_cells"] = int(
            route_prefetch["loaded_cells"]
        )
        visual_stats["route_prefetch_pending_cells"] = int(
            route_prefetch["pending_cells"]
        )
        visual_stats["route_prefetch_failed_cells"] = int(
            route_prefetch["failed_cells"]
        )
        visual_stats["route_prefetch_missing_cells"] = int(
            route_prefetch["missing_cells"]
        )
        visual_stats["route_prefetch_coverage_pct"] = round(
            float(route_prefetch["coverage_pct"]),
            3,
        )
        return visual_stats

    def _record_initial_route_prefetch_stats(
        self,
        route_prefetch: Mapping[str, object],
    ) -> None:
        self._initial_route_prefetch_expected_cells = int(
            route_prefetch["expected_cells"]
        )
        self._initial_route_prefetch_loaded_cells = int(
            route_prefetch["loaded_cells"]
        )
        self._initial_route_prefetch_pending_cells = int(
            route_prefetch["pending_cells"]
        )
        self._initial_route_prefetch_failed_cells = int(
            route_prefetch["failed_cells"]
        )
        self._initial_route_prefetch_missing_cells = int(
            route_prefetch["missing_cells"]
        )
        self._initial_route_prefetch_coverage_pct = float(
            route_prefetch["coverage_pct"]
        )

    def _log_initial_visual_ready_complete(
        self,
        stats: dict,
        *,
        visible_chunk_count: int,
    ) -> None:
        if getattr(self, "_initial_visual_ready_logged", False):
            return
        self._initial_visual_ready_logged = True
        started_at = getattr(self, "_initial_compilation_started_at", None)
        elapsed_s = (
            0.0
            if started_at is None
            else max(0.0, time.perf_counter() - started_at)
        )
        upload_states = getattr(self, "_chunk_upload_states", {})
        required_textures = int(
            getattr(self, "_initial_visual_ready_required_textures", 0)
        )
        resident_textures = int(
            getattr(self, "_initial_visual_ready_resident_textures", 0)
        )
        visible_textures = int(
            getattr(self, "_initial_visual_ready_visible_textures", 0)
        )
        missing_textures = int(
            getattr(self, "_initial_visual_ready_missing_textures", 0)
        )
        expected_chunks = int(
            getattr(self, "_initial_visual_ready_expected_chunks", 0)
        )
        covered_chunks = int(
            getattr(self, "_initial_visual_ready_covered_chunks", 0)
        )
        missing_chunks = int(
            getattr(self, "_initial_visual_ready_missing_chunks", 0)
        )
        coverage_pct = float(
            getattr(self, "_initial_visual_ready_coverage_pct", 100.0)
        )
        route_prefetch_expected = int(
            getattr(self, "_initial_route_prefetch_expected_cells", 0)
        )
        route_prefetch_loaded = int(
            getattr(self, "_initial_route_prefetch_loaded_cells", 0)
        )
        route_prefetch_pending = int(
            getattr(self, "_initial_route_prefetch_pending_cells", 0)
        )
        route_prefetch_failed = int(
            getattr(self, "_initial_route_prefetch_failed_cells", 0)
        )
        route_prefetch_missing = int(
            getattr(self, "_initial_route_prefetch_missing_cells", 0)
        )
        route_prefetch_coverage_pct = float(
            getattr(self, "_initial_route_prefetch_coverage_pct", 100.0)
        )
        world = getattr(self, "world", None)
        startup_radius = int(
            getattr(getattr(world, "config", None), "load_radius_cells", 0) or 0
        )
        _LOG.info(
            "Initial visual readiness completed in %.2fs "
            "(visible=%d loaded=%d pending=%d ready=%d wanted=%d "
            "upload_states=%d textures=%d/%d missing_textures=%d "
            "visible_textures=%d coverage=%d/%d missing_chunks=%d %.1f%% "
            "startup_radius=%d route_prefetch=%d/%d pending=%d failed=%d "
            "missing=%d %.1f%%).",
            elapsed_s,
            int(visible_chunk_count),
            int(stats.get("loaded", 0)),
            int(stats.get("pending", 0)),
            int(stats.get("ready", 0)),
            int(stats.get("wanted", 0)),
            len(upload_states),
            resident_textures,
            required_textures,
            missing_textures,
            visible_textures,
            covered_chunks,
            expected_chunks,
            missing_chunks,
            coverage_pct,
            startup_radius,
            route_prefetch_loaded,
            route_prefetch_expected,
            route_prefetch_pending,
            route_prefetch_failed,
            route_prefetch_missing,
            route_prefetch_coverage_pct,
        )
        benchmark_controller = self._active_benchmark_controller()
        if benchmark_controller is not None:
            benchmark_controller.update_environment(
                {
                    "initial_visual_ready_seconds": round(elapsed_s, 6),
                    "initial_visual_ready_visible_chunks": int(visible_chunk_count),
                    "initial_visual_ready_frames": int(
                        getattr(self, "_initial_visual_ready_frames", 0)
                    ),
                    "initial_visual_ready_required_textures": required_textures,
                    "initial_visual_ready_resident_textures": resident_textures,
                    "initial_visual_ready_visible_textures": visible_textures,
                    "initial_visual_ready_missing_textures": missing_textures,
                    "initial_visual_ready_expected_chunks": expected_chunks,
                    "initial_visual_ready_covered_chunks": covered_chunks,
                    "initial_visual_ready_missing_chunks": missing_chunks,
                    "initial_visual_ready_coverage_pct": round(coverage_pct, 3),
                    "initial_visual_ready_load_radius_chunks": startup_radius,
                    "initial_route_prefetch_expected_cells": route_prefetch_expected,
                    "initial_route_prefetch_loaded_cells": route_prefetch_loaded,
                    "initial_route_prefetch_pending_cells": route_prefetch_pending,
                    "initial_route_prefetch_failed_cells": route_prefetch_failed,
                    "initial_route_prefetch_missing_cells": route_prefetch_missing,
                    "initial_route_prefetch_coverage_pct": round(
                        route_prefetch_coverage_pct,
                        3,
                    ),
                }
            )

    def _log_initial_compilation_complete(self, stats: dict) -> None:
        if getattr(self, "_initial_compilation_logged", False):
            return
        started_at = getattr(self, "_initial_compilation_started_at", None)
        if started_at is None:
            return

        elapsed_s = max(0.0, time.perf_counter() - started_at)
        self._initial_compilation_logged = True
        _LOG.info(
            "Initial map compilation completed in %.2fs "
            "(loaded=%d pending=%d ready=%d wanted=%d).",
            elapsed_s,
            int(stats.get("loaded", 0)),
            int(stats.get("pending", 0)),
            int(stats.get("ready", 0)),
            int(stats.get("wanted", 0)),
        )

    def _log_main_thread_stall(
        self,
        label: str,
        elapsed_s: float,
        **details: object,
    ) -> None:
        if elapsed_s < _MAIN_THREAD_STALL_LOG_THRESHOLD_S:
            return
        last_logs = getattr(self, "_main_thread_stall_last_log_at", None)
        if last_logs is None:
            last_logs = {}
            self._main_thread_stall_last_log_at = last_logs
        now = time.perf_counter()
        last_logged_at = last_logs.get(label)
        if (
            last_logged_at is not None
            and now - last_logged_at < _MAIN_THREAD_STALL_LOG_MIN_INTERVAL_S
        ):
            return
        last_logs[label] = now
        detail_items = [
            f"{name}={value}"
            for name, value in details.items()
            if value is not None
        ]
        detail_text = f" ({' '.join(detail_items)})" if detail_items else ""
        _LOG.warning(
            "Main-thread stall: %s took %.0fms%s.",
            label,
            elapsed_s * 1000.0,
            detail_text,
        )

    def _initial_chunk_load_progress(self, stats: dict) -> float:
        loaded = max(0, int(stats.get("loaded_wanted", stats.get("loaded", 0))))
        ready = max(0, int(stats.get("ready", 0)))
        pending = max(0, int(stats.get("pending", 0)))
        failed_wanted = max(0, int(stats.get("failed_wanted", 0)))
        max_loaded = max(1, int(getattr(self.world.config, "max_loaded_chunks", self._INITIAL_LOAD_MIN_CHUNKS)))
        needed = self._initial_chunk_load_needed(stats, max_loaded)
        # Give partial credit so the bar moves as soon as background
        # decode starts, not only once GPU uploads complete:
        #   pending  0.25  decode in progress
        #   ready    0.75  decode done, upload queued
        #   loaded   1.00  fully on GPU
        #   failed   1.00  terminally settled; render continues with a hole
        effective = loaded + failed_wanted + 0.75 * ready + 0.25 * min(pending, needed)
        return max(0.0, min(1.0, effective / needed))

    def _drain_streaming_worker_failures(self) -> None:
        world = getattr(self, "world", None)
        if world is None or not hasattr(world, "drain_worker_failures"):
            return
        for failure in world.drain_worker_failures(
            max_items=self._STREAMING_FAILURES_PER_FRAME
        ):
            log = _LOG.error if failure.fatal else _LOG.warning
            log(
                "Streaming worker %s for chunk %s during %s on %s: %s: %s",
                "failed" if failure.fatal else "reported a non-fatal failure",
                failure.cell,
                failure.stage,
                failure.thread_name,
                failure.error_type,
                failure.message,
            )

    @staticmethod
    def _frustum_planes(view: np.ndarray, proj: np.ndarray) -> np.ndarray:
        return view_culling.frustum_planes(view, proj)

    @staticmethod
    def _aabb_inside_frustum(planes: np.ndarray,
                              bmin: np.ndarray, bmax: np.ndarray) -> bool:
        return view_culling.aabb_inside_frustum(planes, bmin, bmax)

    def _right_column_ui_scale(self) -> float:
        return float(getattr(self, "_viewer_ui_scale", 1.0))

    def _right_column_geometry_scale(self) -> float:
        return float(
            getattr(self, "_right_column_panel_scale", self.RIGHT_COLUMN_PANEL_SCALE)
        )

    def _right_column_text_scale(self) -> float:
        return float(
            getattr(
                self,
                "_right_column_panel_text_scale",
                self.RIGHT_COLUMN_PANEL_TEXT_SCALE,
            )
        )

    def _right_column_label_text_scale(self) -> float:
        return float(
            getattr(
                self,
                "_right_column_panel_label_text_scale",
                self.RIGHT_COLUMN_PANEL_LABEL_TEXT_SCALE,
            )
        )

    def _right_column_button_text_scale(self) -> float:
        return float(
            getattr(
                self,
                "_right_column_panel_button_text_scale",
                self.RIGHT_COLUMN_PANEL_BUTTON_TEXT_SCALE,
            )
        )

    def _update_right_column_hud_scale(self, window_size: tuple[int, int]) -> None:
        """Keep the always-visible HUD legible as the viewer is resized."""
        viewer_settings = getattr(self, "_viewer_runtime_settings", None)
        viewer_ui_scale = _viewer_ui_scale_for_window_size(
            _viewer_ui_surface_size(getattr(self, "wnd", None), window_size),
            environ={} if viewer_settings is not None else None,
            configured_scale=(
                viewer_settings.viewer_ui_scale
                if viewer_settings is not None
                else None
            ),
        )
        geometry_scale = self.RIGHT_COLUMN_PANEL_SCALE * viewer_ui_scale
        text_scale = (
            self.RIGHT_COLUMN_PANEL_TEXT_SCALE
            * min(viewer_ui_scale, self.RIGHT_COLUMN_PANEL_TEXT_MAX_UI_SCALE)
        )
        label_text_scale = (
            self.RIGHT_COLUMN_PANEL_LABEL_TEXT_SCALE
            * min(viewer_ui_scale, self.RIGHT_COLUMN_PANEL_TEXT_MAX_UI_SCALE)
        )
        button_text_scale = (
            self.RIGHT_COLUMN_PANEL_BUTTON_TEXT_SCALE
            * min(viewer_ui_scale, self.RIGHT_COLUMN_PANEL_TEXT_MAX_UI_SCALE)
        )
        if (
            viewer_ui_scale == self._right_column_ui_scale()
            and geometry_scale == self._right_column_geometry_scale()
            and text_scale == self._right_column_text_scale()
            and label_text_scale == self._right_column_label_text_scale()
            and button_text_scale == self._right_column_button_text_scale()
        ):
            return

        self._viewer_ui_scale = viewer_ui_scale
        self._right_column_panel_scale = geometry_scale
        self._right_column_panel_text_scale = text_scale
        self._right_column_panel_label_text_scale = label_text_scale
        self._right_column_panel_button_text_scale = button_text_scale
        self._layout_cache_size = None
        self._layout_cache_result = None

        for control in (
            getattr(self, "light_stepper", None),
            getattr(self, "ambient_stepper", None),
            getattr(self, "render_distance_stepper", None),
        ):
            setter = getattr(control, "set_scale", None)
            if callable(setter):
                setter(
                    text_scale=text_scale,
                    geometry_scale=geometry_scale,
                    label_text_scale=label_text_scale,
                )

        setter = getattr(getattr(self, "render_mode_buttons", None), "set_scale", None)
        if callable(setter):
            setter(text_scale=button_text_scale, geometry_scale=geometry_scale)

    def _right_column_layout(self, window_size: tuple[int, int]) -> dict:
        """
        Returns a dict with every position the right-side column needs:
        'brightness_anchor', 'ambient_anchor' (the GLOBAL LIGHT stepper),
        'render_distance_anchor' (note: this stepper moved to the right
        column per request, no longer on the left), and 'buttons_top_y'
        -- each stepper anchor already accounts for its own label space
        above it (see StepperControl.render's label_above handling).

        Stack order, top to bottom: Brightness, Global Light, Render
        Distance, then the button block. The panel is the shared horizontal
        layout container: labels, steppers, and buttons all use its content
        center so a wide label cannot make the controls appear right-aligned
        within the backplate.
        """
        self._update_right_column_hud_scale(window_size)
        if window_size == self._layout_cache_size:
            return self._layout_cache_result

        w, h = window_size

        # Label reserve matches StepperControl.render's own label metrics so
        # this stays correct if that label styling ever changes (rather
        # than a second hard-coded guess at the same number).
        from caveviewer.gui import bitmap_font
        panel_scale = self._right_column_geometry_scale()
        panel_label_text_scale = self._right_column_label_text_scale()
        viewer_ui_scale = self._right_column_ui_scale()
        fixed_label_size = bitmap_font.pixel_size_at_text_scale(
            StepperControl.LABEL_TEXT_SIZE,
            StepperControl.FIXED_TEXT_SCALE * panel_label_text_scale,
        )
        label_height = bitmap_font.text_height_px(fixed_label_size)
        label_reserve = label_height + 8 * panel_scale
        label_widths = (
            bitmap_font.text_width_px(self.light_stepper.label, fixed_label_size),
            bitmap_font.text_width_px(self.ambient_stepper.label, fixed_label_size),
            bitmap_font.text_width_px(
                self.render_distance_stepper.label,
                fixed_label_size,
            ),
        )
        stepper_widths = (
            self.light_stepper.total_width(),
            self.ambient_stepper.total_width(),
            self.render_distance_stepper.total_width(),
        )

        button_block_height = RenderModeButtons.total_stack_height(scale=panel_scale)
        content_bottom_inset = (
            self.RIGHT_COLUMN_PANEL_BOTTOM_MARGIN + self.RIGHT_COLUMN_PANEL_BOTTOM_PAD
        ) * viewer_ui_scale

        # Build the stack from the BOTTOM up: button block's bottom sits
        # RIGHT_COLUMN_BOTTOM_MARGIN above the window's bottom edge.
        buttons_bottom_y = h - content_bottom_inset
        buttons_top_y = buttons_bottom_y - button_block_height

        # RenderModeButtons may adjust its effective scale for the available
        # height. Use that same width here so the panel, rendering, and hit
        # testing share one horizontal geometry.
        button_layout = self.render_mode_buttons._group_layout(
            window_size,
            buttons_top_y,
        )
        button_width = RenderModeButtons.BUTTON_WIDTH * button_layout["scale"]

        # Size the interior for the widest rendered item, then center every
        # child in it. Previously the steppers and buttons were independently
        # right-aligned before the panel grew left to include wider labels.
        content_width = max(*stepper_widths, *label_widths, button_width)
        side_pad = self.RIGHT_COLUMN_PANEL_SIDE_PAD * viewer_ui_scale
        panel_right = w - (self.RIGHT_COLUMN_PANEL_RIGHT_MARGIN * viewer_ui_scale)
        panel_left = panel_right - content_width - 2 * side_pad
        content_center_x = panel_left + side_pad + content_width / 2.0

        render_distance_bottom_y = buttons_top_y - self.RIGHT_COLUMN_BUTTON_GROUP_GAP * panel_scale
        render_distance_anchor_y = render_distance_bottom_y - self.render_distance_stepper.total_height()

        ambient_bottom_y = render_distance_anchor_y - label_reserve - self.RIGHT_COLUMN_GAP * panel_scale
        ambient_anchor_y = ambient_bottom_y - self.ambient_stepper.total_height()

        brightness_bottom_y = ambient_anchor_y - label_reserve - self.RIGHT_COLUMN_GAP * panel_scale
        brightness_anchor_y = brightness_bottom_y - self.light_stepper.total_height()

        brightness_anchor_x = content_center_x - stepper_widths[0] / 2.0
        ambient_anchor_x = content_center_x - stepper_widths[1] / 2.0
        render_distance_anchor_x = content_center_x - stepper_widths[2] / 2.0
        label_gap = self.RIGHT_COLUMN_PANEL_LABEL_GAP * panel_scale
        panel_top = min(
            brightness_anchor_y - label_height - label_gap,
            ambient_anchor_y - label_height - label_gap,
            render_distance_anchor_y - label_height - label_gap,
        ) - (self.RIGHT_COLUMN_PANEL_TOP_PAD * viewer_ui_scale)
        panel_bottom = h - (self.RIGHT_COLUMN_PANEL_BOTTOM_MARGIN * viewer_ui_scale)

        result = {
            "brightness_anchor": (brightness_anchor_x, brightness_anchor_y),
            "ambient_anchor": (ambient_anchor_x, ambient_anchor_y),
            "render_distance_anchor": (render_distance_anchor_x, render_distance_anchor_y),
            "buttons_top_y": buttons_top_y,
            "button_right_inset": w - (content_center_x + button_width / 2.0),
            "content_bottom_inset": content_bottom_inset,
            "content_center_x": content_center_x,
            "panel_rect": (panel_left, panel_top, panel_right, panel_bottom),
        }
        self._layout_cache_size = window_size
        self._layout_cache_result = result
        return result

    def _right_column_panel_rect(self, window_size: tuple[int, int], column: dict | None = None) -> tuple[float, float, float, float]:
        """Bounds for the shared backplate behind the right-side HUD column."""
        if column is None:
            column = self._right_column_layout(window_size)
        return column["panel_rect"]

    def _render_right_column_panel(self, window_size: tuple[int, int], column: dict | None = None) -> None:
        """Draw a shared translucent panel behind the right-side HUD controls."""
        if column is None:
            column = self._right_column_layout(window_size)

        x0, y0, x1, y1 = self._right_column_panel_rect(window_size, column)
        w, h = window_size
        verts = []

        def px_to_ndc(x: float, y: float) -> tuple[float, float]:
            nx = (x / w) * 2.0 - 1.0
            ny = 1.0 - (y / h) * 2.0
            return nx, ny

        def add_quad_px(qx0: float, qy0: float, qx1: float, qy1: float, rgba: tuple[float, float, float, float]) -> None:
            nx0, ny0 = px_to_ndc(qx0, qy0)
            nx1, ny1 = px_to_ndc(qx1, qy1)
            top, bottom = max(ny0, ny1), min(ny0, ny1)
            left, right = min(nx0, nx1), max(nx0, nx1)
            quad = [
                (left, bottom), (right, bottom), (right, top),
                (left, bottom), (right, top), (left, top),
            ]
            for vx, vy in quad:
                verts.append((vx, vy, *rgba))

        add_quad_px(x0, y0, x1, y1, self.RIGHT_COLUMN_PANEL_FILL_RGBA)

        border = self.RIGHT_COLUMN_PANEL_BORDER_PX
        border_color = self.RIGHT_COLUMN_PANEL_BORDER_RGBA
        add_quad_px(x0, y0, x1, y0 + border, border_color)
        add_quad_px(x0, y1 - border, x1, y1, border_color)
        add_quad_px(x0, y0, x0 + border, y1, border_color)
        add_quad_px(x1 - border, y0, x1, y1, border_color)

        data = np.array(verts, dtype=np.float32)
        self._hud_panel_vbo.write(data.tobytes())

        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self._hud_panel_vao.render(moderngl.TRIANGLES, vertices=len(verts))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def _render_minimap(self, window_size: tuple[int, int]) -> None:
        """Draw the minimap in the normal HUD, keeping it out of recordings."""
        if self.minimap is not None:
            self.minimap.render(window_size, self.camera.position, self.camera.forward(),
                                self._bookmarks)

    def _render_capture_status_message(self, window_size: tuple[int, int]) -> None:
        now = time.perf_counter()
        status = self._ensure_recording_controller().active_status(now=now)
        if status is None:
            return

        message = status.message
        detail = status.detail
        kind = status.kind or "info"

        w, h = window_size
        self._render_recording_countdown_scrim(window_size, alpha=0.42)

        symbol = {
            "success": "OK",
            "error": "!",
            "cancel": "X",
            "info": "...",
        }.get(kind, "...")
        symbol_size = 5.2 if symbol == "OK" else 3.8 if symbol == "..." else 7.2
        center_x = w / 2.0
        ring_center_y = h / 2.0
        self.import_progress_panel.draw_circle_label(
            center_x=center_x,
            center_y=ring_center_y,
            window_size=window_size,
            label=symbol,
            progress=None if kind == "info" else 1.0,
            pixel_size=symbol_size,
            fixed_text_scale=self.UI_TEXT_SCALE,
            stage=message,
            note=detail,
        )
        self._ensure_capture_workflow().mark_exit_status_presented(
            now=time.perf_counter()
        )

    def _render_recording_countdown_scrim(self, window_size: tuple[int, int], alpha: float = 0.62) -> None:
        """Darken the cave view behind the countdown ring without hiding it."""
        w, h = window_size
        verts = []

        def px_to_ndc(x: float, y: float) -> tuple[float, float]:
            return (x / w) * 2.0 - 1.0, 1.0 - (y / h) * 2.0

        def add_quad_px(qx0: float, qy0: float, qx1: float, qy1: float,
                        rgba: tuple[float, float, float, float]) -> None:
            nx0, ny0 = px_to_ndc(qx0, qy0)
            nx1, ny1 = px_to_ndc(qx1, qy1)
            top, bottom = max(ny0, ny1), min(ny0, ny1)
            left, right = min(nx0, nx1), max(nx0, nx1)
            quad = [
                (left, bottom), (right, bottom), (right, top),
                (left, bottom), (right, top), (left, top),
            ]
            for vx, vy in quad:
                verts.append((vx, vy, *rgba))

        add_quad_px(0, 0, w, h, (0.001, 0.002, 0.005, alpha))
        data = np.array(verts, dtype=np.float32)
        self._status_panel_vbo.write(data.tobytes())

        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self._status_panel_vao.render(moderngl.TRIANGLES, vertices=len(verts))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def _frame_phase(self) -> ViewerFramePhase:
        """Select this callback's non-GL session phase."""
        setup_complete = bool(getattr(self, "_window_setup_complete", False))
        closing_requested = bool(getattr(self, "_closing_requested", False))
        if not setup_complete or closing_requested:
            return ViewerFramePhase.INACTIVE
        request = self._workflow_render_request()
        if request is not None:
            return request.phase
        return self._ensure_frame_scheduler().phase_for(
            ViewerFrameState(
                setup_complete=setup_complete,
                closing_requested=closing_requested,
                iconified=bool(getattr(self, "_is_iconified", False)),
                finalizing_capture=self._capture_close_pending(),
                import_active=bool(getattr(self, "_import_active", False)),
                map_loaded=bool(getattr(self, "_has_map_loaded", False)),
            )
        )

    def on_render(self, current_time: float, frame_time: float):
        if self._frame_phase() is ViewerFramePhase.INACTIVE:
            return
        if not getattr(self, "_first_render_checkpoint_recorded", False):
            self._first_render_checkpoint_recorded = True
            record_runtime_stage(
                "viewer_first_render_entered",
                window_size=getattr(getattr(self, "wnd", None), "size", None),
            )

        # Backends can miss iconify callbacks on Dock minimize; poll a
        # few common window flags each frame as a safety net.
        runtime_iconified = self._query_runtime_iconified_state()
        self._set_background_pause(runtime_iconified, "runtime window state")

        frame_phase = self._frame_phase()
        frame_scheduler = self._ensure_frame_scheduler()
        if frame_phase is ViewerFramePhase.ICONIFIED:
            # Keep minimize mode cheap: no streaming updates/uploads while
            # iconified.  Poll low-frequency completion state without blocking
            # the render/window callback.
            if frame_scheduler.is_due(
                "iconified",
                _ICONIFIED_RENDER_POLL_INTERVAL_S,
                now=time.perf_counter(),
            ):
                self._drain_recording_stop_results()
                if self._slice_work_pending():
                    self._update_slice_export()
                if self._capture_close_pending():
                    self._update_manual_dive_trace()
                    if self._complete_escape_capture_cancellation_if_ready():
                        return
                    self._complete_exit_capture_finalization_if_ready(
                        allow_unpresented_status=True
                    )
                self._drain_due_saved_artifact_reveals()
            return
        frame_scheduler.reset_throttle("iconified")

        bitmap_font.set_raster_scale(_window_pixel_ratio(self.wnd))

        # Keep render-mode button effects synced to loading state even
        # on frames that early-return before normal HUD interaction.
        self._sync_render_mode_loading_policy()
        self._drain_recording_stop_results()
        self._drain_due_saved_artifact_reveals()

        if frame_phase is ViewerFramePhase.FINALIZING_CAPTURE:
            self._update_manual_dive_trace()
            if self._slice_work_pending():
                self._update_slice_export()
            if self._complete_escape_capture_cancellation_if_ready():
                return
            if self._complete_exit_capture_finalization_if_ready():
                return
            # Do not continue navigation, streaming, or map interaction while
            # a user-visible artifact is still being published. The centered
            # status keeps the same visual hierarchy as import and capture UI.
            self.ctx.clear(0.02, 0.02, 0.03)
            self._render_capture_status_message(self.wnd.size)
            return

        if self._startup_focus_enabled:
            self._request_startup_focus_once()

        # Background import in flight: drain worker results on every callback
        # and redraw the progress panel every callback. Window backends may
        # still present/swap after this method returns, so skipping draws here
        # can expose stale back buffers as visible flicker during first-time
        # imports.
        if frame_phase is ViewerFramePhase.IMPORTING:
            self._drain_import_queue()
            if not self._import_active:
                return
            self.ctx.clear(0.02, 0.02, 0.03)
            fraction = self._import_progress_fraction
            # When the real fraction is near zero (numpy is crunching
            # faces and can't report sub-step progress), pulse the indicator
            # gently between 0 and 2 % so it looks alive.  The pulse is
            # capped below the first real progress step (3 %) so the
            # max() inside import_progress_panel takes over cleanly once
            # measurable progress begins.
            if fraction < 0.021:
                t = time.perf_counter()
                fraction = abs(math.sin(t * 1.2)) * 0.02
            import_controller = self._ensure_import_controller()
            frame = self._ensure_map_opening_progress_session().observe_import(
                self._import_map_name,
                self._import_progress_stage,
                fraction,
                note=self._import_progress_note,
                supporting_note_override=import_controller.transient_progress_note(),
            )
            self._render_map_opening_progress(frame)
            return
        frame_scheduler.reset_throttle("import_progress")

        if frame_phase is ViewerFramePhase.STARTUP:
            if getattr(self, "_startup_map_load_pending", None) is not None:
                self._load_startup_map_after_splash()
                return
            if frame_scheduler.is_due(
                "import_pause_notice",
                _IMPORT_PAUSE_NOTICE_RENDER_INTERVAL_S,
                now=time.perf_counter(),
            ):
                if self._render_import_pause_notice_if_active():
                    return
                frame_scheduler.reset_throttle("import_pause_notice")
            elif getattr(self, "_import_pause_notice_until", None) is not None:
                return
            # First frame with no map loaded yet: draw the loading panel
            # immediately so the user sees the logo instead of a blank window.
            # The actual import starts on the next frame so the splash has a
            # chance to present before import startup work contends with the
            # render loop.
            if self._pending_import_started:
                return
            self._render_pending_import_splash()
            if not self._pending_import_splash_rendered:
                self._pending_import_splash_rendered = True
                return
            self._pending_import_started = True
            self._run_pending_import()
            return

        self._render_interactive_frame(current_time, frame_time)

    def _render_interactive_frame(
        self,
        current_time: float,
        frame_time: float,
    ) -> None:
        """Render one interactive frame after non-GL session scheduling."""
        frame_start = time.perf_counter()
        benchmark_controller = self._active_benchmark_controller()
        benchmark_active = (
            benchmark_controller is not None
            and not benchmark_controller.finished
        )
        self._update_texture_validation()

        # Sleep/wake (or a debugger stop) can yield a very large frame_time
        # and leave input/capture state stale (e.g. key-release never seen).
        # Reset transient input flags on these discontinuities.
        if frame_time > 2.0:
            self._reset_transient_input_state("long frame gap")

        t_input = time.perf_counter()
        dt = max(frame_time, 1e-4)
        if benchmark_active:
            if benchmark_controller.started:
                benchmark_controller.update_camera(self.camera, time.perf_counter())
        elif self._recorded_dive_is_active():
            if self._recorded_dive_is_paused():
                self._handle_paused_recorded_dive_input(dt)
                self._update_recorded_dive(now=time.perf_counter())
            elif self._continuous_input_has_navigation_intent(dt):
                self._stop_recorded_dive(reason="manual_control")
                self._handle_continuous_input(dt)
            else:
                self._update_recorded_dive(now=time.perf_counter())
        else:
            self._handle_manual_input_frame(dt, now=time.perf_counter())
        self._update_manual_dive_trace()
        if self._slice_work_pending():
            self._update_slice_export()
        input_ms = (time.perf_counter() - t_input) * 1000.0

        # Apply the render-distance control's current value before the
        # streaming world recalculates this frame -- a click on +/- takes
        # effect immediately rather than waiting for the camera to move
        # (see the matching check in StreamingWorld.update(), which
        # detects a changed load_radius_cells on its own, not just a
        # moved camera -- this assignment is what actually gives it a
        # changed value to detect).
        target_load_radius = self._target_streaming_load_radius()
        if self.world.config.load_radius_cells != target_load_radius:
            self.world.config.load_radius_cells = target_load_radius

        t0 = time.perf_counter()
        streaming_timing = self._new_streaming_frame_timing()
        self._streaming_frame_timing = streaming_timing
        try:
            t_update = time.perf_counter()
            self.world.update(
                self.camera.position.astype(np.float32),
                cell_priority_key=self._streaming_cell_priority_key(),
            )
            update_elapsed_s = time.perf_counter() - t_update
            streaming_timing["update_ms"] = update_elapsed_s * 1000.0
            self._log_main_thread_stall("streaming update", update_elapsed_s)

            pre_drain_stats = self.world.stats()
            (
                upload_chunks_per_frame,
                upload_operations_per_chunk,
                upload_time_budget_ms,
            ) = self._streaming_upload_limits(pre_drain_stats)
            self._current_upload_operations_per_chunk = upload_operations_per_chunk
            self._current_upload_time_budget_ms = upload_time_budget_ms
            t_drain = time.perf_counter()
            t_ready_drain = time.perf_counter()
            self.world.drain_ready_chunks(
                self._on_chunk_ready, self._on_chunk_unload,
                max_per_frame=upload_chunks_per_frame,
                time_budget_ms=upload_time_budget_ms,
            )
            ready_drain_elapsed_s = time.perf_counter() - t_ready_drain
            streaming_timing["ready_drain_ms"] = ready_drain_elapsed_s * 1000.0
            self._log_main_thread_stall(
                "ready chunk drain",
                ready_drain_elapsed_s,
                ready=int(pre_drain_stats.get("ready", 0)),
                pending=int(pre_drain_stats.get("pending", 0)),
                max_per_frame=upload_chunks_per_frame,
                time_budget_ms=upload_time_budget_ms,
            )
            t_failure_drain = time.perf_counter()
            self._drain_streaming_worker_failures()
            streaming_timing["failure_drain_ms"] = (
                time.perf_counter() - t_failure_drain
            ) * 1000.0
            streaming_timing["drain_ms"] = (time.perf_counter() - t_drain) * 1000.0
            self._record_upload_slice_sizes(streaming_timing)
        finally:
            upload_manager = getattr(self, "_chunk_upload_manager", None)
            if upload_manager is not None:
                upload_manager.clear_frame_timing()
            self._streaming_frame_timing = None
        streaming_ms = (time.perf_counter() - t0) * 1000.0
        stats = self.world.stats()
        if not self._initial_chunks_loaded and self._initial_chunk_load_is_ready(stats):
            self._initial_chunks_loaded = True
            self._log_initial_compilation_complete(stats)

        # As soon as prep crosses the readiness threshold, hold a brief
        # fully-complete frame so the progress bar doesn't disappear abruptly.
        if self._initial_chunks_loaded and not self._chunk_prep_completion_armed:
            self._chunk_prep_completion_armed = True
            self._chunk_prep_complete_until = (
                time.perf_counter() + self._CHUNK_PREP_COMPLETE_HOLD_SECONDS
            )

        # Show a loading indicator while the initial chunks stream in from disk.
        # Without this the screen is black until the first chunk arrives, which
        # can take several seconds on slow hardware or large maps.
        now = time.perf_counter()
        if not self._initial_chunks_loaded:
            if benchmark_active and benchmark_controller.exceeded_max_runtime(now):
                self._finish_benchmark(reason="max_runtime_exceeded")
                self.close()
                return
            _map_name = os.path.basename(self.manifest.get("source_obj", "map"))
            raw_fraction = self._initial_chunk_load_progress(stats)
            target = min(self._CHUNK_PREP_MAX_FRACTION, raw_fraction * self._CHUNK_PREP_MAX_FRACTION)
            self._chunk_prep_progress = max(self._chunk_prep_progress, target)
            frame = self._ensure_map_opening_progress_session().observe_streaming(
                _map_name,
                self._chunk_prep_progress,
            )
            self._render_map_opening_progress(frame)
            return

        if self._chunk_prep_complete_until is not None and now < self._chunk_prep_complete_until:
            if benchmark_active and benchmark_controller.exceeded_max_runtime(now):
                self._finish_benchmark(reason="max_runtime_exceeded")
                self.close()
                return
            _map_name = os.path.basename(self.manifest.get("source_obj", "map"))
            frame = self._ensure_map_opening_progress_session().complete(_map_name)
            self._render_map_opening_progress(frame)
            return

        self._chunk_prep_complete_until = None
        self._ensure_map_opening_progress_session().finish()
        if benchmark_active:
            self._sync_render_mode_loading_policy()

        t_scene_setup = time.perf_counter()
        self.ctx.clear(*self.color_picker.color)  # background ("void") color, adjustable via the COLOR button

        aspect = self.wnd.size[0] / max(self.wnd.size[1], 1)
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(aspect)

        self.program["u_view"].write(view.T.tobytes())
        self.program["u_projection"].write(proj.T.tobytes())
        _pos = self.camera.position
        self.program["u_camera_pos"].value = (float(_pos[0]), float(_pos[1]), float(_pos[2]))
        self.program["u_light_color"].value = (1.0, 0.95, 0.85)  # warm headlamp tone
        self.program["u_light_intensity"].value = float(self.light_stepper.value)
        # GLOBAL LIGHT stepper (0-10) maps linearly onto the shader's
        # actual ambient range -- see _AMBIENT_MIN/_AMBIENT_MAX's
        # docstring above for why 0 reproduces the app's original fixed
        # ambient value rather than true darkness.
        ambient_t = self.ambient_stepper.value / self.ambient_stepper.max_value
        ambient_value = self._AMBIENT_MIN + ambient_t * (self._AMBIENT_MAX - self._AMBIENT_MIN)
        self.program["u_ambient"].value = ambient_value
        self.program["u_texture_enabled"].value = self.render_mode_buttons.texture_enabled
        scene_setup_ms = (time.perf_counter() - t_scene_setup) * 1000.0

        t0 = time.perf_counter()

        # Solid pass (textured, or plain gray if Texture is off) only
        # draws when at least one of "show texture" or "wireframe is off"
        # is true. In other words: skip the solid pass entirely when the
        # person has explicitly turned Texture off AND turned Mesh
        # (wireframe) on -- that combination means "show me pure
        # wireframe, nothing else", and the solid pass would otherwise
        # always render underneath the wireframe lines regardless of the
        # Texture toggle, which defeats the point of turning texture off
        # in the first place when inspecting wireframe-only.
        show_solid_pass = self.render_mode_buttons.texture_enabled or not self.render_mode_buttons.wireframe_enabled

        # Frustum-cull loaded chunks against the current view before drawing.
        # Build _visible_cells once so both solid and wireframe passes share
        # the same culled set without repeating the test.
        t_cull = time.perf_counter()
        _visible_cells = self._visible_chunk_gpu_objects(view, proj)
        _chunks_drawn = len(_visible_cells)
        mesh_cull_ms = (time.perf_counter() - t_cull) * 1000.0
        visual_stats = self._initial_visual_readiness_stats(
            stats,
            _chunks_drawn,
            visible_cells=_visible_cells,
            view=view,
            projection=proj,
        )

        # u_texture always refers to sampler unit 0 -- set it once before
        # the loop rather than redundantly on every single draw call.
        def _draw_visible_mesh() -> None:
            self.program["u_texture"].value = 0
            if show_solid_pass:
                for cell, vao_list in _visible_cells:
                    for vao, vbo, mat_name, texture in vao_list:
                        texture.use(location=0)
                        vao.render(moderngl.TRIANGLES)

            # Wireframe pass: drawn whenever Mesh is toggled on. If the solid
            # pass also drew (texture or gray surface visible), this overlays
            # triangulation on top of it. If the solid pass was skipped (the
            # texture-off + wireframe-on combination above), this is the only
            # thing that draws -- true wireframe-only.
            if self.render_mode_buttons.wireframe_enabled:
                # NOTE: this draws coincident wireframe lines directly on top of
                # the solid pass's geometry, which can show minor z-fighting/
                # flicker on some GPUs since both passes write near-identical
                # depth values. A polygon-offset bias would clean this up, but
                # since the bias amount needs hand-tuning against moderngl's
                # actual ctx.polygon_offset API (left out here rather than
                # guess at a value that could silently do nothing or look
                # wrong), this is a known minor cosmetic rough edge -- the
                # wireframe is still fully readable, just not perfectly crisp
                # in rare cases.
                self.ctx.wireframe = True
                for cell, vao_list in _visible_cells:
                    for vao, vbo, mat_name, texture in vao_list:
                        vao.render(moderngl.TRIANGLES)
                self.ctx.wireframe = False

        mesh_gpu_query_wait_ms = 0.0
        t_submit = time.perf_counter()
        if self._gpu_draw_timer_enabled:
            # GPU timer queries are useful diagnostics but reading the result
            # in the same frame can block until the driver has completed the
            # measured work. Keep that synchronization out of normal viewing;
            # enable CAVEVIEWER_GPU_DRAW_TIMER=1 only while actively measuring
            # GPU-side draw cost.
            with self.ctx.query(time=True) as _gpu_q:
                _draw_visible_mesh()
            mesh_submit_ms = (time.perf_counter() - t_submit) * 1000.0
            t_query_wait = time.perf_counter()
            self._last_gpu_draw_ms = _gpu_q.elapsed / 1_000_000
            mesh_gpu_query_wait_ms = (time.perf_counter() - t_query_wait) * 1000.0
        else:
            self._last_gpu_draw_ms = None
            _draw_visible_mesh()
            mesh_submit_ms = (time.perf_counter() - t_submit) * 1000.0
        mesh_draw_ms = (time.perf_counter() - t0) * 1000.0

        def _render_recording_frame(
            framebuffer: moderngl.Framebuffer,
            output_size: tuple[int, int],
        ) -> None:
            output_width, output_height = output_size
            previous_fbo = getattr(self.ctx, "fbo", None)
            previous_screen_viewport = getattr(self.ctx.screen, "viewport", None)
            previous_framebuffer_viewport = getattr(framebuffer, "viewport", None)
            recording_proj = self.camera.projection_matrix(
                output_width / max(output_height, 1)
            )
            try:
                framebuffer.use()
                framebuffer.viewport = (0, 0, output_width, output_height)
                self.ctx.clear(*self.color_picker.color)
                self.program["u_projection"].write(recording_proj.T.tobytes())
                self.program["u_view"].write(view.T.tobytes())
                _draw_visible_mesh()
            finally:
                try:
                    if previous_fbo is not None:
                        previous_fbo.use()
                    else:
                        self.ctx.screen.use()
                except Exception:
                    try:
                        self.ctx.screen.use()
                    except Exception:
                        pass
                if previous_screen_viewport is not None:
                    try:
                        self.ctx.screen.viewport = previous_screen_viewport
                    except Exception:
                        pass
                if previous_framebuffer_viewport is not None:
                    try:
                        framebuffer.viewport = previous_framebuffer_viewport
                    except Exception:
                        pass
                self.ctx.wireframe = False
                self.program["u_projection"].write(proj.T.tobytes())
                self.program["u_view"].write(view.T.tobytes())

        recording_read_ms = 0.0
        recording_stage_ms = 0.0
        recording_drain_ms = 0.0
        workflow_request = self._workflow_render_request()
        capture_overlay_mode = (
            workflow_request.capture_overlay_mode
            if workflow_request is not None
            else self._ensure_capture_workflow().overlay_mode_for(
                CaptureOverlayState(
                    recording_armed=self._recording_hides_hud(),
                    manual_dive_trace_countdown_active=(
                        self._ensure_manual_dive_trace_controller().countdown_active
                    ),
                    slice_countdown_active=(
                        self._ensure_slice_selection_controller().countdown_active
                    ),
                )
            )
        )
        if capture_overlay_mode is CaptureOverlayMode.RECORDING:
            now = time.perf_counter()
            if self._recording_countdown_until is not None and now < self._recording_countdown_until:
                self._render_countdown_overlay(
                    now=now,
                    controller=self._ensure_recording_controller(),
                    start_number=self.RECORDING_COUNTDOWN_START_NUMBER,
                    title=self.RECORDING_COUNTDOWN_TITLE,
                    note=self._countdown_cancel_note("R"),
                )
            else:
                recording_read_ms = self._recording_update_after_scene(
                    now,
                    render_frame=_render_recording_frame,
                )
                recording_stage_ms = self._recording_last_stage_ms
                recording_drain_ms = self._recording_last_drain_ms
                # The countdown has already explained the controls; leave the
                # active recording view clear of a persistent status banner.
                self._render_capture_status_message(self.wnd.size)
            overlay_ms = 0.0
        elif capture_overlay_mode is CaptureOverlayMode.MANUAL_DIVE_TRACE_COUNTDOWN:
            now = time.perf_counter()
            self._render_countdown_overlay(
                now=now,
                controller=self._ensure_manual_dive_trace_controller(),
                start_number=self.MANUAL_DIVE_TRACE_COUNTDOWN_START_NUMBER,
                title=self.MANUAL_DIVE_TRACE_COUNTDOWN_TITLE,
                note=self._countdown_cancel_note("T"),
            )
            overlay_ms = 0.0
        elif capture_overlay_mode is CaptureOverlayMode.SLICE_COUNTDOWN:
            now = time.perf_counter()
            self._render_countdown_overlay(
                now=now,
                controller=self._ensure_slice_selection_controller(),
                start_number=self.SLICE_COUNTDOWN_START_NUMBER,
                title=self.SLICE_COUNTDOWN_TITLE,
                note=self._countdown_cancel_note("C"),
            )
            overlay_ms = 0.0
        else:
            # Overlay HUD elements draw last, on top of the 3D scene, each with
            # their own depth-disabled 2D pass.
            t0 = time.perf_counter()

            # Whole right-side column -- brightness, global light, render
            # distance, then the Mesh/Texture/Shade/Open/Help/Color buttons -- is
            # laid out as one group anchored to the bottom-right corner. See
            # _right_column_layout()'s docstring for why this is computed in
            # one place rather than each piece anchoring itself independently.
            column = self._right_column_layout(self.wnd.size)
            brightness_anchor_x, brightness_anchor_y = column["brightness_anchor"]
            ambient_anchor_x, ambient_anchor_y = column["ambient_anchor"]
            render_distance_anchor_x, render_distance_anchor_y = column["render_distance_anchor"]
            buttons_top_y = column["buttons_top_y"]

            self._render_right_column_panel(self.wnd.size, column)
            self.light_stepper.render(self.wnd.size, brightness_anchor_x, brightness_anchor_y, label_above=True)
            self.ambient_stepper.render(self.wnd.size, ambient_anchor_x, ambient_anchor_y, label_above=True)
            self.render_distance_stepper.render(self.wnd.size, render_distance_anchor_x, render_distance_anchor_y,
                                                label_above=True)

            self._render_minimap(self.wnd.size)

            self.render_mode_buttons.render(self.wnd.size, buttons_top_y,
                              help_active=self.controls_overlay.is_manual_mode,
                              color_active=self.color_picker.is_active,
                              right_inset=column["button_right_inset"])

            # Color picker panel draws on top of the regular HUD elements (it
            # dims the 3D view behind it, same visual language as the Help
            # screen) but still below the controls overlay, consistent with
            # Help also losing to a loading overlay if both somehow overlap.
            self.color_picker.render(self.wnd.size)

            # Controls/loading overlay draws last of all, on top of every
            # other UI element -- while it's showing, it's meant to be the
            # thing you're looking at (it's explaining what the other UI
            # pieces do), so it should never be obscured by them.
            self.controls_overlay.update(visual_stats)
            self.controls_overlay.render(self.wnd.size)
            if not self._render_active_capture_instruction(self.wnd.size):
                self._render_dive_status(self.wnd.size)
            self._render_capture_status_message(self.wnd.size)
            overlay_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - frame_start) * 1000.0
        other_ms = max(
            0.0,
            total_ms
            - input_ms
            - streaming_ms
            - scene_setup_ms
            - mesh_draw_ms
            - recording_read_ms
            - overlay_ms,
        )

        if benchmark_active:
            benchmark_now = time.perf_counter()
            if not getattr(self, "_initial_visual_ready", False):
                if benchmark_controller.exceeded_max_runtime(benchmark_now):
                    self._finish_benchmark(reason="max_runtime_exceeded")
                    self.close()
                return
            if not benchmark_controller.started:
                self.controls_overlay.dismiss_begin_screen()
                benchmark_controller.update_camera(self.camera, benchmark_now)
                return
            benchmark_complete = benchmark_controller.record_frame(
                now=benchmark_now,
                frame_ms=total_ms,
                streaming_ms=streaming_ms,
                scene_setup_ms=scene_setup_ms,
                mesh_draw_ms=mesh_draw_ms,
                mesh_cull_ms=mesh_cull_ms,
                mesh_submit_ms=mesh_submit_ms,
                overlay_ms=overlay_ms,
                other_ms=other_ms,
                drawn_chunks=_chunks_drawn,
                resident_chunks=len(self._chunk_gpu_objects),
                world_stats=stats,
                streaming_timing=streaming_timing,
            )
            if benchmark_complete:
                self._finish_benchmark(reason="completed")
                self.close()
                return
            if benchmark_controller.exceeded_max_runtime(benchmark_now):
                self._finish_benchmark(reason="max_runtime_exceeded")
                self.close()
                return

        # Spike detection: track a short rolling average of frame times, and
        # if a frame comes in notably above that average, print a one-line
        # breakdown of where the time went. This is the diagnostic for
        # tracking down any remaining stutter -- rather than guess at
        # causes, the next time a stutter happens this will print exactly
        # which section (chunk streaming, mesh draw, or overlay draw) was
        # responsible, plus chunk-loading stats at that moment.
        self._frame_time_history.append(total_ms)
        if len(self._frame_time_history) > 30:
            self._frame_time_history.pop(0)
        rolling_avg = sum(self._frame_time_history) / len(self._frame_time_history)

        if len(self._frame_time_history) >= 10 and total_ms > max(rolling_avg * 3, 25.0):
            stats = self.world.stats()
            gpu_draw_text = self._format_optional_ms(self._last_gpu_draw_ms)
            _LOG.warning(f"FRAME SPIKE: {total_ms:.1f}ms (avg {rolling_avg:.1f}ms) | "
                         f"input={input_ms:.1f}ms streaming={streaming_ms:.1f}ms "
                         f"scene_setup={scene_setup_ms:.1f}ms mesh_draw={mesh_draw_ms:.1f}ms "
                         f"mesh_cull={mesh_cull_ms:.1f}ms "
                         f"mesh_submit={mesh_submit_ms:.1f}ms "
                         f"gpu_query_wait={mesh_gpu_query_wait_ms:.1f}ms "
                         f"gpu_draw={gpu_draw_text} "
                         f"recording_read={recording_read_ms:.1f}ms "
                         f"recording_stage={recording_stage_ms:.1f}ms "
                         f"recording_drain={recording_drain_ms:.1f}ms "
                         f"overlay={overlay_ms:.1f}ms other={other_ms:.1f}ms | "
                         f"drawn={_chunks_drawn}/{len(self._chunk_gpu_objects)} "
                         f"loaded={stats['loaded']} pending={stats['pending']} "
                         f"ready={stats.get('ready', 0)} "
                         f"unload_pending={stats.get('unload_pending', 0)} "
                         f"wanted={stats.get('wanted', 0)}")
            _LOG.warning(
                "FRAME SPIKE STREAMING DETAIL: %s",
                self._format_streaming_frame_timing(streaming_timing),
            )

        self._frame_active_time_s += (total_ms / 1000.0)
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_print > 2.0:
            wall_interval_s = max(now - self._last_fps_print, 1e-6)
            active_interval_s = max(self._frame_active_time_s, 1e-6)
            rendered_fps = self._frame_count / active_interval_s
            wall_fps = self._frame_count / wall_interval_s
            if _LOG.isEnabledFor(logging.DEBUG):
                stats = self.world.stats()
                gpu_draw_text = self._format_optional_ms(self._last_gpu_draw_ms)
                speed_label = "manual_speed"
                displayed_speed = float(self.camera.move_speed)
                if benchmark_active:
                    route_speed = getattr(
                        benchmark_controller.scenario,
                        "metadata",
                        {},
                    ).get("actual_route_speed_m_per_second")
                    if isinstance(route_speed, (int, float)):
                        speed_label = "route_speed"
                        displayed_speed = float(route_speed)
                _LOG.debug(f"rendered_fps={rendered_fps:.1f} wall_fps={wall_fps:.1f} "
                           f"frame_cost={rolling_avg:.1f}ms "
                           f"| chunks loaded={stats['loaded']} "
                           f"pending={stats['pending']} "
                           f"unload_pending={stats.get('unload_pending', 0)} "
                           f"drawn={_chunks_drawn}/{len(self._chunk_gpu_objects)} "
                           f"| {speed_label}={displayed_speed:.1f}m/s "
                           f"| mesh_cull={mesh_cull_ms:.1f}ms "
                           f"mesh_submit={mesh_submit_ms:.1f}ms "
                           f"gpu_query_wait={mesh_gpu_query_wait_ms:.1f}ms "
                           f"recording_read={recording_read_ms:.1f}ms "
                           f"recording_stage={recording_stage_ms:.1f}ms "
                           f"recording_drain={recording_drain_ms:.1f}ms "
                           f"gpu_draw={gpu_draw_text}")
            self._frame_count = 0
            self._frame_active_time_s = 0.0
            self._last_fps_print = now

    render = on_render  # back-compat alias for older moderngl-window releases

    def _resolve_key(self, keys, *candidate_names):
        """
        Different moderngl-window/pyglet versions have used different names
        for the same key (e.g. LEFT_CONTROL vs LEFT_CTRL). Rather than hard-
        code one name and risk another AttributeError crash on a different
        installed version, try each known alias in turn and cache whichever
        one actually exists on this version's Keys class.
        """
        cache = getattr(self, "_key_resolve_cache", None)
        if cache is None:
            cache = {}
            self._key_resolve_cache = cache
        return viewer_input.resolve_key(keys, *candidate_names, cache=cache)

    def _install_backend_modifier_probe(self) -> None:
        """Capture raw backend modifier bitmasks before they are reduced to shift/ctrl/alt."""
        handler = getattr(self.wnd, "_handle_modifiers", None)
        if not callable(handler):
            return

        def wrapped_handle_modifiers(mods):
            try:
                self._last_raw_modifiers = int(mods)
            except Exception:
                self._last_raw_modifiers = 0
            return handler(mods)

        self.wnd._handle_modifiers = wrapped_handle_modifiers

    def _raw_command_modifier_down(self) -> bool:
        raw_mods = int(getattr(self, "_last_raw_modifiers", 0) or 0)
        backend_module = type(self.wnd).__module__
        return viewer_input.raw_command_modifier_down(raw_mods, backend_module)

    def _key_is_down(self, keys, *candidate_names) -> bool:
        """Return True if any candidate key exists on this backend and is currently held."""
        return viewer_input.key_is_down(keys, self._keys_down, *candidate_names)

    def _resolve_key_optional(self, keys, *candidate_names):
        """Return key code if present on this backend, else None."""
        return viewer_input.resolve_key_optional(keys, *candidate_names)

    def _digit_for_key(self, keys, key) -> int | None:
        """Return bookmark slot (1..9) for a key press across backend key name variants."""
        return viewer_input.digit_for_key(keys, key)

    def _is_zero_key(self, keys, key) -> bool:
        """Check if the key is the 0 key across backend key name variants."""
        return viewer_input.is_zero_key(keys, key)

    def _command_is_down(self, modifiers: KeyModifiers) -> bool:
        keys = self.wnd.keys
        return viewer_input.command_is_down(
            modifiers,
            keys,
            self._keys_down,
            command_modifier_uses_control_fallback=(
                self._active_presentation_profile()
                .command_modifier_uses_control_fallback
            ),
            raw_command_down=self._raw_command_modifier_down(),
        )

    def _control_is_down(self, modifiers: KeyModifiers) -> bool:
        """Check if Control/Ctrl modifier key is currently down."""
        keys = self.wnd.keys
        return viewer_input.control_is_down(modifiers, keys, self._keys_down)

    def _shift_is_down(self, modifiers: KeyModifiers) -> bool:
        """Check if Shift modifier key is currently down."""
        keys = self.wnd.keys
        return viewer_input.shift_is_down(modifiers, keys, self._keys_down)

    def _bookmark_save_modifier_is_down(self, modifiers: KeyModifiers) -> bool:
        """Check if the platform-specific bookmark save modifier is down."""
        save_modifier = self._active_presentation_profile().bookmark_save_modifier
        return viewer_input.bookmark_save_modifier_is_down(
            save_modifier=save_modifier,
            command_down=self._command_is_down(modifiers),
            control_down=self._control_is_down(modifiers),
        )

    def _load_bookmarks(self) -> None:
        self._bookmarks = viewer_bookmarks.load_bookmarks(
            self._bookmarks_path,
            logger=_LOG,
        )

    def _save_bookmarks(self) -> None:
        viewer_bookmarks.save_bookmarks(
            self._bookmarks_path,
            self._bookmarks,
            logger=_LOG,
        )

    def _manual_dive_trace_pose(
        self,
    ) -> manual_dive_trace.ManualDivePose | None:
        camera = getattr(self, "camera", None)
        if camera is None:
            return None
        try:
            return manual_dive_trace.ManualDivePose.from_camera(camera)
        except (AttributeError, TypeError, ValueError):
            return None

    def _slice_camera_position(self) -> tuple[float, float, float] | None:
        """Return the current camera position as a finite export anchor."""
        camera = getattr(self, "camera", None)
        position = getattr(camera, "position", None)
        try:
            values = tuple(float(value) for value in position)
        except (TypeError, ValueError):
            return None
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            return None
        return values  # type: ignore[return-value]

    def _slice_storage_directory(self) -> str:
        """Resolve and preflight the user-configured app-managed map storage."""
        runtime_settings = getattr(self, "_runtime_settings", None)
        if runtime_settings is None:
            from caveviewer.gui.preferences import load_preferences

            configured = str(load_preferences()["map_library_dir"]).strip()
        else:
            configured = runtime_settings.map_library_configuration().directory
        if not configured:
            raise ValueError("The Preferences map-storage folder is not configured.")
        directory = os.path.abspath(os.path.expanduser(configured))
        os.makedirs(directory, exist_ok=True)
        if not os.path.isdir(directory):
            raise ValueError("The Preferences map-storage folder is unavailable.")
        return directory

    def _slice_cave_name(self) -> str:
        """Return the original cave label used as the stable slice-name prefix."""
        manifest = getattr(self, "manifest", None)
        slice_metadata = (
            manifest.get(map_slicing.SLICE_MANIFEST_KEY)
            if isinstance(manifest, Mapping)
            else None
        )
        root_cave_name = (
            slice_metadata.get("root_cave_name")
            if isinstance(slice_metadata, Mapping)
            else None
        )
        if root_cave_name:
            root_label = os.path.basename(str(root_cave_name).strip())
            if root_label:
                # This is already a display label, not a source filename.  In
                # particular, preserve a legitimate cave-name suffix such as
                # ".2" instead of treating it as a file extension.
                return root_label
        map_root = getattr(self, "map_root", None)
        if map_root:
            # The selected folder is the name the Map Library presented to the
            # user. Prefer it over opaque source-model names such as ``D5.obj``
            # so a new slice can inherit the original cave's metadata.
            map_label = os.path.basename(os.path.normpath(str(map_root).strip()))
            if map_label:
                return map_label
        source_obj = manifest.get("source_obj") if isinstance(manifest, Mapping) else None
        raw_name = str(source_obj or "Cave").strip()
        return os.path.splitext(os.path.basename(raw_name))[0] or "Cave"

    def _clear_slice_context(self) -> None:
        self._slice_source_cache_dir = None
        self._slice_storage_parent = None
        self._slice_display_base = None
        self._slice_root_cave_name = None

    def _slice_unavailable(self, detail: str) -> None:
        self._show_capture_status(
            "Slice unavailable",
            detail,
            kind="error",
            duration=self._ensure_artifact_capture_presentation().confirmation_seconds,
        )

    def _start_slice_countdown(self) -> bool:
        """Preflight a Ctrl+C slice action and arm the shared capture countdown."""
        if not self._has_map_loaded:
            return False
        selection = self._ensure_slice_selection_controller()
        if (
            selection.countdown_active
            or selection.selection_active
            or selection.saving
            or self._ensure_slice_export_controller().active
        ):
            return False
        if self._capture_start_blocked(CaptureOwner.SLICE):
            return False
        source_cache_dir = getattr(self, "cache_dir", None)
        if not source_cache_dir or not os.path.isdir(source_cache_dir):
            self._slice_unavailable("This map has no readable precompiled cache.")
            return False
        try:
            map_slicing.validate_slice_source(source_cache_dir)
            storage_parent = self._slice_storage_directory()
            root_cave_name = self._slice_cave_name()
            display_name = map_slicing.next_slice_display_name(
                storage_parent,
                root_cave_name,
            )
        except Exception as exc:
            self._slice_unavailable(str(exc))
            return False

        color_picker = getattr(self, "color_picker", None)
        if color_picker is not None:
            color_picker.hide()
        controls_overlay = getattr(self, "controls_overlay", None)
        if controls_overlay is not None and controls_overlay.is_manual_mode:
            controls_overlay.hide_help()
        if not selection.start_countdown(
            now=time.perf_counter(),
            start_number=self.SLICE_COUNTDOWN_START_NUMBER,
        ):
            return False
        self._slice_source_cache_dir = str(source_cache_dir)
        self._slice_storage_parent = storage_parent
        self._slice_display_base = display_name
        self._slice_root_cave_name = root_cave_name
        _LOG.info(
            "Slice countdown started. Press %s+C to stop or Escape to cancel.",
            self._primary_shortcut_label(),
        )
        return True

    def _start_slice_export(
        self,
        anchors: SliceAnchors,
        *,
        closing: bool = False,
    ) -> bool:
        """Build and launch a worker request after active slicing ends."""
        source_cache_dir = getattr(self, "_slice_source_cache_dir", None)
        storage_parent = getattr(self, "_slice_storage_parent", None)
        if not source_cache_dir or not storage_parent:
            self._ensure_slice_selection_controller().complete_export()
            self._clear_slice_context()
            self._slice_unavailable("The active slice no longer has a save location.")
            return False
        root_cave_name = (
            getattr(self, "_slice_root_cave_name", None)
            or self._slice_cave_name()
        )
        display_name = map_slicing.sanitize_slice_name(
            getattr(self, "_slice_display_base", None)
            or map_slicing.next_slice_display_name(storage_parent, root_cave_name)
        )
        try:
            request = map_slicing.SliceExportRequest(
                source_cache_dir=source_cache_dir,
                output_dir=map_slicing.unique_slice_output_dir(
                    storage_parent,
                    display_name,
                ),
                bounds=map_slicing.SliceBounds.from_anchors(
                    anchors.start,
                    anchors.end,
                    padding=self.SLICE_PADDING,
                ),
                entry_position=anchors.start,
                display_name=display_name,
                root_cave_name=root_cave_name,
            )
        except Exception as exc:
            self._ensure_slice_selection_controller().complete_export()
            self._clear_slice_context()
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().failed_status(
                    "Slice",
                    str(exc),
                )
            )
            return False

        failure = self._ensure_slice_export_controller().start(request)
        if failure is not None:
            self._ensure_slice_selection_controller().complete_export()
            self._clear_slice_context()
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().failed_status(
                    "Slice",
                    failure.error,
                )
            )
            return False
        if not closing:
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().saving_status(
                    "Slice",
                    cancelable=True,
                )
            )
        _LOG.info("Started slice export to %s", request.output_dir)
        return True

    def _finish_active_slice(self, *, closing: bool = False) -> bool:
        selection = self._ensure_slice_selection_controller()
        position = self._slice_camera_position()
        if position is None:
            if closing:
                selection.cancel_selection()
                self._clear_slice_context()
                self._show_artifact_capture_status(
                    self._ensure_artifact_capture_presentation().failed_status(
                        "Slice",
                        "Could not read the final camera position.",
                    )
                )
                return False
            self._slice_unavailable("Could not read the current camera position.")
            return False
        anchors = selection.finish_selection(position)
        if anchors is None:
            if closing:
                selection.cancel_selection()
                self._clear_slice_context()
                self._show_artifact_capture_status(
                    self._ensure_artifact_capture_presentation().failed_status(
                        "Slice",
                        "Could not finalize the active slice.",
                    )
                )
            return False
        return self._start_slice_export(anchors, closing=closing)

    def _toggle_slice(self) -> bool:
        """Use Ctrl/Cmd+C to start/cancel a countdown or finish an active slice."""
        selection = self._ensure_slice_selection_controller()
        exporter = self._ensure_slice_export_controller()
        if exporter.active or selection.saving:
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().saving_status(
                    "Slice",
                    cancelable=True,
                )
            )
            return True
        if selection.countdown_active:
            selection.cancel_countdown()
            self._clear_slice_context()
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().canceled_status("Slice"),
                now=time.perf_counter(),
            )
            return True
        if selection.selection_active:
            return self._finish_active_slice()
        return self._start_slice_countdown()

    def _cancel_slice_interaction(self) -> bool:
        """Honor Escape for a user-owned slice without canceling close finalization."""
        if self._exit_capture_finalization_active():
            return False
        selection = self._ensure_slice_selection_controller()
        if selection.countdown_active:
            selection.cancel_countdown()
            self._clear_slice_context()
        elif selection.selection_active:
            selection.cancel_selection()
            self._clear_slice_context()
        elif self._ensure_slice_export_controller().active:
            if not self._ensure_slice_export_controller().request_cancel():
                return False
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().canceling_status(
                    "Slice"
                )
            )
            return True
        else:
            return False
        self._show_artifact_capture_status(
            self._ensure_artifact_capture_presentation().canceled_status(
                "Slice",
                after_escape=True,
            ),
            now=time.perf_counter(),
        )
        return True

    def _cancel_active_capture(self) -> bool:
        """Cancel the one recording, trace, or slice lifecycle owning capture."""
        if self._exit_capture_finalization_active():
            return False
        owner = self._capture_owner()
        if owner is CaptureOwner.VIDEO:
            return self._cancel_recording_capture()
        if owner is CaptureOwner.DIVE_TRACE:
            return self._cancel_manual_dive_trace_capture()
        if owner is CaptureOwner.SLICE:
            return self._cancel_slice_interaction()
        return False

    def _update_slice_export(self, *, now: float | None = None) -> None:
        """Advance countdown anchors and apply child export outcomes on the UI thread."""
        current_time = time.perf_counter() if now is None else now
        selection = self._ensure_slice_selection_controller()
        if selection.countdown_ready(now=current_time):
            position = self._slice_camera_position()
            if position is None or not selection.begin_selection(position):
                selection.complete_export()
                self._clear_slice_context()
                self._slice_unavailable("Could not read the current camera position.")

        exporter = self._ensure_slice_export_controller()
        for update in exporter.poll():
            if isinstance(update, SliceExportSucceeded):
                selection.complete_export()
                self._clear_slice_context()
                try:
                    from caveviewer.gui.map_history import remember_recent_map_path

                    remember_recent_map_path(update.output_dir)
                except Exception:
                    pass
                if self._exit_capture_finalization_active():
                    # Reveal only once exit finalization is ready to hand the
                    # window back to the platform; some file managers would
                    # otherwise focus the viewer again before it closes.
                    self._slice_reveal_before_close = True
                    self._slice_reveal_output_path = update.output_dir
                else:
                    self._show_artifact_capture_status(
                        self._ensure_artifact_capture_presentation().saved_status(
                            "Slice",
                            update.output_dir,
                            now=current_time,
                            reveal=True,
                        ),
                        now=current_time,
                    )
            elif isinstance(update, SliceExportCanceled):
                selection.complete_export()
                self._clear_slice_context()
                if not self._exit_capture_finalization_active():
                    self._show_artifact_capture_status(
                        self._ensure_artifact_capture_presentation().canceled_status(
                            "Slice",
                            after_escape=True,
                        ),
                        now=current_time,
                    )
            elif isinstance(update, SliceExportFailed):
                selection.complete_export()
                self._clear_slice_context()
                self._show_artifact_capture_status(
                    self._ensure_artifact_capture_presentation().failed_status(
                        "Slice",
                        update.error,
                    ),
                    now=current_time,
                )

    def _start_manual_dive_trace_countdown(self) -> bool:
        """Arm a visible countdown before collecting a manual dive trace."""
        if (
            not self._has_map_loaded
            or getattr(self, "_manual_dive_trace", None) is not None
        ):
            return False
        if self._capture_start_blocked(CaptureOwner.DIVE_TRACE):
            return False
        controller = self._ensure_manual_dive_trace_controller()
        if controller.countdown_active:
            return False

        color_picker = getattr(self, "color_picker", None)
        if color_picker is not None:
            color_picker.hide()
        controls_overlay = getattr(self, "controls_overlay", None)
        if controls_overlay is not None and controls_overlay.is_manual_mode:
            controls_overlay.hide_help()

        now = time.perf_counter()
        controller.start_countdown(
            now=now,
            start_number=self.MANUAL_DIVE_TRACE_COUNTDOWN_START_NUMBER,
        )
        _LOG.info(
            "Manual Guided Dive trace countdown started. "
            "Press %s+T to stop or Escape to cancel.",
            self._primary_shortcut_label(),
        )
        return True

    def _start_manual_dive_trace(self) -> bool:
        if (
            not self._has_map_loaded
            or getattr(self, "_manual_dive_trace", None) is not None
        ):
            return False
        if self._capture_start_blocked(CaptureOwner.DIVE_TRACE):
            return False
        pose = self._manual_dive_trace_pose()
        if pose is None:
            _LOG.warning("Manual Guided Dive trace could not read the camera pose.")
            return False
        map_root = getattr(self, "map_root", None)
        if not map_root:
            _LOG.warning(
                "Manual Guided Dive trace could not start because the map root is unknown."
            )
            return False
        recorder = manual_dive_trace.ManualDiveTraceRecorder(
            manual_dive_trace.manual_dive_trace_directory(map_root),
            map_context=manual_dive_trace.manual_dive_trace_map_context(
                self.manifest
            ),
        )
        try:
            output_path = recorder.start(pose)
        except Exception as exc:
            _LOG.warning("Manual Guided Dive trace could not start: %s", exc)
            return False
        self._manual_dive_trace = recorder
        _LOG.info(
            "Manual Guided Dive trace started. Press %s+T to stop and save or "
            "Escape to cancel: %s",
            self._primary_shortcut_label(),
            output_path,
        )
        return True

    def _stop_manual_dive_trace(self, *, reason: str) -> bool:
        self._ensure_manual_dive_trace_controller().clear_countdown()
        recorder = getattr(self, "_manual_dive_trace", None)
        if recorder is None:
            return False
        try:
            output_path = recorder.stop(
                self._manual_dive_trace_pose(),
                reason=reason,
            )
        except Exception as exc:
            _LOG.warning("Manual Guided Dive trace could not stop cleanly: %s", exc)
            output_path = recorder.output_path
        self._manual_dive_trace = None
        writers = getattr(self, "_manual_dive_trace_writers", None)
        if writers is None:
            writers = []
            self._manual_dive_trace_writers = writers
        show_completion = reason in {"user_stopped", "writer_failed"}
        writers.append(
            _PendingManualDiveTraceWriter(
                recorder=recorder,
                show_completion=show_completion,
                reveal_on_success=reason == "user_stopped",
            )
        )
        _LOG.info("Manual Guided Dive trace is saving: %s", output_path)
        if show_completion:
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().saving_status(
                    "Dive trace",
                    cancelable=True,
                )
            )
        return True

    def _cancel_manual_dive_trace_capture(self) -> bool:
        """Cancel trace countdown or discard the active/pending trace writer."""
        if self._exit_capture_finalization_active():
            return False
        canceled = False
        cleanup_pending = False
        controller = self._ensure_manual_dive_trace_controller()
        if controller.countdown_active:
            controller.clear_countdown()
            canceled = True

        writers = getattr(self, "_manual_dive_trace_writers", None)
        if writers is None:
            writers = []
            self._manual_dive_trace_writers = writers

        recorder = getattr(self, "_manual_dive_trace", None)
        if recorder is not None:
            try:
                requested = bool(recorder.cancel())
            except Exception as exc:
                _LOG.warning("Manual Guided Dive trace could not cancel: %s", exc)
                requested = False
            if requested:
                self._manual_dive_trace = None
                writers.append(
                    _PendingManualDiveTraceWriter(
                        recorder=recorder,
                        show_completion=True,
                        reveal_on_success=False,
                    )
                )
                canceled = True
                cleanup_pending = True
            else:
                self._show_artifact_capture_status(
                    self._ensure_artifact_capture_presentation().cancellation_failed_status(
                        "Dive trace",
                        "The trace writer could not start cleanup.",
                    ),
                    now=time.perf_counter(),
                )
                # Keep the recorder attached so its resources remain owned and
                # a later Escape or normal stop can retry cleanup safely.
                return True

        for index, pending_writer in enumerate(tuple(writers)):
            if pending_writer.recorder is recorder:
                continue
            try:
                requested = bool(pending_writer.recorder.cancel())
            except Exception as exc:
                _LOG.warning(
                    "Pending Manual Guided Dive trace could not cancel: %s",
                    exc,
                )
                requested = False
            if not requested:
                continue
            writers[index] = _PendingManualDiveTraceWriter(
                recorder=pending_writer.recorder,
                show_completion=True,
                reveal_on_success=False,
            )
            canceled = True
            cleanup_pending = True

        if not canceled:
            return False
        presentation = self._ensure_artifact_capture_presentation()
        self._show_artifact_capture_status(
            (
                presentation.canceling_status("Dive trace")
                if cleanup_pending
                else presentation.canceled_status(
                    "Dive trace",
                    after_escape=True,
                )
            ),
            now=time.perf_counter(),
        )
        _LOG.info("Manual Guided Dive trace canceled with Escape.")
        return True

    def _toggle_manual_dive_trace(self) -> bool:
        if getattr(self, "_manual_dive_trace", None) is not None:
            return self._stop_manual_dive_trace(reason="user_stopped")
        controller = self._ensure_manual_dive_trace_controller()
        if controller.countdown_active:
            controller.clear_countdown()
            now = time.perf_counter()
            self._show_artifact_capture_status(
                self._ensure_artifact_capture_presentation().canceled_status(
                    "Dive trace"
                ),
                now=now,
            )
            _LOG.info("Manual Guided Dive trace countdown canceled.")
            return True
        return self._start_manual_dive_trace_countdown()

    def _update_manual_dive_trace(self, *, now: float | None = None) -> None:
        """Advance the trace countdown and apply finished writer outcomes."""
        now = time.perf_counter() if now is None else now
        controller = self._ensure_manual_dive_trace_controller()
        if controller.countdown_ready(now=now):
            controller.clear_countdown()
            if not self._start_manual_dive_trace():
                self._show_capture_status(
                    "Dive trace unavailable",
                    "Could not start the trace for this map.",
                    kind="error",
                    duration=(
                        self._ensure_artifact_capture_presentation().confirmation_seconds
                    ),
                    now=now,
                )

        recorder = getattr(self, "_manual_dive_trace", None)
        if recorder is not None:
            if recorder.writer_failed:
                self._stop_manual_dive_trace(reason="writer_failed")
            else:
                pose = self._manual_dive_trace_pose()
                if pose is not None:
                    recorder.observe(pose)

        pending = getattr(self, "_manual_dive_trace_writers", [])
        for pending_writer in tuple(pending):
            result = pending_writer.recorder.poll_result()
            if result is None:
                continue
            pending.remove(pending_writer)
            show_completion = (
                pending_writer.show_completion
                and not self._exit_capture_finalization_active()
            )
            if result.canceled:
                if show_completion:
                    if result.error:
                        status = self._ensure_artifact_capture_presentation().cancellation_failed_status(
                            "Dive trace",
                            "The partial trace could not be removed.",
                        )
                    else:
                        status = self._ensure_artifact_capture_presentation().canceled_status(
                            "Dive trace",
                            after_escape=True,
                        )
                    self._show_artifact_capture_status(status, now=now)
                if result.error:
                    _LOG.warning(
                        "Manual Guided Dive trace cancellation cleanup failed: %s",
                        result.error,
                    )
                else:
                    _LOG.info(
                        "Manual Guided Dive trace canceled and partial output removed."
                    )
            elif result.completed:
                _LOG.info("Manual Guided Dive trace saved: %s", result.output_path)
                if show_completion:
                    status = self._ensure_artifact_capture_presentation().saved_status(
                        "Dive trace",
                        result.output_path,
                        now=now,
                        reveal=pending_writer.reveal_on_success,
                    )
                    self._show_artifact_capture_status(status, now=now)
            else:
                _LOG.warning(
                    "Manual Guided Dive trace failed to save: %s (%s)",
                    result.partial_path,
                    result.error or "unknown error",
                )
                if show_completion:
                    self._show_artifact_capture_status(
                        self._ensure_artifact_capture_presentation().failed_status(
                            "Dive trace",
                            result.error or "The trace writer did not finish.",
                        ),
                        now=now,
                    )
        self._drain_due_saved_artifact_reveals(now=now)

    def _mark_manual_dive_trace_discontinuity(
        self,
        before: manual_dive_trace.ManualDivePose | None,
        *,
        reason: str,
    ) -> None:
        recorder = getattr(self, "_manual_dive_trace", None)
        if recorder is None or before is None:
            return
        after = self._manual_dive_trace_pose()
        if after is None:
            return
        recorder.mark_discontinuity(before, after, reason=reason)

    def _save_bookmark_slot(self, slot: int) -> None:
        if not self._has_map_loaded:
            return
        self._bookmarks[slot] = viewer_bookmarks.bookmark_from_camera(
            self.camera.position,
            yaw=self.camera.yaw,
            pitch=self.camera.pitch,
        )
        self._save_bookmarks()
        _LOG.info(f"Saved camera bookmark {slot}.")

    def _recall_bookmark_slot(self, slot: int) -> bool:
        if not self._has_map_loaded:
            return False
        data = self._bookmarks.get(slot)
        if not data:
            _LOG.info(f"Bookmark {slot} is empty.")
            return False

        if self._recorded_dive_is_active():
            self._stop_recorded_dive(reason="bookmark_recall")
        trace_pose_before_recall = self._manual_dive_trace_pose()
        pos = data["position"]
        self.camera.position = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        self.camera.yaw = float(data["yaw"])
        pitch = float(data["pitch"])
        pitch_limit = getattr(self.camera, "_pitch_limit", None)
        if pitch_limit is not None:
            pitch = max(-float(pitch_limit), min(float(pitch_limit), pitch))
        self.camera.pitch = pitch
        self.camera.roll = 0.0  # Reset roll when loading a bookmark
        self._mark_manual_dive_trace_discontinuity(
            trace_pose_before_recall,
            reason="bookmark_recall",
        )

        self.controls_overlay.show_panel()
        _LOG.info(f"Recalled camera bookmark {slot}.")
        return True

    def _delete_bookmark_slot(self, slot: int) -> None:
        if not self._has_map_loaded:
            return
        if slot not in self._bookmarks:
            _LOG.info(f"Bookmark {slot} does not exist; nothing to delete.")
            return
        del self._bookmarks[slot]
        self._save_bookmarks()
        _LOG.info(f"Deleted camera bookmark {slot}.")

    def _handle_bookmark_hotkey(self, key, modifiers: KeyModifiers) -> bool:
        if not self._has_map_loaded:
            return False
        keys = self.wnd.keys
        slot = self._digit_for_key(keys, key)
        if slot is None:
            return False

        # Platform-specific bookmark save modifier (Command on macOS, Control on Windows/Linux).
        # Shift+digit is accepted as a fallback on macOS for backends that don't report Command.
        save_modifier_down = self._bookmark_save_modifier_is_down(modifiers)
        shift_down = self._key_is_down(keys, "LEFT_SHIFT", "RIGHT_SHIFT", "LSHIFT", "RSHIFT")
        ctrl_down = self._control_is_down(modifiers)
        backspace_down = self._key_is_down(
            keys,
            "DELETE", "DEL",
            "FORWARD_DELETE", "FWDDELETE",
        )

        action = viewer_bookmarks.bookmark_hotkey_action(
            slot,
            save_modifier_down=save_modifier_down,
            shift_down=shift_down,
            ctrl_down=ctrl_down,
            backspace_down=backspace_down,
            shift_digit_save_fallback=(
                self._active_presentation_profile()
                .shift_digit_bookmark_save_fallback
            ),
        )
        if action is viewer_bookmarks.BookmarkHotkeyAction.NONE:
            return False
        if action is viewer_bookmarks.BookmarkHotkeyAction.DELETE:
            self._delete_bookmark_slot(slot)
            return True
        if action is viewer_bookmarks.BookmarkHotkeyAction.SAVE:
            self._save_bookmark_slot(slot)
            return True
        self._recall_bookmark_slot(slot)
        return True

    def _handle_fly_speed_hotkey(self, key, modifiers: KeyModifiers) -> bool:
        """Apply one persistent fly-speed step without entering motion state."""
        speed_step = viewer_input.fly_speed_adjustment_step_for_key(
            self.wnd.keys,
            key,
            shift_down=self._shift_is_down(modifiers),
        )
        if speed_step is None:
            return False
        if self._recorded_dive_is_paused():
            return True

        camera = getattr(self, "camera", None)
        if camera is None:
            return True
        camera.adjust_speed(speed_step)
        return True

    def _option_look_active(self) -> bool:
        if not self._active_presentation_profile().option_left_mouse_look_enabled:
            return False
        return self._key_is_down(
            self.wnd.keys,
            "LEFT_ALT", "RIGHT_ALT", "LEFT_OPTION", "RIGHT_OPTION", "LALT", "RALT",
        )

    def _continuous_input_intent(self, dt: float) -> viewer_input.ContinuousInputIntent:
        return viewer_input.continuous_input_intent(
            keys=self.wnd.keys,
            keys_down=self._keys_down,
            dt=dt,
            key_look_pixels_per_second=self._KEY_LOOK_PIXELS_PER_SECOND,
        )

    def _continuous_input_has_navigation_intent(self, dt: float) -> bool:
        intent = self._continuous_input_intent(dt)
        return bool(intent.has_motion or intent.has_look or intent.has_roll)

    def _handle_continuous_input(self, dt: float):
        intent = self._continuous_input_intent(dt)
        if intent.has_motion:
            self._move_camera(
                intent.forward_amount,
                intent.right_amount,
                intent.up_amount,
                dt,
                intent.speed_multiplier,
            )
        if intent.has_look:
            self.camera.look(intent.yaw_delta, intent.pitch_delta)
        if intent.has_roll:
            self.camera.barrel_roll(intent.roll_delta)

    def _handle_paused_recorded_dive_input(self, dt: float) -> None:
        """Permit look-only inspection without turning a paused dive into flight."""
        intent = self._continuous_input_intent(dt)
        if intent.has_look:
            self.camera.look(intent.yaw_delta, intent.pitch_delta)

    def _handle_manual_input_frame(self, dt: float, *, now: float) -> None:
        """Apply manual camera controls for the current frame."""
        del now
        self._handle_continuous_input(dt)

    def on_key_event(self, key, action, modifiers: KeyModifiers):
        # Cocoa may dispatch key callbacks before viewer controls exist or
        # after teardown has started. Input is not actionable in either state.
        if (
            not getattr(self, "_window_setup_complete", False)
            or self._input_is_suppressed()
        ):
            return

        if self.controls_overlay is None:
            return
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            actions = ViewerKeyPressActions(
                window_shortcut=lambda: self._handle_window_shortcut(key, modifiers),
                recorded_dive=lambda: self._handle_recorded_dive_hotkey(
                    key, modifiers
                ),
                begin_screen=lambda: self._handle_begin_screen_hotkey(key),
                capture_escape=lambda: self._handle_capture_escape_hotkey(key),
                fly_speed=lambda: self._handle_fly_speed_hotkey(key, modifiers),
                bookmark=lambda: self._handle_bookmark_hotkey(key, modifiers),
                manual_dive_trace=lambda: self._handle_manual_dive_trace_hotkey(
                    key, modifiers
                ),
                slice=lambda: self._handle_slice_hotkey(key, modifiers),
                recording=lambda: self._handle_recording_hotkey(key, modifiers),
                reset_view=lambda: self._handle_reset_view_shortcut(key, modifiers),
            )
            workflows = self.__dict__.get("_workflow_coordinator")
            action_handled = (
                workflows.dispatch_key_press(actions)
                if workflows is not None
                else self._ensure_action_dispatcher().dispatch_key_press(actions)
            )
            if action_handled:
                return
            self._keys_down.add(key)
        elif viewer_input.key_event_is_press_or_repeat(keys, action):
            repeat_args = {
                "waiting_for_begin": self.controls_overlay.is_waiting_for_begin,
                "fly_speed": lambda: self._handle_fly_speed_hotkey(key, modifiers),
            }
            workflows = self.__dict__.get("_workflow_coordinator")
            if workflows is not None:
                workflows.dispatch_key_repeat(**repeat_args)
            else:
                self._ensure_action_dispatcher().dispatch_key_repeat(**repeat_args)
        elif action == keys.ACTION_RELEASE:
            self._keys_down.discard(key)

    key_event = on_key_event

    def _primary_shortcut_is_down(self, modifiers: KeyModifiers) -> bool:
        """Return whether the platform-native application modifier is active."""
        if self._active_presentation_profile().tk_primary_modifier_name == "Command":
            return self._command_is_down(modifiers)
        return self._control_is_down(modifiers)

    def _primary_shortcut_label(self) -> str:
        """Return the platform-native label for an application shortcut."""
        return self._active_presentation_profile().primary_shortcut_modifier_label

    def _handle_window_shortcut(self, key, modifiers: KeyModifiers) -> bool:
        """Handle modifier-based import-pause and open-map shortcuts."""
        if not self._primary_shortcut_is_down(modifiers):
            return False

        pause_key = self._resolve_key_optional(self.wnd.keys, "P")
        if (
            pause_key is not None
            and key == pause_key
            and self._shift_is_down(modifiers)
        ):
            if self._import_active:
                self._request_import_pause()
                return True

        open_key = self._resolve_key_optional(self.wnd.keys, "O")
        if open_key is not None and key == open_key:
            if self._has_map_loaded and not self._import_active:
                self._handle_open_button_click()
            return True

        return False

    def _request_import_pause(self) -> None:
        self._ensure_import_controller().request_pause()

    def _handle_begin_screen_hotkey(self, key) -> bool:
        """Keep the introductory overlay's single-key input boundary intact."""
        if not self.controls_overlay.is_waiting_for_begin:
            return False
        space_key = self._resolve_key_optional(
            self.wnd.keys,
            "SPACE",
            "SPACEBAR",
        )
        if (
            space_key is not None
            and key == space_key
            and self.controls_overlay.is_ready_to_begin
        ):
            self.controls_overlay.dismiss_begin_screen()
        return True

    def _handle_manual_dive_trace_hotkey(
        self,
        key,
        modifiers: KeyModifiers,
    ) -> bool:
        """Use Ctrl/Cmd+T to start or stop a map-local manual route trace."""
        if not self._has_map_loaded:
            return False
        trace_key = self._resolve_key_optional(self.wnd.keys, "T")
        if trace_key is None or key != trace_key:
            return False
        if not self._primary_shortcut_is_down(modifiers):
            return False
        if self._capture_shortcut_is_ignored(CaptureOwner.DIVE_TRACE):
            return True
        self._toggle_manual_dive_trace()
        return True

    def _handle_slice_hotkey(self, key, modifiers: KeyModifiers) -> bool:
        """Use Ctrl/Cmd+C to arm, cancel, or finish a portable cave slice."""
        if not getattr(self, "_has_map_loaded", False):
            return False
        slice_key = self._resolve_key_optional(self.wnd.keys, "C")
        if (
            slice_key is None
            or key != slice_key
            or not self._primary_shortcut_is_down(modifiers)
        ):
            return False
        if self._capture_shortcut_is_ignored(CaptureOwner.SLICE):
            return True
        return self._toggle_slice()

    def _handle_capture_escape_hotkey(self, key) -> bool:
        """Own Escape so capture cancellation precedes delayed viewer close."""
        escape_key = self._resolve_key_optional(self.wnd.keys, "ESCAPE", "ESC")
        if escape_key is None or key != escape_key:
            return False
        if self._capture_owner() is not None:
            return self._begin_escape_capture_cancellation()
        self.on_close()
        return True

    def _handle_recorded_dive_hotkey(
        self,
        key,
        modifiers: KeyModifiers,
    ) -> bool:
        """Use Space to pause or resume an opened Recorded Dive."""
        del modifiers
        if not self._recorded_dive_is_active():
            return False
        pause_key = self._resolve_key_optional(self.wnd.keys, "SPACE", "SPACEBAR")
        if pause_key is None or key != pause_key:
            return False
        return self._toggle_recorded_dive_pause()

    def _handle_recording_hotkey(self, key, modifiers: KeyModifiers) -> bool:
        """Use Ctrl/Cmd+R to start, cancel, or stop recording."""
        if not self._has_map_loaded:
            return False
        record_key = self._resolve_key_optional(self.wnd.keys, "R")
        if record_key is None or key != record_key:
            return False
        if not self._primary_shortcut_is_down(modifiers):
            return False
        if self._capture_shortcut_is_ignored(CaptureOwner.VIDEO):
            return True
        self._toggle_recording()
        return True

    def _handle_reset_view_shortcut(self, key, modifiers: KeyModifiers) -> bool:
        """Handle CMD+0 (macOS) or CTRL+0 (Windows/Linux) to reset view."""
        keys = self.wnd.keys

        # Check if this is the 0 key
        if not self._is_zero_key(keys, key):
            return False

        if self._primary_shortcut_is_down(modifiers):
            if self._recorded_dive_is_active():
                self._stop_recorded_dive(reason="view_reset")
            self.camera.reset_view()
            return True

        return False

    def _request_startup_focus_once(self) -> None:
        """Attempt to bring the app window to foreground once after startup."""
        if self._startup_focus_requested:
            return
        self._startup_focus_requested = True

        self._active_presentation_actions_adapter().focus_viewer_window(self.wnd)

    def _reset_transient_input_state(self, reason: str) -> None:
        """Clear transient input/capture flags that can get stuck across sleep/focus changes."""
        self._keys_down.clear()
        self._mouse_look_active = False
        self._mouse_look_left_option_active = False
        self._last_mouse_pos = None
        self.color_picker.on_mouse_release()
        if hasattr(self.wnd, "mouse_exclusivity"):
            self.wnd.mouse_exclusivity = False

        now = time.time()
        if now - self._last_input_reset_log > 3.0:
            _LOG.info(f"Input state reset ({reason}).")
            self._last_input_reset_log = now

    def _query_runtime_iconified_state(self) -> bool:
        """Best-effort minimized/backgrounded detection across window backends."""
        for target in (getattr(self.wnd, "_window", None), self.wnd):
            if target is None:
                continue
            for attr in ("minimized", "is_minimized", "iconified"):
                try:
                    if hasattr(target, attr) and bool(getattr(target, attr)):
                        return True
                except Exception:
                    pass
            for attr in ("visible", "is_visible"):
                try:
                    if hasattr(target, attr):
                        value = getattr(target, attr)
                        value = value() if callable(value) else value
                        if value is False:
                            return True
                except Exception:
                    pass
        return False

    def _set_background_pause(self, should_pause: bool, reason: str) -> None:
        self._is_iconified = bool(should_pause)
        if self._is_background_paused == self._is_iconified:
            return

        self._is_background_paused = self._is_iconified
        if self._is_background_paused:
            self._reset_transient_input_state(reason)
            controller = getattr(self, "_recorded_dive_controller", None)
            if (
                controller is not None
                and controller.active
                and controller.state
                is not recorded_dive.RecordedDivePlaybackState.PAUSED
            ):
                self._recorded_dive_background_paused = controller.pause(
                    now=time.perf_counter()
                )
            if self._has_map_loaded and hasattr(self, "world"):
                self.world.pause()
        else:
            if self._has_map_loaded and hasattr(self, "world"):
                self.world.resume()
            controller = getattr(self, "_recorded_dive_controller", None)
            if (
                getattr(self, "_recorded_dive_background_paused", False)
                and controller is not None
                and controller.state
                is recorded_dive.RecordedDivePlaybackState.PAUSED
            ):
                controller.resume(self.camera, now=time.perf_counter())
            self._recorded_dive_background_paused = False

    def on_focus_event(self, focused: bool):
        # On focus loss/gain, clear transient pressed/captured state so a
        # missed release event cannot leave controls unresponsive.
        self._reset_transient_input_state("focus change")

        # Fallback for platforms where iconify callback isn't reliable:
        # if focus is lost, pause; if focus returns and window is not
        # actually minimized, resume.
        if not focused:
            self._set_background_pause(True, "focus lost")
        else:
            self._set_background_pause(self._query_runtime_iconified_state(), "focus gained")

    focus_event = on_focus_event

    def on_iconify_event(self, iconified: bool):
        # Minimize/restore paths can behave similarly to focus changes.
        self._set_background_pause(bool(iconified), "window iconified")

    iconify_event = on_iconify_event

    def _handle_mouse_look_motion(self, x, y, dx, dy):
        # Cocoa can deliver passive mouse-move callbacks while the native
        # window exists but before our Python-side controls are fully built.
        # Treat those early/late events as no-ops so ctypes does not print
        # ignored callback exceptions to stderr.
        if (
            not getattr(self, "_window_setup_complete", False)
            or self._input_is_suppressed()
        ):
            return

        # Color picker's RGB sliders still use continuous drag (a
        # separate feature from the brightness/render-distance controls
        # below, which were converted to discrete +/- steppers) -- this
        # still needs to take priority over camera look while one of its
        # sliders is being dragged, same reasoning as before.
        color_picker = getattr(self, "color_picker", None)
        if color_picker is not None and color_picker.is_dragging:
            color_picker.on_mouse_drag(x, y, self.wnd.size)
            return
        # macOS-friendly fallback: Option + pointer movement can drive
        # look even without a physical click/drag gesture.
        if self._option_look_active() or self._mouse_look_active:
            # On the first event after mouse exclusivity is enabled the
            # backend warps the cursor to the window centre, generating a
            # large spurious delta.  _last_mouse_pos being None is the
            # sentinel for "just activated": absorb that one event and
            # record a real position so subsequent deltas are applied.
            if self._last_mouse_pos is None:
                self._last_mouse_pos = (x, y)
                return
            self._last_mouse_pos = (x, y)
            if (
                self._recorded_dive_is_active()
                and not self._recorded_dive_is_paused()
            ):
                self._stop_recorded_dive(reason="mouse_look")
            self.camera.look(dx, dy)

    def on_mouse_position_event(self, x, y, dx, dy):
        self._handle_mouse_look_motion(x, y, dx, dy)

    mouse_position_event = on_mouse_position_event

    def on_mouse_drag_event(self, x, y, dx, dy):
        # Win32 can dispatch a drag before the native window's Python-side
        # controls have completed construction. Do not dereference the overlay
        # until initialization has established it.
        if (
            not getattr(self, "_window_setup_complete", False)
            or self._input_is_suppressed()
        ):
            return
        if self.controls_overlay.is_waiting_for_begin:
            return
        self._handle_mouse_look_motion(x, y, dx, dy)

    mouse_drag_event = on_mouse_drag_event

    def on_mouse_press_event(self, x, y, button):
        # A press can arrive through the same native callback path before
        # `controls_overlay` is available; treat it as non-actionable.
        if (
            not getattr(self, "_window_setup_complete", False)
            or self._input_is_suppressed()
        ):
            return
        if self.controls_overlay.is_waiting_for_begin:
            return
        if self.controls_overlay.is_manual_mode:
            self.controls_overlay.hide_help()
            return

        look_button_name = self._active_presentation_profile().mouse_look_button_name
        look_button = self.wnd.mouse.left if look_button_name == "left" else self.wnd.mouse.right

        if self._recording_hides_hud():
            if button == self.wnd.mouse.left and self._option_look_active():
                self._mouse_look_active = True
                self._mouse_look_left_option_active = True
                self._last_mouse_pos = None
                self.wnd.mouse_exclusivity = True
                return
            if button == look_button:
                self._mouse_look_active = True
                self._last_mouse_pos = None
                self.wnd.mouse_exclusivity = True
            return

        if button == self.wnd.mouse.left:
            # macOS-friendly mouse-look: Option + left-drag avoids relying
            # on right-click behavior (which can vary across trackpads/mice).
            if self._option_look_active():
                self._mouse_look_active = True
                self._mouse_look_left_option_active = True
                self._last_mouse_pos = None
                self.wnd.mouse_exclusivity = True
                return

            # Check order: all three steppers, then mesh/texture toggle
            # buttons, then minimap. All four pieces (brightness, global
            # light, render distance, button block) now live together in
            # the same bottom-right column -- check order only matters in
            # the sense that each needs to happen before falling through
            # to the next, since their hit areas don't overlap.
            column = self._right_column_layout(self.wnd.size)
            brightness_anchor_x, brightness_anchor_y = column["brightness_anchor"]
            ambient_anchor_x, ambient_anchor_y = column["ambient_anchor"]
            render_distance_anchor_x, render_distance_anchor_y = column["render_distance_anchor"]
            buttons_top_y = column["buttons_top_y"]

            # While map-loading overlays are active (startup fullscreen or
            # teleport panel), keep the right-side button block inert.
            # Manual HELP mode is intentionally excluded so the same
            # buttons remain usable when the user explicitly opens help.
            buttons_locked_for_loading = self._buttons_locked_for_loading()

            if self.light_stepper.on_mouse_press(x, y, brightness_anchor_x, brightness_anchor_y):
                return

            if self.ambient_stepper.on_mouse_press(x, y, ambient_anchor_x, ambient_anchor_y):
                return

            if self.render_distance_stepper.on_mouse_press(x, y, render_distance_anchor_x, render_distance_anchor_y):
                return

            if buttons_locked_for_loading:
                if (
                    self.render_mode_buttons.hit_test_mesh(x, y, self.wnd.size, buttons_top_y, column["button_right_inset"])
                    or self.render_mode_buttons.hit_test_texture(x, y, self.wnd.size, buttons_top_y, column["button_right_inset"])
                    or self.render_mode_buttons.hit_test_shade(x, y, self.wnd.size, buttons_top_y, column["button_right_inset"])
                    or self.render_mode_buttons.hit_test_help(x, y, self.wnd.size, buttons_top_y, column["button_right_inset"])
                    or self.render_mode_buttons.hit_test_color(x, y, self.wnd.size, buttons_top_y, column["button_right_inset"])
                    or self.render_mode_buttons.hit_test_open(x, y, self.wnd.size, buttons_top_y, column["button_right_inset"])
                ):
                    return

            clicked_button = self.render_mode_buttons.on_mouse_press(
                x, y, self.wnd.size, buttons_top_y, column["button_right_inset"]
            )
            if clicked_button == "shade":
                self._apply_shading_toggle()
                return
            elif clicked_button == "help":
                # Toggle: if the help screen is already showing (manual
                # mode), a second click closes it; otherwise show it.
                # Showing help intentionally overrides whatever loading
                # overlay might currently be active (e.g. a brief teleport
                # panel) -- an explicit click is a clear request to see
                # the controls right now, which should win over a
                # transient loading indicator.
                if self.controls_overlay.is_manual_mode:
                    self.controls_overlay.hide_help()
                else:
                    self.controls_overlay.show_help()
                return
            elif clicked_button == "color":
                if self.color_picker.is_active:
                    self.color_picker.hide()
                else:
                    self.color_picker.show()
                return
            elif clicked_button == "open":
                self._handle_open_button_click()
                return
            elif clicked_button is not None:
                # "mesh" or "texture" -- already toggled internally by
                # render_mode_buttons.on_mouse_press, nothing further needed here.
                return

            # While the color picker panel is open, it behaves like a
            # modal -- clicks inside the panel interact with its sliders.
            # A click outside closes the picker and is consumed so that
            # dismissing it cannot also trigger unrelated world/UI actions
            # underneath on the same click.
            if self.color_picker.is_active:
                if self.color_picker.hit_test_panel(x, y, self.wnd.size):
                    self.color_picker.on_mouse_press(x, y, self.wnd.size)
                else:
                    self.color_picker.hide()
                return

            minimap_target = None
            if self._has_map_loaded and self.minimap is not None:
                minimap_target = self.minimap.world_xz_for_click(x, y, self.wnd.size)
            if minimap_target is not None:
                if self._recorded_dive_is_active():
                    self._stop_recorded_dive(reason="minimap_teleport")
                target_x, target_z = minimap_target
                trace_pose_before_teleport = self._manual_dive_trace_pose()
                # Land at an actual occupied height near that X/Z, rather
                # than blindly keeping the camera's previous Y -- a click
                # on the (top-down, height-blind) minimap doesn't tell us
                # which vertical level was meant, so we look up real chunk
                # bounds at that column and pick whichever level is
                # closest to the camera's current height (see
                # find_landing_position in caveviewer.core.chunking.metadata).
                # This is what prevents landing above or below the actual
                # passage.
                old_x = float(self.camera.position[0])
                old_z = float(self.camera.position[2])
                landing_x, landing_y, landing_z = chunker.find_landing_position(
                    self.manifest, target_x, target_z,
                    preferred_y=float(self.camera.position[1]),
                )
                self.camera.position[0] = landing_x
                self.camera.position[1] = landing_y
                self.camera.position[2] = landing_z

                # Reorient toward the teleport direction so the camera looks
                # into the new area rather than potentially facing blank space.
                # Only rotate when the click is far enough away to give a
                # meaningful direction (>0.5 m threshold avoids jitter for
                # near-by clicks that don't imply a clear travel direction).
                dx = landing_x - old_x
                dz = landing_z - old_z
                if math.hypot(dx, dz) > 0.5:
                    self.camera.yaw   = math.atan2(dz, dx)
                    self.camera.pitch = 0.0
                    self.camera.roll  = 0.0
                self._mark_manual_dive_trace_discontinuity(
                    trace_pose_before_teleport,
                    reason="minimap_teleport",
                )

                # Show the controls panel briefly while the newly-teleported
                # area's chunks stream in around the camera -- same content
                # as the full-screen startup overlay, just smaller since
                # teleporting is quick and shouldn't block the whole view.
                self.controls_overlay.show_panel()
                return

            # On Windows/Linux, left-click that doesn't hit any UI activates mouse look
            if look_button_name == "left":
                self._mouse_look_active = True
                self._last_mouse_pos = None
                self.wnd.mouse_exclusivity = True
            return
        if button == look_button and look_button_name == "right":
            self._mouse_look_active = True
            self._last_mouse_pos = None
            self.wnd.mouse_exclusivity = True

    mouse_press_event = on_mouse_press_event

    def on_mouse_release_event(self, x, y, button):
        # Win32 may send a release before construction initializes the mouse
        # state. Match the other mouse entry points and ignore it safely.
        if (
            not getattr(self, "_window_setup_complete", False)
            or self._input_is_suppressed()
        ):
            return
        look_button_name = self._active_presentation_profile().mouse_look_button_name
        look_button = self.wnd.mouse.left if look_button_name == "left" else self.wnd.mouse.right

        if button == self.wnd.mouse.left:
            if self._mouse_look_left_option_active:
                self._mouse_look_left_option_active = False
                self._mouse_look_active = False
                self.wnd.mouse_exclusivity = False
                return
            # On Windows/Linux, left-click release ends mouse look
            if self._mouse_look_active and look_button_name == "left":
                self._mouse_look_active = False
                self.wnd.mouse_exclusivity = False
                return
            self.color_picker.on_mouse_release()
            return
        if button == look_button and look_button_name == "right":
            self._mouse_look_active = False
            self.wnd.mouse_exclusivity = False

    mouse_release_event = on_mouse_release_event

    def on_mouse_scroll_event(self, x_offset, y_offset):
        if self._input_is_suppressed():
            return
        if self._recorded_dive_is_paused():
            return
        camera = getattr(self, "camera", None)
        if camera is None:
            return
        camera.adjust_speed(y_offset)

    mouse_scroll_event = on_mouse_scroll_event

    def _cancel_active_import(self) -> None:
        self._ensure_import_controller().cancel_active_import()

    def _shutdown_active_import(self) -> None:
        self._ensure_import_controller().shutdown()

    def _hide_window_before_close(self) -> None:
        """Remove the native viewer before releasing its visible GL surface."""
        window = getattr(self, "wnd", None)
        if window is None:
            return
        try:
            window.visible = False
        except Exception:
            # Backends without a visibility property still retain the regular
            # close path below; this is a presentation-only improvement.
            pass

    def _complete_window_close(self) -> None:
        """Release viewer resources after any active capture has finished."""
        workflows = self.__dict__.get("_workflow_coordinator")
        if workflows is not None:
            if not workflows.begin_shutdown():
                return
        elif self._closing_requested:
            return
        self._closing_requested = True
        if workflows is None:
            self._ensure_capture_workflow().complete_close_workflows()
        try:
            self._slice_reveal_before_close = False
            self._slice_reveal_output_path = None

            if hasattr(self, "wnd"):
                try:
                    self.wnd.mouse_exclusivity = False
                except Exception:
                    pass

            if getattr(self, "_import_active", False):
                self._shutdown_active_import()

            self._finish_benchmark(reason="viewer_closed")

            if self._has_map_loaded:
                self._teardown_current_map(final_shutdown=True)
            self._release_window_resources()

            # Ensure the backend window loop receives an explicit close request.
            if hasattr(self, "wnd") and hasattr(self.wnd, "close"):
                try:
                    self.wnd.close()
                except Exception:
                    pass
        finally:
            if workflows is not None:
                workflows.complete_shutdown()

    def on_close(self):
        if self._closing_requested:
            return
        if self._ensure_import_controller().request_pause_for_close():
            self._defer_backend_close_request()
            return
        if self._escape_capture_cancellation_active():
            # Escape owns this shutdown request until discard cleanup and the
            # three-second no-save confirmation have both completed.
            self._defer_backend_close_request()
            return
        if self._exit_capture_finalization_active():
            # Repeated close requests must not tear down the OpenGL context
            # while the non-daemon writer is publishing the user's file.
            self._defer_backend_close_request()
            return

        slice_selection = self._ensure_slice_selection_controller()
        if slice_selection.countdown_active:
            # Match recording/trace behavior: an armed countdown has not
            # captured a user artifact, so closing simply disarms it.
            slice_selection.cancel_countdown()
            self._clear_slice_context()

        artifact_names = self._exit_capture_artifacts()
        if artifact_names:
            self._begin_exit_capture_finalization(artifact_names)
            return

        self._complete_window_close()

    close = on_close


def _run_moderngl_window_config(config_class: type, args=None) -> None:
    """
    Run moderngl-window while preserving CaveViewer's normal shutdown path
    when the blocking render loop is interrupted by Ctrl+C/SIGINT.

    moderngl-window destroys the backend window only after its loop exits
    normally.  A KeyboardInterrupt can arrive inside any render callback and
    bypass that tail cleanup, so create the config explicitly and close/destroy
    the window ourselves before re-raising to the application boundary.
    """
    config_class_name = getattr(config_class, "__name__", str(config_class))
    record_runtime_stage(
        "viewer_window_config_create_begin",
        config_class=config_class_name,
    )
    try:
        config = mglw.create_window_config_instance(config_class, args=args)
    except BaseException as error:
        record_runtime_exception(
            "viewer_window_config_create_failed",
            error,
            config_class=config_class_name,
        )
        raise
    record_runtime_stage(
        "viewer_window_config_created",
        config_class=config_class_name,
        window_backend=type(getattr(config, "wnd", None)).__name__,
    )
    window_destroyed_by_runner = False
    try:
        record_runtime_stage("viewer_window_loop_begin")
        mglw.run_window_config_instance(config)
        window_destroyed_by_runner = True
        record_runtime_stage("viewer_window_loop_returned")
    except BaseException as error:
        record_runtime_exception("viewer_window_loop_exception", error)
        wnd = getattr(config, "wnd", None)
        if wnd is not None:
            try:
                if not getattr(wnd, "is_closing", False):
                    wnd.close()
            except Exception:
                _LOG.exception("Error while closing viewer after interrupted window loop.")
        raise
    finally:
        record_runtime_stage(
            "viewer_window_cleanup_begin",
            loop_returned=window_destroyed_by_runner,
        )
        if not window_destroyed_by_runner:
            wnd = getattr(config, "wnd", None)
            if wnd is not None:
                try:
                    wnd.destroy()
                except Exception:
                    pass
        record_runtime_stage("viewer_window_cleanup_complete")


def _session_window_config_class(
    session: ViewerSession,
    *,
    window_size: tuple[int, int],
) -> type[CaveViewerWindow]:
    """Bind one immutable session to the class-based ModernGL launch API."""

    return type(
        "CaveViewerSessionWindow",
        (CaveViewerWindow,),
        {
            "__module__": __name__,
            "_viewer_session": session,
            "window_size": window_size,
            "vsync": session.config.vsync,
        },
    )


def _launch_viewer_window(
    session: ViewerSession,
    *,
    window_size_override: tuple[int, int] | None = None,
) -> None:
    """Launch with dimensions expressed in the selected backend's coordinates."""
    platform_runtime = session.config.platform_runtime
    record_runtime_stage("viewer_launch_preflight_begin")
    try:
        preflight = viewer_launch_preflight(
            platform_runtime=platform_runtime,
        )
        target = authorized_viewer_launch_target(preflight)
    except BaseException as error:
        record_runtime_exception("viewer_launch_preflight_failed", error)
        raise
    record_runtime_stage(
        "viewer_launch_target_authorized",
        route=target.route_key,
    )
    if window_size_override is not None:
        requested_window_size = window_size_override
        window_size_fraction = None
        fallback_window_size = window_size_override
    elif _presentation_profile_for_runtime(
        platform_runtime
    ).viewer_uses_glfw_native_initial_size:
        # Linux GLFW sizing happens after the Wayland/X11 backend is selected,
        # using that backend's DPI-aware work-area coordinate system.
        requested_window_size = _DEFAULT_WINDOW_SIZE
        window_size_fraction = _DESKTOP_WINDOW_SCALE
        fallback_window_size = _DEFAULT_WINDOW_SIZE
    else:
        requested_window_size = _desktop_relative_window_size()
        window_size_fraction = _DESKTOP_WINDOW_SCALE
        fallback_window_size = _DEFAULT_WINDOW_SIZE
    config_class = _session_window_config_class(
        session,
        window_size=requested_window_size,
    )
    request = ViewerWindowLaunchRequest(
        config_class=config_class,
        runner=_run_moderngl_window_config,
        window_size_fraction=window_size_fraction,
        fallback_window_size=fallback_window_size,
        force_resizable_window=True,
    )
    record_runtime_stage(
        "viewer_native_launch_begin",
        requested_window_size=requested_window_size,
        window_size_fraction=window_size_fraction,
    )
    try:
        _window_backend_adapter_for_runtime(
            platform_runtime
        ).launch_viewer(
            target,
            request,
        )
    except BaseException as error:
        record_runtime_exception("viewer_native_launch_failed", error)
        raise
    record_runtime_stage("viewer_native_launch_returned")


def _normalize_map_root(
    map_root: str | os.PathLike[str] | None,
) -> str | None:
    """Return an absolute map root, or ``None`` when launch context lacks one."""
    if map_root is None:
        return None
    raw_map_root = os.fspath(map_root).strip()
    if not raw_map_root:
        return None
    return os.path.abspath(os.path.expanduser(raw_map_root))


def run_viewer(
    cache_dir: str,
    textures_dir: str,
    recorded_dive_trace: recorded_dive.RecordedDiveTrace | None = None,
    platform_runtime: PlatformRuntime | None = None,
    runtime_settings: RuntimeSettings | None = None,
    map_root: str | os.PathLike[str] | None = None,
):
    manifest = chunker.load_manifest(cache_dir)
    session = ViewerSession(
        ViewerSessionConfig(
            mode=ViewerLaunchMode.READY_CACHE,
            cache_dir=cache_dir,
            textures_dir=textures_dir,
            map_root=_normalize_map_root(map_root),
            manifest=manifest,
            recorded_dive_trace=recorded_dive_trace,
            platform_runtime=platform_runtime,
            runtime_settings=runtime_settings,
            vsync=(
                runtime_settings.viewer_configuration().vsync
                if runtime_settings is not None
                else _env_bool("CAVEVIEWER_VSYNC", True)
            ),
        )
    )

    try:
        _launch_viewer_window(session)
    finally:
        bitmap_font.clear_runtime_style()


def _cache_manifest_sha256(cache_dir: str) -> str:
    manifest_path = os.path.join(cache_dir, chunker.MANIFEST_NAME)
    digest = hashlib.sha256()
    with open(manifest_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_viewer_benchmark(
    cache_dir: str,
    textures_dir: str,
    scenario,
    output_dir: str,
    *,
    runtime_settings: RuntimeSettings | None = None,
):
    """Run a deterministic viewer benchmark against an existing chunk cache."""
    import platform as _platform

    summary_path = os.path.join(output_dir, "summary.json")
    manifest = chunker.load_manifest(cache_dir)
    streaming_settings = _benchmark_streaming_settings_snapshot(
        scenario,
        runtime_settings=runtime_settings,
    )
    streaming_fingerprint = _benchmark_streaming_settings_fingerprint(
        streaming_settings
    )
    viewer_settings = (
        runtime_settings.viewer_configuration()
        if runtime_settings is not None
        else None
    )
    benchmark_platform_runtime = None
    if runtime_settings is not None:
        from caveviewer.gui.platform.runtime import create_platform_runtime

        benchmark_platform_runtime = create_platform_runtime(
            runtime_settings=runtime_settings
        )
    benchmark_config = ViewerBenchmarkConfig(
        scenario=scenario,
        output_dir=output_dir,
        environment={
            "app_version": APP_VERSION,
            "python": sys.version.split()[0],
            "platform": _platform.platform(),
            "cache_dir": os.path.abspath(cache_dir),
            "textures_dir": os.path.abspath(textures_dir),
            "cache_manifest_sha256": _cache_manifest_sha256(cache_dir),
            "scenario": scenario.name,
            "scenario_fingerprint": scenario.fingerprint,
            "source_sha": os.environ.get("GITHUB_SHA")
            or (
                viewer_settings.commit_identifier
                if viewer_settings is not None
                else os.environ.get("CAVEVIEWER_COMMIT", "")
            ),
            "vsync_env": (
                str(viewer_settings.vsync).lower()
                if viewer_settings is not None
                else os.environ.get("CAVEVIEWER_VSYNC", "")
            ),
            "streaming_settings": streaming_settings,
            "streaming_settings_fingerprint": streaming_fingerprint,
            "render_distance_chunks": streaming_settings["render_distance_chunks"],
            "memory_target_percent": streaming_settings["system_ram_target_percent"],
            "gpu_memory_target_percent": streaming_settings[
                "gpu_memory_target_percent"
            ],
            "gpu_memory_override_gb": streaming_settings["gpu_memory_override_gb"],
            "io_workers": streaming_settings["io_workers"],
            "io_reserved_cpus": streaming_settings["io_reserved_cpus"],
            "upload_chunks_per_frame": streaming_settings[
                "upload_chunks_per_frame"
            ],
            "upload_groups_per_frame": streaming_settings[
                "upload_groups_per_frame"
            ],
            "upload_time_budget_ms": streaming_settings["upload_time_budget_ms"],
        },
    )
    session = ViewerSession(
        ViewerSessionConfig(
            mode=ViewerLaunchMode.BENCHMARK,
            cache_dir=cache_dir,
            textures_dir=textures_dir,
            manifest=manifest,
            benchmark=benchmark_config,
            platform_runtime=benchmark_platform_runtime,
            runtime_settings=runtime_settings,
            vsync=(
                viewer_settings.vsync
                if viewer_settings is not None
                else _env_bool("CAVEVIEWER_VSYNC", True)
            ),
        )
    )

    try:
        _launch_viewer_window(session)
        return summary_path
    finally:
        bitmap_font.clear_runtime_style()


def run_viewer_with_pending_import(
    model_descriptor: dict,
    textures_dir: str,
    recorded_dive_trace: recorded_dive.RecordedDiveTrace | None = None,
    platform_runtime: PlatformRuntime | None = None,
    runtime_settings: RuntimeSettings | None = None,
):
    """
    Launches the viewer window for a map that needs FIRST-TIME import
    (no generated cache yet) -- used by caveviewer.app's main() instead
    of run_viewer() specifically so the import can run AFTER the window
    is open, showing real progress in the same in-window panel the OPEN
    button already uses, rather than the old behavior of running the
    import entirely before any window existed (which could only show a
    plain console progress bar, with nowhere graphical to draw into yet).

    model_descriptor is whatever caveviewer.app's find_model_file()
    returned -- a small dict identifying which format (.obj, .glb)
    and the relevant file path(s), format-agnostic so this single
    function/code path covers every supported source format rather than
    needing a separate pending-import entry point per format.

    The window opens immediately with no map loaded; the actual import
    is triggered from inside CaveViewerWindow.on_render()'s first frame
    (see _run_pending_import) once the window is confirmed to have
    rendered and is genuinely on screen.
    """
    session = ViewerSession(
        ViewerSessionConfig(
            mode=ViewerLaunchMode.PENDING_IMPORT,
            pending_import=PendingImportRequest(
                model_descriptor=model_descriptor,
                textures_dir=textures_dir,
            ),
            recorded_dive_trace=recorded_dive_trace,
            platform_runtime=platform_runtime,
            runtime_settings=runtime_settings,
            vsync=(
                runtime_settings.viewer_configuration().vsync
                if runtime_settings is not None
                else _env_bool("CAVEVIEWER_VSYNC", True)
            ),
        )
    )

    try:
        _launch_viewer_window(session)
    except BaseException as error:
        outcome = session.outcome
        # Some native backends surface a programmatic window close as
        # SystemExit. Suppress it only after the import controller has recorded
        # the recoverable startup failure that requested that close.
        if isinstance(error, SystemExit) and outcome.kind == "import_failed":
            _LOG.info("Viewer returned to the library after startup import failure.")
            return outcome
        # Suppress the known "no initial map" runtime error that can occur
        # when the viewer is launched without a preloaded map and the GUI
        # is closed; let other RuntimeErrors propagate.
        msg = str(error)
        if isinstance(error, RuntimeError) and (
            "viewer session has neither a ready cache" in msg.lower()
        ):
            # Clean exit without a traceback
            _LOG.info("Viewer exited without a preloaded map.")
            return outcome
        raise
    else:
        return session.outcome
    finally:
        bitmap_font.clear_runtime_style()
