"""Session-scoped, non-OpenGL workflow composition for the native viewer.

``CaveViewerWindow`` adapts backend and render-thread facts into immutable
snapshots.  This module owns the controller graph and the deterministic
cross-workflow decisions made from those facts; it never imports or mutates
OpenGL resources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from caveviewer.benchmarking.results import BenchmarkController
from caveviewer.gui.artifact_capture_controller import (
    ArtifactCapturePresentationController,
)
from caveviewer.gui.import_controller import MapImportController
from caveviewer.gui.manual_dive_trace_controller import (
    ManualDiveTraceStateController,
)
from caveviewer.gui.map_opening_progress import MapOpeningProgressSession
from caveviewer.gui.recording_controller import RecordingStateController
from caveviewer.gui.slice_export_controller import SliceExportController
from caveviewer.gui.slice_selection_controller import SliceSelectionController
from caveviewer.gui.viewer_action_dispatch import (
    ViewerActionDispatcher,
    ViewerKeyPressActions,
)
from caveviewer.gui.viewer_capture_workflow import (
    CaptureOverlayMode,
    CaptureOverlayState,
    CaptureOwner,
    CaptureOwnershipState,
    ViewerCaptureWorkflow,
)
from caveviewer.gui.viewer_frame_scheduler import (
    ViewerFramePhase,
    ViewerFrameScheduler,
    ViewerFrameState,
)
from caveviewer.gui.viewer_session import ViewerSession


class ViewerShutdownPhase(Enum):
    """Idempotent lifecycle of the coordinator's native-window shutdown."""

    ACTIVE = auto()
    CLOSING = auto()
    CLOSED = auto()


@dataclass(frozen=True, slots=True)
class ViewerWorkflowSnapshot:
    """Render-thread facts needed for all cross-workflow decisions."""

    setup_complete: bool
    closing_requested: bool
    iconified: bool
    import_active: bool
    map_loaded: bool
    capture_close_pending: bool
    recording_owned: bool
    recording_armed: bool
    recording_active: bool
    manual_dive_trace_countdown_active: bool
    manual_dive_trace_active: bool
    manual_dive_trace_finalizing: bool
    slice_countdown_active: bool
    slice_selection_active: bool
    slice_saving: bool
    slice_export_active: bool


@dataclass(frozen=True, slots=True)
class ViewerRenderRequest:
    """Non-GL decisions a viewer callback may execute for one snapshot."""

    phase: ViewerFramePhase
    capture_owner: CaptureOwner | None
    active_capture_owner: CaptureOwner | None
    capture_overlay_mode: CaptureOverlayMode
    slice_work_pending: bool
    slice_interaction_active: bool


@dataclass(slots=True)
class ViewerWorkflowCoordinator:
    """Own the non-GL controller graph and cross-workflow transitions."""

    session: ViewerSession
    recording_frame_interval: float = 1.0 / 30.0
    frame_scheduler: ViewerFrameScheduler = field(default_factory=ViewerFrameScheduler)
    capture: ViewerCaptureWorkflow = field(default_factory=ViewerCaptureWorkflow)
    actions: ViewerActionDispatcher = field(default_factory=ViewerActionDispatcher)
    map_opening: MapOpeningProgressSession = field(
        default_factory=MapOpeningProgressSession
    )
    manual_dive_trace: ManualDiveTraceStateController = field(
        default_factory=ManualDiveTraceStateController
    )
    slice_selection: SliceSelectionController = field(
        default_factory=SliceSelectionController
    )
    slice_export: SliceExportController = field(default_factory=SliceExportController)
    artifact_presentation: ArtifactCapturePresentationController = field(
        default_factory=ArtifactCapturePresentationController
    )
    recording: RecordingStateController = field(init=False)
    import_controller: MapImportController | None = field(default=None, init=False)
    benchmark_controller: BenchmarkController | None = field(default=None, init=False)
    shutdown_phase: ViewerShutdownPhase = field(
        default=ViewerShutdownPhase.ACTIVE,
        init=False,
    )

    def __post_init__(self) -> None:
        self.recording = RecordingStateController(
            frame_interval=float(self.recording_frame_interval)
        )

    def render_request(self, state: ViewerWorkflowSnapshot) -> ViewerRenderRequest:
        """Resolve frame, capture, and overlay policy from one immutable snapshot."""

        capture_owner = self.capture.owner_for(
            CaptureOwnershipState(
                recording_owned=state.recording_owned,
                manual_dive_trace_owned=(
                    state.manual_dive_trace_countdown_active
                    or state.manual_dive_trace_active
                    or state.manual_dive_trace_finalizing
                ),
                slice_owned=(
                    state.slice_countdown_active
                    or state.slice_selection_active
                    or state.slice_saving
                    or state.slice_export_active
                ),
            )
        )
        active_capture_owner = self.capture.owner_for(
            CaptureOwnershipState(
                recording_owned=state.recording_active,
                manual_dive_trace_owned=state.manual_dive_trace_active,
                slice_owned=state.slice_selection_active,
            )
        )
        return ViewerRenderRequest(
            phase=self.frame_scheduler.phase_for(
                ViewerFrameState(
                    setup_complete=state.setup_complete,
                    closing_requested=state.closing_requested,
                    iconified=state.iconified,
                    finalizing_capture=state.capture_close_pending,
                    import_active=state.import_active,
                    map_loaded=state.map_loaded,
                )
            ),
            capture_owner=capture_owner,
            active_capture_owner=active_capture_owner,
            capture_overlay_mode=self.capture.overlay_mode_for(
                CaptureOverlayState(
                    recording_armed=state.recording_armed,
                    manual_dive_trace_countdown_active=(
                        state.manual_dive_trace_countdown_active
                    ),
                    slice_countdown_active=state.slice_countdown_active,
                )
            ),
            slice_work_pending=(
                state.slice_countdown_active or state.slice_export_active
            ),
            slice_interaction_active=(
                state.slice_countdown_active
                or state.slice_selection_active
                or state.slice_saving
                or state.slice_export_active
            ),
        )

    def dispatch_key_press(self, actions: ViewerKeyPressActions) -> bool:
        """Apply the session's stable key-action priority."""

        return self.actions.dispatch_key_press(actions)

    def dispatch_key_repeat(
        self,
        *,
        waiting_for_begin: bool,
        fly_speed: Callable[[], bool],
    ) -> bool:
        """Apply repeat policy without exposing the dispatcher to the window."""

        return self.actions.dispatch_key_repeat(
            waiting_for_begin=waiting_for_begin,
            fly_speed=fly_speed,
        )

    def ensure_import_controller(
        self,
        factory: Callable[[], MapImportController],
    ) -> MapImportController:
        """Create the session's import lifecycle owner exactly once."""

        if self.import_controller is None:
            self.import_controller = factory()
        return self.import_controller

    def set_benchmark_controller(self, controller: BenchmarkController) -> None:
        """Attach the benchmark lifecycle created from the session config."""

        self.benchmark_controller = controller

    def finish_benchmark(self, *, reason: str) -> bool:
        """Finish an active benchmark once; report whether work was performed."""

        controller = self.benchmark_controller
        if controller is None or controller.finished:
            return False
        controller.finish(reason=reason)
        return True

    def begin_shutdown(self) -> bool:
        """Reserve shutdown for the first backend close path only."""

        if self.shutdown_phase is not ViewerShutdownPhase.ACTIVE:
            return False
        self.shutdown_phase = ViewerShutdownPhase.CLOSING
        self.capture.complete_close_workflows()
        return True

    def complete_shutdown(self) -> None:
        """Finish coordinator cleanup safely after partial or successful teardown."""

        if self.shutdown_phase is ViewerShutdownPhase.CLOSED:
            return
        self.capture.complete_close_workflows()
        self.map_opening.abandon()
        self.shutdown_phase = ViewerShutdownPhase.CLOSED
