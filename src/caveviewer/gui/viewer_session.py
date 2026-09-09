"""Typed, session-scoped inputs and outcomes for one native viewer run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from caveviewer.core.preferences.runtime_settings import RuntimeSettings
    from caveviewer.gui.platform.runtime import PlatformRuntime
    from caveviewer.gui.recorded_dive import RecordedDiveTrace


class ViewerLaunchMode(Enum):
    """Select the mutually exclusive input contract for one viewer launch."""

    READY_CACHE = "ready_cache"
    PENDING_IMPORT = "pending_import"
    BENCHMARK = "benchmark"


@dataclass(frozen=True, slots=True)
class PendingImportRequest:
    """Describe a first-time import that begins after the first visible frame."""

    model_descriptor: Mapping[str, Any]
    textures_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_descriptor",
            MappingProxyType(dict(self.model_descriptor)),
        )


@dataclass(frozen=True, slots=True)
class ViewerBenchmarkConfig:
    """Carry benchmark policy without exposing a mutable launch dictionary."""

    scenario: Any
    output_dir: str
    environment: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(dict(self.environment)),
        )


@dataclass(frozen=True, slots=True)
class ViewerSessionConfig:
    """Immutable launch snapshot consumed by one ``CaveViewerWindow`` instance."""

    mode: ViewerLaunchMode
    cache_dir: str | None = None
    textures_dir: str | None = None
    map_root: str | None = None
    manifest: Mapping[str, Any] | None = None
    pending_import: PendingImportRequest | None = None
    benchmark: ViewerBenchmarkConfig | None = None
    recorded_dive_trace: RecordedDiveTrace | None = None
    platform_runtime: PlatformRuntime | None = None
    runtime_settings: RuntimeSettings | None = None
    vsync: bool = True

    def __post_init__(self) -> None:
        if self.manifest is not None:
            object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        self._validate_mode()

    def _validate_mode(self) -> None:
        ready_cache = self.cache_dir is not None and self.manifest is not None
        if self.mode is ViewerLaunchMode.READY_CACHE:
            valid = ready_cache and self.pending_import is None and self.benchmark is None
        elif self.mode is ViewerLaunchMode.PENDING_IMPORT:
            valid = (
                self.cache_dir is None
                and self.manifest is None
                and self.pending_import is not None
                and self.benchmark is None
            )
        elif self.mode is ViewerLaunchMode.BENCHMARK:
            valid = ready_cache and self.pending_import is None and self.benchmark is not None
        else:
            valid = False
        if not valid:
            mode_name = (
                self.mode.value
                if isinstance(self.mode, ViewerLaunchMode)
                else repr(self.mode)
            )
            raise ValueError(f"Invalid viewer session inputs for {mode_name!r} mode")


@dataclass(frozen=True, slots=True)
class ViewerSessionOutcome:
    """Describe why one native viewer session returned to its owner."""

    kind: str = "window_closed"
    message: str = ""
    suggestion: str = ""


@dataclass(slots=True)
class ViewerSession:
    """Own mutable completion state for exactly one immutable launch snapshot."""

    config: ViewerSessionConfig
    outcome: ViewerSessionOutcome = field(default_factory=ViewerSessionOutcome)

    def record_outcome(
        self,
        *,
        kind: str,
        message: str = "",
        suggestion: str = "",
    ) -> None:
        """Replace the result recorded by the render-thread lifecycle owner."""

        self.outcome = ViewerSessionOutcome(
            kind=str(kind),
            message=str(message),
            suggestion=str(suggestion),
        )
