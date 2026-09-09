"""Tests for the native viewer's per-launch session boundary."""

from __future__ import annotations

import pytest

from caveviewer.gui.viewer_session import (
    PendingImportRequest,
    ViewerBenchmarkConfig,
    ViewerLaunchMode,
    ViewerSession,
    ViewerSessionConfig,
    ViewerSessionOutcome,
)


def test_ready_cache_config_snapshots_its_manifest_mapping():
    manifest = {"version": 1}

    config = ViewerSessionConfig(
        mode=ViewerLaunchMode.READY_CACHE,
        cache_dir="/cache",
        textures_dir="/textures",
        manifest=manifest,
    )
    manifest["version"] = 2

    assert config.manifest == {"version": 1}
    with pytest.raises(TypeError):
        config.manifest["version"] = 3


def test_pending_import_config_snapshots_its_model_descriptor():
    descriptor = {"format": "glb", "glb_path": "/maps/cave.glb"}

    request = PendingImportRequest(descriptor, "/maps")
    descriptor["glb_path"] = "/maps/other.glb"

    assert request.model_descriptor["glb_path"] == "/maps/cave.glb"
    with pytest.raises(TypeError):
        request.model_descriptor["format"] = "obj"


def test_benchmark_config_snapshots_its_environment_mapping():
    environment = {"platform": "test"}

    benchmark = ViewerBenchmarkConfig(object(), "/results", environment)
    environment["platform"] = "changed"

    assert benchmark.environment == {"platform": "test"}
    with pytest.raises(TypeError):
        benchmark.environment["platform"] = "changed-again"


@pytest.mark.parametrize(
    "config",
    [
        ViewerSessionConfig,
        lambda: ViewerSessionConfig(
            mode=ViewerLaunchMode.READY_CACHE,
            cache_dir="/cache",
            manifest=None,
        ),
        lambda: ViewerSessionConfig(
            mode=ViewerLaunchMode.PENDING_IMPORT,
            cache_dir="/cache",
            manifest={},
            pending_import=PendingImportRequest(
                {"obj_path": "/maps/cave.obj"},
                "/maps",
            ),
        ),
        lambda: ViewerSessionConfig(
            mode=ViewerLaunchMode.BENCHMARK,
            cache_dir="/cache",
            manifest={},
        ),
    ],
)
def test_session_config_rejects_incomplete_or_conflicting_launch_inputs(config):
    with pytest.raises((TypeError, ValueError)):
        config()


def test_session_outcomes_are_owned_independently_by_each_launch():
    config = ViewerSessionConfig(
        mode=ViewerLaunchMode.READY_CACHE,
        cache_dir="/cache",
        manifest={},
    )
    first = ViewerSession(config)
    second = ViewerSession(config)

    first.record_outcome(kind="import_failed", message="first failed")

    assert first.outcome == ViewerSessionOutcome(
        kind="import_failed",
        message="first failed",
    )
    assert second.outcome == ViewerSessionOutcome()
