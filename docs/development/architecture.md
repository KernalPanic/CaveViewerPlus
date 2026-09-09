# Architecture

This document describes the current architectural boundaries. The filesystem
contract is documented in [repository-layout.md](repository-layout.md).

## Documentation authority

| Concern | Canonical document | Other documents should do |
| --- | --- | --- |
| Cross-layer ownership and dependency direction | This document | Link to the relevant heading. |
| Runtime configuration resolution and transport | [Runtime configuration](runtime-configuration.md) | Keep commands and generated tables in source setup. |
| Platform adapter routes and native behavior | [`platform-adapters.md`](../../src/caveviewer/gui/platform/platform-adapters.md) | Link back here for the general boundary. |
| Commands, environment reference, and troubleshooting | [Source setup](source-setup.md) | Avoid restating architecture mechanics. |
| Releases, channels, and workflow sequencing | [Releases](releases.md) | Link to the update boundary here when needed. |
| Visual-branding roles and platform outputs | [Branding](branding.md) | Preserve the stable product-identity boundary. |
| Directory-local instructions | Nearest `AGENTS.md` | Link to development docs instead of copying narrative. |

Focused documents own subsystem mechanics; this page owns allowed dependency,
thread, process, and state-machine boundaries.

## Dependency direction

```text
caveviewer.app
    ├── caveviewer.storage_paths XDG and portable storage roots
    ├── caveviewer.core       preferences, discovery, import/cache, streaming policy
    └── caveviewer.gui        startup panels, dialogs, rendering, platform integration
          ├── caveviewer.core
          └── caveviewer.benchmarking benchmark controller adapter

caveviewer.benchmark              direct cache/scenario benchmark CLI
    ├── caveviewer.benchmarking    scenarios, metrics, comparisons, and routes
    └── caveviewer.gui             viewer runtime adapter

caveviewer.benchmarking
    ├── map_runner                 local map benchmark CLI orchestration
    └── caveviewer.core.navigation reusable route and centerline primitives
```

`caveviewer.core` must not import `caveviewer.gui`, `caveviewer.app`, or
benchmarking. GUI, benchmark, and application entry-point code may call core
services. Benchmarking may call reusable core policy. Concrete Tk and OpenGL
work stays in the GUI layer. Platform behavior is selected through
`caveviewer.gui.platform` profiles, focused action facades, and compatibility
adapters where a concern has not yet moved.

`caveviewer.benchmarking` owns benchmark scenario parsing, measurement
summaries, regression comparisons, benchmark-specific route selection, and the
generic local map benchmark runner exposed as `caveviewer-map-benchmark`. It may
depend on reusable core navigation/streaming policy, but it must not own viewer
presentation or render-thread OpenGL resources. `viewer_window.py` adapts a
`BenchmarkController` into the real render loop when the benchmark CLI launches
the viewer.

Visual branding follows the same composition direction. A GUI-free resolver
may validate semantic profile inputs, while the application composition
boundary creates one immutable branding snapshot for GUI consumers. Tk,
OpenGL, Pillow-backed exports, and native packaging tools remain outside core
domain policy. The complete surface and stable-identity contracts live in
[Branding](branding.md).

## Runtime settings

`caveviewer.core.preferences.runtime_settings` owns the declarative runtime
settings registry and the immutable `RuntimeSettings` composition snapshot. It
has no GUI, Tk, OpenGL, or application-entry-point dependency. The composition
root supplies saved preference values, environment values, command-line
overrides, and stable platform facts explicitly; the resolver never mutates the
process environment. Each resolved value retains provenance (`built_in`,
`preferences`, `environment`, or `cli`) and rejected fall-back values are
returned as immutable issues for the composition boundary to report.

`caveviewer.core.release_metadata` is a separate core-only boundary for the
immutable release metadata frozen into a package. It validates the small
versioned resource and supplies a safe stable fallback for source checkouts or
historical packages. `caveviewer.app` loads it once into `RuntimePlatformFacts`;
runtime settings then use that injected value as the built-in update-channel
default. The loader does not import GUI, Tk, OpenGL, platform adapters, or the
application entry point.

`PreferenceSpec` remains the sole authority for persisted setting validation,
ranges, conversion, and built-in defaults. Runtime entries reference those
specifications instead of duplicating their metadata. Persisted settings retain
the existing precedence: valid saved preference, then valid environment value,
then built-in default. Environment-only values can have an explicitly declared
CLI override, primary variable, legacy alias, and built-in order. Packaging and
development-shell variables are intentionally excluded from the application
snapshot.

`caveviewer.core.preferences.transfer` owns the bounded, GUI-free portable JSON
contract. It atomically exports complete immutable snapshots, rejects malformed
documents, ignores unknown keys, and resolves missing or invalid declared values
independently to defaults. `caveviewer.gui.preferences_workflow` coordinates
those file operations, while `PreferencesPanel` owns native chooser presentation
and stages imported/default snapshots until Apply. Core preference code does not
depend on Tk or desktop-service adapters.

`caveviewer.app` owns a `RuntimeSettingsSession` for the interactive process.
It replaces its snapshot after a successful Preferences save and passes the
current immutable value to platform policy, splash composition, Map Library,
viewer launch, and import-child requests. Narrow serializable subsections own
worker transport; workers never depend on a parent changing `os.environ` after
they start. The benchmark entry point follows the same composition path rather
than exporting saved Preferences into process state.

## Route primitives

`caveviewer.core.navigation.centerline` and `caveviewer.core.navigation.route`
provide deterministic manifest-derived centerlines and generic camera-route utilities.
Benchmarking uses them to construct repeatable benchmark scenarios; they do not
affect normal viewer free flight, cache construction, Manual Guided Dive traces,
Recorded Dive playback, or cave slicing.

## Viewer workflows

An explicit `Cmd/Ctrl+T` manual route trace is a separate diagnostic surface.
The same shortcut first shows a render-thread 3-2-1 countdown, then begins
sampling the camera pose after movement. It sends JSONL records through a
bounded queue to one background writer and marks bookmark/minimap teleports as
discontinuities instead of counting them as flown distance.
Completed traces live under the map-local `_guided_dives` directory. Their
location is anchored to the map root rather than the generated-cache location,
so atomic cache replacement and managed-cache storage do not erase the
reference flight. They remain optional ground truth: cache construction never
consumes them as required map metadata.
Video recordings and completed traces keep their separate capture writers, but
share one post-save presentation workflow. After a user-visible stop, the
viewer keeps a saving status visible until the writer has atomically published
the final file, then shows a three-second success confirmation before asking
the runtime's `SavedArtifactRevealAdapter` to reveal that MP4 or JSONL file.
Non-user map-change and shutdown stops remain silent; failure feedback and
native reveal are reserved for the user-visible workflow. Reveal is best-effort
and cannot change the completed artifact result.

Video recording, manual tracing, and cave slicing share one mutually exclusive
capture-ownership boundary, including countdown and asynchronous finalization.
Once a capture owns that boundary, recognized shortcuts for other capture types
are consumed before their toggle or presentation paths run, so they neither
change state nor display status. The owner's shortcut remains available to
finish and publish an active capture.
Every countdown presents its normal stop-and-save shortcut alongside **Press
Esc to cancel**. Video recording, manual tracing, and cave slicing all stay
banner-free after the countdown so the cave view remains unobstructed. The
normal shortcut publishes the artifact; Escape follows a distinct discard
path. Recording cancellation
immediately releases render-thread readback buffers, drops queued raw frames,
asks the encoder finalizer to stop, and removes its MP4. Trace cancellation
drains the bounded pose queue, stops the writer, and removes both its private
`.part` file and any not-yet-presented JSONL publication. Slice cancellation
clears selection state or signals the export child, whose atomic staging path
is removed before the controller returns to idle. These worker/process cleanups
remain polled and non-blocking on the render thread. Successful cleanup ends
with an artifact-specific cancellation confirmation that explicitly states no
artifact was saved and remains visible for three seconds. Save confirmations
retain their three-second period.

The viewer claims Escape from `moderngl-window` instead of leaving it as the
backend's default exit key. Backends such as GLFW invoke their close callback
before forwarding that key event, which would otherwise enter normal
save-on-close finalization and display **Finishing…** before CaveViewer could
cancel. CaveViewer routes Escape first: an active capture enters a distinct
discard-before-close state, suppresses further input and native close requests,
shows **Canceling…** while cleanup is pending, holds the final no-save status
for three seconds, and only then releases the window. Escape without a capture
still closes immediately. Native title-bar close requests retain the existing
save-on-close path and its **Finishing…** status.

`Ctrl+C` (`Cmd+C` on macOS) owns the separate cave-slice workflow. It first
uses the same render-thread 3-2-1 countdown presentation as recording and
manual tracing, then records the current camera position as a slice start
anchor. A second shortcut press records the end anchor and starts a bounded
child-process export; the normal viewer remains interactive while geometry and
assets are written. The exporter selects source render chunks intersecting the
padded axis-aligned volume and copies those complete chunk files without
rewriting their geometry. It regenerates the detailed minimap footprint from
the included vertices, stages a standalone precompiled-map directory, and
atomically publishes it below the Preferences **Downloaded maps folder**
location. Complete chunks may extend beyond the two camera endpoints; this is
the deliberate tradeoff that preserves the original walls, UVs, normals, and
material groups. The saved slice contains its own manifest, render chunks,
referenced cache-local texture assets, and a small `.cvslice` source marker, so
it can be copied to a machine that lacks the parent OBJ/GLB and still open
through the normal precompiled-map path.
It records additive parent/selection metadata and derives a distinct Guided
Dive identity, but does not copy parent bookmarks or prior traces. On normal
completion it uses the shared saving/success/reveal presentation; slice success
reveals the new map directory rather than opening
it. Closing during active slicing turns the current camera position into the
end anchor, defers window teardown until publication reaches a terminal state,
and then reveals a successfully published slice directory before closing.
Closing during the pre-start countdown simply cancels that countdown.

Recorded Dive is the separate trace-playback path. Opening a completed JSONL
associates its bounded source basename, cache-manifest version, chunk size,
triangle count, and versioned cache identity with the local map. Cache
construction writes that identity from a streaming SHA-256 of the source file
and a canonical SHA-256 of the completed manifest. Normal rendering remains
compatible with manifests that predate this additive field, but Guided Dive
recording and v2 playback require a rebuild when it is absent.
Normal cache validation rebuilds a stale or missing map-local cache before
viewing; playback refuses a different geometry or cache layout. The trace's
first pose replaces ordinary cache-derived viewer placement, and every pose is
applied directly on the render thread without navigation clamping, smoothing,
collision rejection, or route planning.
Map Library exposes **Open guided dive…** only when the selected map's
canonical `_guided_dives` directory has a JSONL file. Its action first obtains
a fresh file-selection preflight, then invokes the authorized desktop file
picker only if its typed route remains executable. The selected file then
obtains one fresh domain capability fact: it must remain map-local, parse
within the bounded trace contract, resolve to that map's source, and match a
current cache manifest exactly. `decide_guided_dive_playback` hides a map with
no trace and otherwise fails closed with a concise disabled-state explanation.
Both checks are action-time preflights, not `PlatformRuntime.feature_gates`
entries; startup repeats the manifest validation at the viewer boundary to
cover a filesystem change after splash has closed.
Position and orientation are interpolated by trace time; a declared
discontinuity remains an instantaneous jump. `StreamingWorld` receives a
bounded chronological lookahead tube, and the playback clock freezes whenever
the next pose's local render chunks are not GPU-resident. This makes trace time
independent of render frame rate while allowing slower hardware to buffer
without skipping part of the recorded flight.
While a user-paused dive is inspecting the cave, only camera look is applied:
its trace time and position remain frozen. Resuming reapplies the authoritative
recorded pose at that timestamp, then returns through the same chunk-buffering
path before the trace clock advances again.

The process boundary also installs main-thread and worker-thread exception
hooks. `ApplicationDiagnostics` is a generic optional sink with no output path
until an explicit consumer binds one, so diagnostics never become cache-local
artifacts. On every supported desktop, the application binds it to a per-run
user-profile JSONL file and pairs it with `RuntimeDiagnostics`: a
human-readable session log that receives normal application logging and, when
available, fatal native fault tracebacks. The viewer records checkpoints before
native launch, context creation, and the first render so a crash can be located
even when the packaged application has no console. Core diagnostics catalog
policy orders eligible text logs, bounds error-excerpt reads, removes session
artifacts older than 24 hours at startup, and retains no more than the newest
ten sessions; Help presentation and native file reveal remain GUI and
platform-adapter responsibilities.

On Windows, the module entry point also owns one short-lived
`StartupDiagnostics` session before importing `caveviewer.app`. It writes one
user-owned `startup.log`, records pre-splash composition checkpoints, and owns
one 20-second all-thread traceback watchdog. The session ends immediately after
the Tk splash is deiconified, cancelling the watchdog and closing the log; it
does not retain a timer or verbose file logging during ordinary viewing.

## Startup and map import

Core import services discover supported models and dispatch them to the OBJ or
GLB parser. `core.map.source_model` owns the immutable source-format registry,
source selection, and selected-format capability facts; that registry is the
release-policy source of truth for discovery, map-picker guidance, and Linux
package metadata validation. `gui.map_opening.map_source_import_preflight`
pairs one selected descriptor's capability with
`gui.features.decide_map_source_import` before the GUI accepts an import; an
executable decision must select the exact route declared by that source format.
It is intentionally not stored in
`PlatformRuntime.feature_gates`: the released format list is static, but the
user-selected descriptor and its required companion assets vary per action.
`core.map.importer` owns parse/cache orchestration, and app/GUI/CLI code adapts
those services to console or Tk progress displays. Parsers produce CPU-side
mesh and material data. `src/caveviewer/core/chunking/builder.py` partitions
that data and builds a cache in a private staging directory. Cache locations are
selected through `core.map.cache_paths`; the default generated cache directory
is `_cache` inside the source map folder. Explicit `CAVEVIEWER_MAP_CACHE_DIR`
or CLI `--cache-root` callers use hashed cache directories under that separate
root. The older `.caveviewer_cache` directory is not auto-discovered.
Chunks, the manifest, and referenced texture assets are published in one atomic
directory transaction. Failures must remove staging output and preserve any
previously valid generated cache.

The splash Map Library can request a forced rebuild only for an existing
generated cache whose source model is still readable. Its map-local
`gui.map_cache_rebuild` preflight is evaluated for each overflow menu and again
at the action boundary; it stays outside `PlatformRuntime.feature_gates` because
source files, cache safety, and active builders vary per map. The dedicated
`CacheRebuildJobController` owns that child import from the splash, reports
inline progress and cooperative OBJ pause checkpoints, and never opens a
viewer after publication. Rebuilds use the same staging/publish transaction as
normal imports, so the old cache remains available until replacement succeeds.
Closing the splash requests a bounded cooperative pause; its rebuild child is
kept non-daemon long enough to save the checkpoint or safely finish rather than
being destructively terminated with the Tk window.
When the splash no longer has input focus, terminal cache-rebuild success and
failure also use the optional desktop-notification boundary; progress and
intentional pauses remain inline only. The notification route is freshly
preflighted at send time and cannot affect the rebuild when unavailable.
At the core build boundary, `core.map.cache_build_lock` atomically claims a
private sibling lock directory for each cache target. A second cooperative GUI
or CLI build fails closed rather than racing the target; normal completion,
failure, or pause releases the lock while preserving any resumable staging
checkpoint.

`core.map.slicing` applies that same lock/staging discipline to derivative
maps: it holds the parent cache lock while reading chunks, locks the new output
target, validates texture paths remain cache-local, and rechecks the parent
manifest identity before publication. Cancellation and failures remove only
the private slice staging directory; neither can modify the parent cache.

First-time imports launched from the viewer run in a spawned child process
through `src/caveviewer/gui/import_process.py`. The viewer process owns OpenGL,
window events, progress rendering, and desktop idle/suspend inhibition; the
child process owns parsing, cache construction, texture staging, and cache
publication. Progress, completion, and traceback-bearing failure events cross
back to the viewer through a process queue. This keeps desktop event loops
responsive during CPU-heavy imports and isolates import crashes from the UI
process. Viewer shutdown asks the parent-side relay worker to stop, waits for a
short bounded interval, terminates any reachable active child process, and then
ignores late import messages so closed windows cannot apply stale completion
events. The child emits heartbeat events with the current stage and RAM snapshot
while it is working, runs at reduced desktop priority, and caps common native
compute-library thread counts before importing NumPy-heavy modules.
Parent-side cancellation, shutdown, or abnormal child exit cleans abandoned
private staging directories for the target cache when the child is no longer
alive.

Import preflight is intentionally early. OBJ imports count vertices, UVs,
normals, and triangulated faces before allocating large arrays; that count is
used to reject imports whose estimated peak footprint exceeds currently
available system RAM. Disk preflight runs before parsing and includes both the
source model and staged texture assets when they are known. `build_cache()`
repeats disk checks before writing so direct callers and mid-import free-space
changes remain covered.

Chunk-file construction treats its configured worker count as a maximum. It
starts with one task, samples current system RAM after completed work, and
admits only one additional concurrent worker per sample while utilization is
below 80%. Unknown availability or memory pressure keeps the build at its
already-admitted concurrency, with one worker always able to make progress.

The render-cache manifest records chunk metadata, spatial bounds, material
references, the minimap occupancy footprint, and an additive
`guided_dive_identity` when source hashing succeeds. This identity
is produced during cache construction, not while the render thread starts a
manual trace. Existing cache manifests stay renderable; they must be rebuilt
before the versioned Guided Dive contract can record or replay against them.
Historical `navigation_certificate/` sidecars are not render-cache data.
They are ignored during normal cache validation and map opening, without a cache
format migration or background inspection.

The render-chunk binary format remains at version 1: unknown manifest fields
and extra subdirectories inside a selected generated cache are ignored, while
imports write only the active cache artifacts.

## Runtime streaming

`src/caveviewer/core/streaming/world.py` coordinates worker lifecycle and
render-thread callbacks. Runtime streaming depends on focused core policy
modules:

- `caveviewer.core.hardware.system_memory`: typed total/current system-RAM
  availability probes and legacy total-RAM fallback.
- `caveviewer.core.hardware.gpu_memory`: typed active-GPU memory probes and
  conservative fallback budgets.
- `caveviewer.core.hardware.memory_targets`: RAM and GPU utilization target
  parsing.
- `caveviewer.core.workers.allocation`: CPU caps and shared worker RAM admission.
- `caveviewer.core.streaming.budget`: pure typed-memory-to-residency policy,
  chunk-size estimation, and residency limits.
- `caveviewer.core.streaming.scheduler`: backlog, selection, and eviction.
- `caveviewer.core.textures.decoding`: worker-safe CPU texture decode,
  inspection, and texture budget selection.

Workers load and prepare CPU payloads. The viewer performs OpenGL uploads and
unloads on the render thread. Internal residency state and external GPU state
must remain transactionally consistent when callbacks fail.
On Linux, each streaming worker raises its own nice value by the configured
increment so chunk preparation yields CPU time to the GUI/render thread; a
process-wide `os.nice()` call is intentionally not used for this runtime pool.
Streaming starts one worker and considers one additional worker only after a
prepared chunk is resident in the bounded ready queue, so each memory sample
includes real decode cost. Pool growth stops when system RAM utilization
reaches 80% or availability cannot be measured and may resume if pressure
later falls.

Streaming memory probes are converted into immutable capability facts before
the pure residency policy runs. Measured RAM and GPU budgets use their selected
targets; unknown inputs use a deterministic 1 GB fallback envelope, and unknown
RAM cannot raise the normal conservative utilization target. This keeps a probe
failure from becoming an unbounded residency allowance while preserving a
minimal streaming path.

Geometry visibility is not limited by full-resolution texture residency.
`StreamingWorld` selects chunks using spatial distance and chunk residency
budgets; oversized texture sets are handled by splitting CPU texture decode
from OpenGL texture ownership. `core.textures.decoding` derives a decode-time
maximum texture dimension from detected GPU memory, target percentage, and
unique texture count, then workers decode Pillow image data into CPU bytes.
`gui.texture_manager` consumes those decoded payloads on the render thread,
creates/reuses/releases OpenGL textures, and enforces render-thread ownership
for texture GPU work. `gui.chunk_upload` owns resident chunk GPU bookkeeping,
partial upload state, unload cleanup, and shade-mode VBO rewrites. Runtime
uploads advance through render-thread operation queues: texture allocation is
separated from row-band writes, and dense material groups are split into
triangle-aligned VBO slices whose storage reservation and data writes advance
separately where the OpenGL context supports it. Texture and VBO slice sizes
start conservatively and shrink automatically after measured upload stalls.
This keeps the visible cave geometry from collapsing to only the few chunks
whose original texture tiles fit in VRAM, while still preventing obviously
oversized texture uploads. GPU memory detection is platform-specific:
NVIDIA uses `nvidia-smi` when available, Linux AMD uses DRM sysfs, low-VRAM AMD
integrated GPUs add 50% of reported GTT/shared memory capped at 2 GB, Windows
AMD/Intel currently use an 8 GB fallback budget, and macOS currently uses a
conservative 1 GB fallback when no override is set. Texture cap selection logs
the budget inputs and the selected common dimension before the first oversized
texture is resized.

## UI and platform boundaries

Tk panels and dialogs should keep validation and workflow state in testable
controller or model modules. `caveviewer.gui.platform` contains OS-specific
focus, update, and system integration behavior. Unsupported platforms use the
default adapter.

### Capability, policy, and feature-gating contract

Platform-dependent feature work follows one direction:

```text
edge probe -> immutable CapabilityResult -> pure policy -> FeatureDecision
          -> injected adapter or service -> feature execution
```

Probes report facts and diagnostics-safe evidence; they do not choose product
behavior. Policies receive only those facts and return an immutable decision
with a stable `reason_code`, concise user-safe `explanation`, and selected
`route`. Adapters and `DesktopServices` perform the chosen native action but
do not decide whether the product feature is available.

`PlatformRuntime.feature_gates` contains only process-stable decisions, such
as automatic-update compatibility and the selected native route for revealing
a verified update package. It is composed once after command-line overrides,
then injected into every interactive viewer path, including a direct CLI map
launch. A mutable action prerequisite, such as an ffmpeg path or a writable
recording folder, uses an on-demand preflight instead of a cached startup gate.
The preflight pairs one fresh capability result with the policy decision
derived from that same snapshot. `gui.platform.recording_preflight` is the
shared recording boundary: it delegates to an injected runtime when available
and otherwise constructs the same `VideoRecordingPreflight`, so callers cannot
separate a recording probe from its policy decision. Every executable typed
preflight validates
that its decision names the expected feature and the route declared by its
available target; shared validation lives in `gui.features.preflight` so
action boundaries fail closed on a malformed or mismatched pair.
The GUI architecture-boundary tests enforce these ownership rules: only the
pure policy module constructs `FeatureDecision` values, only
`gui.platform.runtime` composes `FeatureGateRegistry`, and `UpdateManager`'s
normal clients use the runtime-composed typed update target APIs.

Static GUI presentation uses a parallel, deliberately non-capability path:

```text
composed platform name -> immutable PresentationProfile -> injected UI consumer
native UI effect       -> PresentationActionsAdapter -> action-time call
```

`PresentationProfile` contains only process-stable conventions: font families
and candidates, splash and embedded-panel layouts plus native-dialog layouts,
shortcut and mouse-input labels, text scaling, startup focus policy, and backend
sizing preferences.
It is selected without creating Tk widgets or probing the display. The narrow
action adapter performs only process DPI setup, macOS About-menu registration,
and best-effort viewer focus. It selects direct Windows, macOS, or fallback
implementations from the composed platform fact and does not depend on
any general-purpose platform facade; Linux fontconfig lookup remains an
action-time font fallback rather than a profile-selection side effect.

Automatic updates have a typed static boundary. `select_update_profile()` maps
only the composed platform and process architecture to an immutable
`UpdateProfile`: install channel, supported architectures, manifest layout,
user agent, accepted package kinds, and signed-manifest field aliases.
Environment and CLI-derived overrides turn that profile into an
`UpdateConfiguration`; `probe_automatic_update()` then produces an immutable
`UpdateTarget` only when both manifest endpoints are configured and the target
is supported. The target carries both the existing install channel and the
expected manifest channel; the checker accepts a declared signed
`release_channel` only when it matches that expected channel (while accepting
legacy manifests without the field during the migration window). `UpdateManager` requires that composed runtime: its default
checker and downloader calls use the target and focused `TlsTrustAdapter`, so
release policy and manifest parsing stay within their typed update boundary.

Verified update-package reveal uses a focused adapter. At composition it
declares `finder`, `explorer`, or Linux `desktop_service` without mounting a
DMG, launching a file manager, or contacting D-Bus. The pure policy stores the
resulting static decision, and `UpdateManager` checks it again immediately
before revealing the verified payload. The action remains non-executing:
macOS's existing read-only DMG mount/reveal path, Windows Explorer selection,
and Linux desktop-service fallback are implemented by direct focused adapters.

Verified update-package storage uses a similarly focused adapter, but it is
not a feature gate. Checksum verification has already completed when
`UpdateManager` calls `UpdatePackageStorageAdapter`, while the availability of
a local destination can change at any time. The adapter promotes the temporary
verified payload and returns its final path; a storage exception is an ordinary
update-workflow failure and still runs the normal temporary-file cleanup. Its
direct platform implementations preserve macOS DMG naming, Linux AppImage
permissions, and legacy package Downloads handling. A Windows EXE eligible for
automatic installation is instead kept under the current user's private
`%LOCALAPPDATA%\CaveViewer\updates` root. To avoid publishing a partial
cross-filesystem copy, adapters copy into a hidden temporary sibling, flush and
close it, then atomically rename it to the chosen non-conflicting final path. A
failed promotion removes only that hidden file and never exposes a final
package path.

Saved-artifact reveal is another focused action, not a feature gate. A video
encoder or trace writer has already reported success when `CaveViewerWindow`
calls `SavedArtifactRevealAdapter`, and the action is only a post-save
convenience after a user-visible stop. A failure to launch Finder, Explorer, or
the Linux desktop reveal route is logged but cannot downgrade the completed
artifact's success state. Direct focused implementations own Finder selection,
Explorer selection, and the injected Linux desktop-service route; none delegates
to a general-purpose platform facade.

Recording encoder startup is likewise a focused action adapter, not another
recording gate. After the existing on-demand preflight has confirmed ffmpeg and
the output directory, `RecordingProcessAdapter` supplies only the native
non-command `Popen` options immediately before the encoder session starts. It
does not select an ffmpeg binary, build the command, or alter recording policy.
The direct Windows adapter preserves console suppression through `STARTUPINFO`
and `CREATE_NO_WINDOW`, while the default, macOS, and Linux adapter returns no
extra launch options.

TLS trust augmentation is also a focused action adapter, not a capability gate.
Each update-network request creates Python's normal verifying SSL context, then
asks `TlsTrustAdapter` to add any native trust roots before it contacts a
manifest, signature, package-availability probe, or verified payload URL. The
direct Windows adapter preserves `CA`/`ROOT` certificate-store augmentation;
the default, macOS, and Linux adapter adds nothing. Neither path disables
certificate verification.
The process-global `truststore` startup compatibility path remains separate;
this adapter does not change process initialization or network policy.
The neutral file-download transport receives that explicit context from its
caller, allowing Map Library archive downloads to retain the same trust setup
without depending on update configuration or updater compatibility APIs.

Directory selection follows the same on-demand contract. Its immutable target
declares an executable route rather than performing a desktop request:
Linux declares `portal_then_tk`, portable desktop services declare `tk`, and
legacy injected services use the conservative `injected` route. The declaration
does not create Tk resources or contact D-Bus. An executable directory
preflight must prove that its decision's route matches the typed target, and
the directory-selection boundary rechecks the desktop service's route
declaration immediately before it invokes a chooser. A changed or mismatched
route fails closed before native chooser setup. Map-opening and Preferences
browse actions obtain that fresh preflight immediately before invoking the
chooser; the Portal service still owns the action-time fallback to Tk if its
current request fails. The Preferences “Downloaded maps folder” control
therefore shares the same on-demand contract used by Map Library storage
without adding a separate startup gate.

File opening uses a separate but matching on-demand contract. Its immutable
`FileSelectionTarget` declares the Portal/Tk composite, Tk route, or
conservative injected route without creating Tk resources or contacting D-Bus.
An executable file-selection preflight must match its policy route to that
typed target, and its boundary rechecks the desktop declaration immediately
before `choose_file`. A changed or unavailable route fails closed before the
Guided Dive picker opens. Linux Portal-to-Tk fallback remains inside
`LinuxPortalDesktopServices`; after a selected file returns, the separate
map-local Guided Dive trace/cache preflight remains authoritative.

Desktop notifications use the same typed, on-demand boundary but remain an
optional enhancement rather than a gate on their owning workflow.
`DesktopNotificationTarget` declares Linux's `portal_then_noop` composite, a
portable no-op route, or a conservative injected route without sending a
message or contacting D-Bus. A portable no-op reports unavailable; an
available Portal or injected route is rechecked immediately before send or
withdraw. The narrow notification action boundary turns an unavailable,
unknown, changed, or failed route into a logged no-op, so an update download or
Map Library download always retains its normal state and outcome. Notification
preflights are deliberately not entries in `PlatformRuntime.feature_gates`.

Idle/suspend inhibition follows a separate matching on-demand contract.
`IdleSuspendInhibitionTarget` declares Linux's `portal_then_noop` composite, a
portable no-op route, or a conservative injected route without opening D-Bus
or starting an inhibition worker. The acquisition boundary rechecks the typed
route immediately before it asks for a scoped handle and converts unavailable,
unknown, changed, or failed acquisition into a logged no-op. Releasing an
already acquired handle is ordinary cleanup and is never re-gated, so imports,
Map Library downloads, and update downloads always release their valid handle
even if desktop availability changes later. Inhibition preflights are likewise
not entries in `PlatformRuntime.feature_gates`.

Viewer launch follows the same on-demand contract because display endpoints
and an explicit `CAVEVIEWER_WINDOW_SYSTEM` choice can change while the splash
UI remains open. `ViewerLaunchTarget` carries either the native ModernGL route
or an ordered Linux GLFW X11/Wayland plan; the side-effect-free probe does not
import or initialize GLFW, create a test window, or allocate a rendering
context. A fresh `ViewerLaunchPreflight` must match its typed target and is
rechecked immediately before native execution. `WindowBackendAdapter` then
executes exactly that target, retaining the existing automatic Linux retry only
for recognized backend/context initialization failures. A shader, map, or
other renderer/application failure never selects a second backend. The target
and adapter provide a future seam for a macOS Metal route without changing
map-opening policy callers; they do not implement Metal themselves. Viewer
launch is mutable and therefore does not enter `PlatformRuntime.feature_gates`.

Feature-state semantics are fixed:

| State | Presentation | Execution |
| --- | --- | --- |
| `enabled` | Normal feature affordance | The selected normal route may run. |
| `degraded` | Available with its fallback explained | The selected safe fallback route may run. |
| `disabled` | May show a concise explanation | No route may run. |
| `hidden` | Do not present the feature | No route may run. |

`UNKNOWN` is a capability fact, not a feature state. Each policy explicitly
chooses whether it can use a conservative fallback or must fail closed. A UI
button state is never the enforcement boundary: services re-evaluate mutable
preconditions immediately before irreversible work. Development overrides may
disable behavior for testing but never bypass a hard safety or compatibility
requirement.

GUI architecture guardrails are executable. The test file
`tests/unit/gui/test_gui_architecture_boundaries.py` checks that GUI modules do
not import upward into `caveviewer.app`, that direct platform checks stay
inside `src/caveviewer/gui/platform`, and that GUI Python modules carry
ownership docstrings instead of placeholder module-path docstrings.

The splash Map Library is split by responsibility: `map_library.py` builds
presentation-independent recent-map titles, `map_library_sources.py` owns the
source-neutral catalog contract and enabled-source composition,
`map_library_controller.py` owns source-qualified row, transfer, and
availability state, `map_library_workflow.py` owns catalog reconciliation and
Tk-thread workflow transitions, `map_library_panel.py` owns Tk rows, scroll,
status, and overflow-menu presentation, and `splash_screen.py` wires those
pieces to session actions such as opening maps and preferences.

The short launch surface is separate from the persistent Map Library: it uses
the same solid Void background and routine flat-progress geometry as viewer
map loading while the splash prepares its composed main surface, then releases
that surface before Map Library, Preferences, Help, or About are shown.
`gui.splash_visuals` owns bounded supersampled ring and vector-icon
rasterization. Tk callers alone turn those images into `PhotoImage` values on
the UI thread, so update-download, Map Library progress circles, chevrons, and
other curved Canvas affordances share smooth high-DPI edges without placing
image work in import or update workers. The OpenGL viewer requests 4x
multisampling for its
default presentation framebuffer; the shared import/map-opening/cache/capture
ring shader uses framebuffer derivatives for its edges, and the minimap uses a
higher-density round-marker tessellation. FreeType text and axis-aligned panels
remain on their existing resolution-aware rendering paths.

Each `MapLibrarySource` returns a `MapCatalogRefresh` with a stable source id,
ordered entries, and an explicit authority result. The initial production
adapter is `GitHubReleaseMapLibrarySource`; it preserves the configured GitHub
release behavior, but the workflow does not depend on GitHub endpoints or
release parsing. Map identity is `(source_id, catalog_id)`, so future enabled
sources can use the same catalog id without colliding in rows, active work,
registry records, or local storage. GitHub keeps its established map-folder
layout; other source ids use an app-managed source namespace below the selected
map-library directory.

The bundled `cave_metadata_catalog.v1.json` is a separate offline,
descriptive catalog. `cave_metadata.py` validates it and resolves only exact
or conservative unambiguous name/alias matches; map-source entries may supply
an explicit `cave_metadata_id` to avoid heuristic association. A metadata
match changes only the row subtitle and enables an in-splash **About Cave**
surface rendered by `cave_metadata_panel.py`. It never authorizes an action,
changes source reconciliation, or describes the correctness of the 3D map.
The splash composition root owns user-selected external source opening through
`DesktopServices`.

An authoritative refresh is the source's current list: it adds newly available
maps, removes stale undownloaded rows, and marks a missing app-managed local
installation as a former map while keeping it in its prior **CaveViewer Maps**
position. A failed or invalid refresh is non-authoritative and may show
cached/bundled entries, but it never marks a map removed.
`standard_library_maps.py` stores the GitHub catalog cache and a versioned
private managed-install registry. The registry records only known app-managed
paths, allowing a removed upstream map to remain visible without scanning user
folders. A former row has a muted title and remains a normal local map: it can
open, run Guided Dive, rebuild or remove cache data, and remove its files. It
cannot download or update because its source no longer offers it. Removing the
local files removes the former row through the normal map-removal path.

`map_library_panel.py` schedules scroll-region synchronization after every row
or section change so asynchronously added catalog rows stay reachable. Its
override-redirect overflow menu owns scoped temporary splash-root pointer and
focus bindings while open; those bindings are removed on close and never use a
global binding that could affect other windows.
The Map Library also owns the Guided Dive action-time handoff: it receives the
splash-owned runtime, authorizes file opening only after the map-local
discovery policy is enabled, runs the selected trace/cache preflight, and
leaves splash only after the resulting target is executable. It does not reuse
the directory-selection gate for this file-picker action.

Directory selection, file selection, file reveal, notifications, and
idle/suspend inhibition use the separate `DesktopServices` capability. Linux
asks XDG Desktop Portal first and falls back to Tk or `xdg-open` only when the
portal is unavailable. Map-folder and Guided Dive file selection are each
policy-gated independently of the other desktop actions: an enabled Portal/Tk
composite or degraded Tk/injected route may run, while a missing, indeterminate,
or changed chooser route is blocked before the chooser is invoked. Long map
library downloads request desktop notification and inhibit support through this
same capability, but the visible Map Library panel suppresses duplicate
desktop notifications because it already presents progress and completion
actions. Background update downloads request notification and inhibit support
while the package is being downloaded and verified; a visible splash suppresses
duplicate desktop notifications because it already presents the update state
and actions. Notification sends and withdrawals use a fresh optional-route
preflight; unavailable or indeterminate notification support is a diagnostic
no-op, not a download failure. Idle/suspend inhibition acquisition uses the
same optional preflight discipline, while closing an acquired handle remains
best-effort cleanup.
Uncached map imports request idle/suspend inhibition while parsing and
building the cache. These requests remain best-effort so desktop integration
cannot break the underlying work. Portal
requests use explicit states:

```text
IDLE -> REQUESTING -> WAITING -> {COMPLETED, CANCELLED, FAILED}
```

Startup map sessions accept either a folder containing a supported map or one
direct `.glb`/`.obj` file. This keeps Linux `Exec ... %f` desktop launches and
the in-app folder chooser on the same import/cache path as desktop-shell direct
file launches.
`gui.map_opening` owns the shared directory chooser and selected-folder
resolution used by startup compatibility wrappers and the in-viewer Open
action, so viewer rendering code does not import upward into `caveviewer.app`.

Linux viewer windows use GLFW 3.4. `CAVEVIEWER_WINDOW_SYSTEM=auto` prefers
X11/XWayland when `DISPLAY` is available, then retries Wayland only for a
recognized GLFW initialization/window-creation failure. This keeps source,
debugger, and AppImage launches on the same GNOME window-management path with
normal titlebar and resize decorations. Explicit `wayland` and `x11` modes
never silently switch protocols. The Wayland application ID and X11 window class
both use `io.github.caveviewer.caveviewer`. Initial window geometry is 80% of
GLFW's primary-monitor work area in screen coordinates. Framebuffer DPI scaling
remains enabled, while duplicate X11 monitor scaling of that already-relative
geometry is suppressed during window creation.
OpenGL HUD text is rasterized at framebuffer scale for crispness, while the
always-visible right-side viewer controls use a separate responsive HUD scale
based on the current viewer surface size. That keeps maximized and AppImage
windows legible without requiring user-provided environment variables.
`CaveViewerWindow` is the OpenGL boundary: it owns the context, framebuffer
resources, shader/program calls, and GPU uploads on the render thread. Its
non-GL session ordering is deliberately delegated to focused coordinators.
Each public viewer launcher snapshots its inputs in an immutable
`gui.viewer_session.ViewerSessionConfig`; mutable completion state belongs to
that launch's `ViewerSession`. A short-lived ModernGL configuration subclass
binds the session to the class-based backend API, so the reusable
`CaveViewerWindow` class never carries map, benchmark, runtime, or outcome
state between native-window runs.
`gui.viewer_frame_scheduler` selects the iconified, capture-finalization,
import, startup, or interactive frame phase and owns non-blocking throttles.
`gui.viewer_capture_workflow` owns cross-capture exit finalization and overlay
priority; `gui.viewer_action_dispatch` owns keyboard action priority. These
coordinators call no OpenGL APIs and remain unit-testable without a window
backend.

`gui.recording_capture` owns framebuffer readback resources and staged frame
draining. `gui.recording` owns ffmpeg command construction, encoder
writer/stderr workers, and asynchronous stop finalization.
`gui.recording_controller` owns recording countdowns, transient status messages,
capture timing, and dropped-frame accounting so those workflow decisions remain
testable without constructing an OpenGL window.
`gui.manual_dive_trace_controller` similarly owns manual-trace countdown and
post-save reveal timing; `gui.manual_dive_trace` remains the background JSONL
writer and capture model.

Tk and OpenGL objects are main-thread resources. Background threads may parse,
read, decode, and prepare bytes, but may not mutate widgets or create/release GL
objects.

## User storage

`caveviewer.storage_paths` is the platform-neutral path boundary. Linux uses
the XDG configuration, data, cache, state, and runtime roots; macOS, Windows,
and unsupported platforms currently preserve the historical `~/.caveviewer/`
root until their storage conventions are migrated separately. Preferences
are configuration; remembered chooser locations are state; generated map
caches are rebuildable cache data stored in the source map folder's `_cache`
subdirectory by default. Downloaded map-library entries are ordinary user
downloads by default, stored under the user's Downloads folder or the folder
selected by `CAVEVIEWER_MAP_LIBRARY_DIR` or Preferences, and their generated
caches live inside the downloaded map folder unless an explicit cache-root
override is set.
`CAVEVIEWER_HOME` creates isolated `config/`, `data/`, `cache/`, `state/`, and
`runtime/` children for portable or test runs, but map caches still default to
adjacent `_cache` directories. `CAVEVIEWER_MAP_CACHE_DIR` overrides only the
map-cache root for advanced runs that need generated caches on a separate
filesystem. Relative XDG variables are ignored as required by the specification,
and relative CaveViewer storage-root/cache overrides are rejected.

On Linux, migration from `~/.caveviewer/` and older `~/.caveviewer_*` files is
copy-once and non-destructive. Older app-data `map_library` and `sample_maps`
directories are moved into the configured map-library location when possible.
Explicit-root map cache keys derive from the canonical source path without
reading or hashing a multi-gigabyte map.

The GUI process owns one `caveviewer.gui.update_manager.UpdateManager`, created
by `caveviewer.app` before the splash/viewer session loop and shut down when
that loop exits. Update state is explicit and validated:

```text
IDLE -> CHECKING -> {UP_TO_DATE, AVAILABLE, IDLE on check error}
AVAILABLE -> DOWNLOADING -> VERIFYING -> READY -> HANDOFF_VERIFYING -> INSTALLING -> SHUTDOWN
                |              |          |                  |
                +--------------+-> FAILED -> DOWNLOADING     +-> READY (handoff failure/cancel)
(DOWNLOADING or VERIFYING) -- cancel request --> worker cleanup --> AVAILABLE
any non-SHUTDOWN state -> SHUTDOWN
```

Network, verification, and staging-file work runs in manager-owned workers.
The splash polls immutable snapshots and performs widget updates on the Tk
thread. The viewer and `core.streaming.world` have no update dependency, so
opening a map neither cancels a download nor introduces update UI into the
viewer. A visible splash can request cancellation only while the manager is
downloading or verifying. That request sets the worker's cancellation event;
the Tk thread neither removes files nor changes update state. The worker
honors the request during transfer or hashing, checks it again before package
persistence begins, then removes staging files and returns to `AVAILABLE`.
Previously verified packages remain intact. Closing a splash still does not
cancel a download; process shutdown also requests cancellation and waits for
temporary files to be removed.

The update checker returns one immutable outcome: `UpdateAvailable`,
`UpdateNotAvailable`, or `UpdateCheckFailed`. Only `UpdateAvailable` contains
an `UpdateArtifact`, whose version, HTTPS URL, package kind, positive size, and
SHA-256 were validated before signature verification. After signature
verification, a bounded HEAD request (with a one-byte ranged-GET fallback)
confirms that the advertised package resolves before `UpdateAvailable` can be
returned. A missing preview manifest and a signed candidate whose package is
HTTP 404 or 410 both produce `UpdateNotAvailable`; transient probe failures
produce the quiet `UpdateCheckFailed` path. A Windows EXE also needs
the signed `windows_installer` channel and an explicit installer policy: the
default `verified` policy needs an exact Authenticode certificate subject,
while `unsigned-community` must declare no publisher. `UpdateManager` stores that available outcome only for the
download/retry workflow and passes its non-optional artifact to the worker.
Release notes remain a published manifest field but are not carried through the
manager or splash without a designed UI. The immutable update snapshot carries
focused reveal and, where safe, install-action labels, so the splash does not
consult a broad platform adapter.

Verified packages normally remain manual: Finder mounts macOS DMGs read-only
and reveals the `.app`, Explorer selects a Windows ZIP migration payload, and
Linux asks the desktop portal to reveal its package with a containing-folder
fallback. The sole execution boundary is `WindowsUpdatePackageInstallerAdapter`.
It is available only to a frozen EXE whose exact executable path matches the
per-user Inno Setup provenance marker. After an explicit splash action it
rehashes the private EXE. The default `verified` policy additionally requires a
valid Authenticode chain, exact signed publisher, and RFC-3161 timestamp. The
only unsigned alternative is the signed-manifest `unsigned-community` policy,
which has no declared Authenticode publisher and still requires the manifest
signature, package hash, size, provenance marker, and explicit user action.
It then uses a distinct argument vector to start `CaveViewerSetup.exe /SP-
/SILENT /SUPPRESSMSGBOXES /NORESTART /LOG=... --update --wait-pid <pid>
--expected-version <version>`. The explicit splash action is the user-consent
boundary; Inno Setup retains a visible progress window but requires no further
answer during the normal handoff. Windows-owned trust prompts remain outside
this contract. The installer owns the bounded wait, new-payload verification,
provenance update, and relaunch. A failure returns the manager to `READY`; the
Tk splash closes only after the detached installer process has started.

## Updates and release assets

`updates/` is a published data surface used by installed applications through
raw repository URLs. Its platform and architecture paths are compatibility
contracts. The public verification key under `src/caveviewer/resources/` is
bundled with the application; private signing material must never enter the
repository.

Windows uses `updates/windows/<channel>.json`; EXE manifests additionally bind
the `windows_installer` channel and Authenticode certificate subject into the
Ed25519-signed manifest. Linux distribution is x86_64-only
and uses `updates/linux/x86_64/<channel>.json`. macOS uses architecture-specific
`updates/macos/<arm64|x86_64>/<channel>.json` paths. Every published manifest
has a companion `.sig` file; a platform's absent preview pair represents an
empty channel. Top-level macOS manifests and signatures remain legacy ARM64
aliases whenever that ARM64 channel exists. The update client requires a valid
signature and a resolvable package URL before offering a newer manifest.
Platform build scripts generate
`caveviewer/resources/release_metadata.v1.json` under `build/`, include it in
the frozen payload, and package metadata repeats the selected
`release_channel`. The finalizer validates each package metadata channel before
its shared GitHub write, uploads the selected assets, then verifies GitHub's
reported asset name, HTTPS browser URL, byte size, and SHA-256. Only that
verified API URL enters the signed manifest. A failure leaves the prior
manifest pair unchanged even if an unadvertised release asset was uploaded.

Build, package, publish, and manifest-generation workflows live under
`scripts/`. The PyInstaller contract lives at
`packaging/pyinstaller/CaveViewer.spec`; all build consumers use the installed
package and the same package-resource paths. The four platform workflows may be
run independently. `All Platform Release` runs one shared test gate, packages
all four targets in parallel from one immutable source revision, and hands the
artifacts to a single finalizer. In GitHub Actions, only that finalizer creates
the release, verifies its uploaded assets, signs manifests, and pushes release
metadata, preserving one owner for shared mutable state. Published workflows
run from `release/next`; the finalizer commits only to that branch. Source first
enters protected `main` through a required-check PR, and finalized metadata
returns from `release/next` to `main` through another required-check PR. The
operational contract and verification checklist live in
[releases.md](releases.md).
