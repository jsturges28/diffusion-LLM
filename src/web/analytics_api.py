"""HTTP routes and response assembly for saved-run Analytics.

The saved-run reader remains in ``src.analytics.metrics``. This
module owns the HTTP boundary above it: route registration, response
shapes, error translation, model-aware labels, and delete handling.

It deliberately does not import ``server``. The supervisor supplies
the current results root and host probe through frozen dependencies,
so a test or embedding can replace either without rebuilding this
module or creating a second source of truth.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.analytics.metrics import (
    CONVERGENCE_BASIS_CHARACTERS,
    CONVERGENCE_BASIS_SETTLEMENT,
    CONVERGENCE_BASIS_TOKENS,
    UnsupportedRunVersionError,
    canvas_boundaries,
    compute_convergence,
    convergence_from_positions,
    convergence_from_records,
    convergence_from_settlement,
    list_runs,
    load_run_frames,
    load_run_metadata,
    masks_are_real,
    read_frame_texts,
    records_match_frames,
    run_schema_version,
    tokens_produced_series,
    total_elapsed_seconds,
)
from src.backends.params import ParamValue, coerce, default_of
from src.backends.protocol import (
    SAVED_MODEL_TYPE_AUTOREGRESSIVE,
    ParamSpec,
)
from src.backends.registry import REGISTRY
from src.inference.kgw_watermark import (
    KGW_DISPLAY_Z_THRESHOLD_DEFAULT,
    detection_status,
    detection_z_score,
)
from src.web import run_store


logger = logging.getLogger("diffusion_supervisor")

ResultsDirReader = Callable[[], Path]
StringProbe = Callable[[], Optional[str]]


@dataclass(frozen=True)
class AnalyticsApiDependencies:
    """Supervisor-owned values the Analytics API may read."""

    results_dir: ResultsDirReader
    repo_root: Path
    gpu_name: StringProbe

    def __post_init__(self) -> None:
        assert callable(self.results_dir)
        assert isinstance(self.repo_root, Path)
        assert callable(self.gpu_name)


# How many runs one comparison may carry. The chart is a legend and
# a handful of lines; past this it is unreadable before it is slow.
COMPARE_RUNS_MAX = 12

# Parameters a legend label may name before it stops being a label.
COMPARE_LABEL_PARAMS_MAX = 3

# What became of one selection. Every id gets exactly one of these.
COMPARE_STATUS_DATA = "data"
COMPARE_STATUS_UNAVAILABLE = "unavailable"
COMPARE_STATUS_ERROR = "error"

# Why a selection carries no data. Separate from the message so the
# browser can group or style them without matching on prose.
COMPARE_NOT_FOUND = "not_found"
COMPARE_INVALID_ID = "invalid_id"
COMPARE_UNSUPPORTED = "unsupported_version"
COMPARE_UNREADABLE = "unreadable"
COMPARE_NO_CURVE = "no_curve"

COMPARE_REASONS = (
    COMPARE_NOT_FOUND,
    COMPARE_INVALID_ID,
    COMPARE_UNSUPPORTED,
    COMPARE_UNREADABLE,
    COMPARE_NO_CURVE,
)

# Parameters a model that stops adaptively declares. The step budget
# travels with the thresholds because the readout compares against it.
STOP_RULE_PARAMS: Tuple[str, ...] = (
    "confidence_threshold",
    "stability_threshold",
    "max_denoising_steps",
)
WATERMARK_THRESHOLD_PARAM = "watermark_z_threshold"

assert COMPARE_RUNS_MAX > 1, "a comparison needs two runs"
assert len(set(COMPARE_REASONS)) == len(COMPARE_REASONS)
assert len(STOP_RULE_PARAMS) == 3, "the rule has two gates and a cap"


@dataclass(frozen=True)
class AnalyticsApi:
    """Bound Analytics handlers registered by the router factory."""

    dependencies: AnalyticsApiDependencies

    def _results_dir(self) -> Path:
        results_dir = self.dependencies.results_dir()
        assert isinstance(results_dir, Path)
        return results_dir

    async def analytics_list_runs(self) -> JSONResponse:
        results_dir = self._results_dir()
        runs = await asyncio.to_thread(list_runs, results_dir)
        return JSONResponse(content=runs)

    async def analytics_run_metrics(
        self, run_id: str
    ) -> JSONResponse:
        results_dir = self._results_dir()
        try:
            result = await asyncio.to_thread(
                _compute_run_metrics, results_dir, run_id
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=404, content={"error": str(exc)}
            )
        except UnsupportedRunVersionError as exc:
            return _unsupported_version_response(exc)
        except ValueError as exc:
            return JSONResponse(
                status_code=400, content={"error": str(exc)}
            )
        return JSONResponse(content=result)

    async def analytics_run_metadata(
        self, run_id: str
    ) -> JSONResponse:
        """Return full metadata for the one run being opened."""
        results_dir = self._results_dir()
        try:
            metadata = await asyncio.to_thread(
                _run_metadata, results_dir, run_id
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=404, content={"error": str(exc)}
            )
        except UnsupportedRunVersionError as exc:
            return _unsupported_version_response(exc)
        except ValueError as exc:
            return JSONResponse(
                status_code=400, content={"error": str(exc)}
            )
        return JSONResponse(content=metadata)

    async def analytics_run_frames(self, run_id: str) -> JSONResponse:
        results_dir = self._results_dir()
        try:
            result = await asyncio.to_thread(
                _compute_run_frames, results_dir, run_id
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=404, content={"error": str(exc)}
            )
        except UnsupportedRunVersionError as exc:
            return _unsupported_version_response(exc)
        except ValueError as exc:
            return JSONResponse(
                status_code=400, content={"error": str(exc)}
            )
        return JSONResponse(content=result)

    async def analytics_compare(self, ids: str = "") -> JSONResponse:
        """Compare a bounded set, accounting for every selection."""
        run_ids = _compare_selection(ids)
        if len(run_ids) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "ids parameter is required"},
            )
        if len(run_ids) > COMPARE_RUNS_MAX:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Compare accepts up to {COMPARE_RUNS_MAX}"
                        f" runs; {len(run_ids)} were selected."
                    )
                },
            )

        results_dir = self._results_dir()
        results = [
            await _compare_one(results_dir=results_dir, run_id=run_id)
            for run_id in run_ids
        ]
        assert len(results) == len(run_ids), "one record per id"
        return JSONResponse(content=results)

    async def analytics_system_info(self) -> JSONResponse:
        """Return host and data-root facts used by Analytics."""
        results_dir = self._results_dir()
        return JSONResponse(
            content={
                "gpu_name": self.dependencies.gpu_name(),
                "results_dir": run_store.display_path(
                    results_dir, self.dependencies.repo_root
                ),
            }
        )

    async def analytics_delete_run(self, run_id: str) -> JSONResponse:
        results_dir = self._results_dir()
        try:
            await asyncio.to_thread(
                _delete_run_blocking, results_dir, run_id
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": str(exc),
                },
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": str(exc),
                },
            )
        except OSError as exc:
            logger.exception("failed to delete run %s", run_id)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": str(exc),
                },
            )
        logger.info("deleted run %s", run_id)
        return JSONResponse(content={"success": True})


def create_analytics_router(
    dependencies: AnalyticsApiDependencies,
) -> APIRouter:
    """Build the Analytics router around narrow live dependencies."""
    assert isinstance(dependencies, AnalyticsApiDependencies)

    api = AnalyticsApi(dependencies)
    router = APIRouter()
    router.add_api_route(
        "/api/analytics/runs",
        api.analytics_list_runs,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/analytics/runs/{run_id}/metrics",
        api.analytics_run_metrics,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/analytics/runs/{run_id}/metadata",
        api.analytics_run_metadata,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/analytics/runs/{run_id}/frames",
        api.analytics_run_frames,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/analytics/compare",
        api.analytics_compare,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/analytics/system",
        api.analytics_system_info,
        methods=["GET"],
    )
    router.add_api_route(
        "/api/analytics/runs/{run_id}",
        api.analytics_delete_run,
        methods=["DELETE"],
    )
    assert len(router.routes) == 7, "all Analytics routes registered"
    return router


def _compute_run_metrics(
    results_dir: Path, run_id: str
) -> Dict[str, Any]:
    """Build the metrics payload for one guarded run."""
    run_dir = run_store.resolve_run_dir(results_dir, run_id)
    metadata = load_run_metadata(run_dir)
    frames = read_frame_texts(run_dir, metadata)
    convergence, basis, produced_from = _run_convergence(
        run_dir, frames, metadata.get("canvas_index")
    )

    result: Dict[str, Any] = {
        "run_id": run_id,
        "convergence": convergence,
        "convergence_basis": basis,
        "total_frames": len(frames),
        "model_type": str(metadata.get("model_type", "diffusion")),
        "model_label": _model_label(metadata),
    }
    for key in (
        "per_frame_elapsed",
        "elapsed_seconds",
        "remask_edits",
        "mean_conf",
        "original_per_frame_elapsed",
        "original_elapsed_seconds",
        "original_mean_conf",
    ):
        if key in metadata:
            result[key] = metadata[key]

    repaired = total_elapsed_seconds(
        metadata.get("per_frame_elapsed")
    )
    if repaired is not None:
        result["elapsed_seconds"] = repaired
    canvas_index = metadata.get("canvas_index")
    if canvas_index:
        result["canvas_boundaries"] = canvas_boundaries(canvas_index)
    result["tokens_produced"] = tokens_produced_series(
        produced_from, canvas_index
    )
    return result


def _model_label(metadata: Dict[str, Any]) -> str:
    """Return the display name of the model that made a run."""
    backend = str(metadata.get("backend", ""))
    entry = REGISTRY.get(backend)
    if entry is None:
        return backend
    return entry.display_name


def _run_convergence(
    run_dir: Path,
    frames: List[str],
    canvas_index: Any = None,
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """Choose a run's honest curve and throughput source.

    A malformed token stream falls back to character counting rather
    than taking down the page. The returned basis tells the page that
    the weaker measure was used.
    """
    try:
        loaded = load_run_frames(run_dir)
    except (ValueError, OSError):
        logger.warning(
            "token records unreadable for %s; counting characters",
            run_dir.name,
        )
        loaded = None

    if loaded is not None and loaded.get("records_available"):
        positions = loaded.get("positions")
        if positions is not None and len(positions) == len(frames):
            by_count = convergence_from_positions(len(positions))
            return (
                by_count,
                CONVERGENCE_BASIS_TOKENS,
                by_count,
            )
        token_frames = loaded.get("frames")
        if records_match_frames(token_frames, len(frames)):
            by_mask = convergence_from_records(token_frames)
            if masks_are_real(token_frames):
                return (
                    by_mask,
                    CONVERGENCE_BASIS_TOKENS,
                    by_mask,
                )
            return (
                convergence_from_settlement(
                    token_frames, canvas_index
                ),
                CONVERGENCE_BASIS_SETTLEMENT,
                by_mask,
            )

    by_characters = compute_convergence(frames)
    return (
        by_characters,
        CONVERGENCE_BASIS_CHARACTERS,
        by_characters,
    )


def _unsupported_version_response(
    exc: UnsupportedRunVersionError,
) -> JSONResponse:
    """Translate a future run into the update-specific response."""
    return JSONResponse(
        status_code=400,
        content={
            "error": (
                "This run was saved by a newer version of the app"
                f" (format {exc.version}), which this build cannot"
                " read. Update to open it."
            ),
            "unsupported_version": True,
        },
    )


def _run_metadata(results_dir: Path, run_id: str) -> Dict[str, Any]:
    """Build one guarded run's complete metadata payload."""
    run_dir = run_store.resolve_run_dir(results_dir, run_id)
    metadata = load_run_metadata(run_dir)
    run_schema_version(metadata)
    metadata["has_diff"] = (
        run_dir / "original_tokens.json"
    ).is_file()
    repaired = total_elapsed_seconds(
        metadata.get("per_frame_elapsed")
    )
    if repaired is not None:
        metadata["elapsed_seconds"] = repaired
    return metadata


def _compute_run_frames(
    results_dir: Path, run_id: str
) -> Dict[str, Any]:
    """Build the durable token-stream payload for one run."""
    run_dir = run_store.resolve_run_dir(results_dir, run_id)
    metadata = load_run_metadata(run_dir)
    data = load_run_frames(run_dir)
    positions = data["positions"]
    original_positions = data["original_positions"]
    watermark = _watermark_payload(metadata, data)
    return {
        "run_id": run_id,
        "frames": None if positions is not None else data["frames"],
        "positions": positions,
        "original_frames": (
            None
            if original_positions is not None
            else data["original_frames"]
        ),
        "original_positions": original_positions,
        "records_available": data["records_available"],
        "alternatives": data["alternatives"],
        "alternatives_available": data["alternatives_available"],
        "original_alternatives": data["original_alternatives"],
        "sampler_alternatives": data["sampler_alternatives"],
        "sampler_alternatives_available": (
            data["sampler_alternatives_available"]
        ),
        "original_sampler_alternatives": (
            data["original_sampler_alternatives"]
        ),
        "candidates": data["candidates"],
        "original_candidates": data["original_candidates"],
        "remask_edits": metadata.get("remask_edits", []),
        "canvas_index": metadata.get("canvas_index"),
        "stop_rule": _stop_rule(metadata),
        "signals": metadata.get(run_store.SIGNALS_KEY),
        "watermark": watermark,
        "watermark_display_threshold": (
            _watermark_display_threshold(metadata)
            if watermark is not None
            else None
        ),
    }


def _watermark_payload(
    metadata: Dict[str, Any], data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Pair worker attestation with a token-record recomputation."""
    attested = metadata.get("watermark")
    if not isinstance(attested, dict):
        return None
    described = dict(attested)
    model_id = metadata.get("backend")
    if isinstance(model_id, str) and model_id:
        described.setdefault("model_id", model_id)
    payload: Dict[str, Any] = {
        "attested": described,
        "recomputed": None,
        "record_consistency": "unavailable",
    }
    records = _watermark_final_records(data)
    p0 = attested.get("p0")
    if not records or not _valid_probability(p0):
        return payload
    if not all(
        isinstance(record.get("g"), bool)
        and isinstance(record.get("we"), bool)
        for record in records
    ):
        return payload
    scored = sum(record["we"] is True for record in records)
    green = sum(
        record["we"] is True and record["g"] is True
        for record in records
    )
    exact_p0 = float(p0)
    recomputed = {
        "status": detection_status(scored),
        "green_count": green,
        "scored_count": scored,
        "green_rate": green / scored if scored > 0 else 0.0,
        "z_score": detection_z_score(
            green_count=green,
            scored_count=scored,
            p0=exact_p0,
        ),
        "p0": exact_p0,
    }
    payload["recomputed"] = recomputed
    payload["record_consistency"] = (
        "consistent"
        if _watermark_scores_match(attested, recomputed)
        else "mismatch"
    )
    return payload


def _watermark_final_records(
    data: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """The saved run's final current layer, never its baseline."""
    positions = data.get("positions")
    if isinstance(positions, list):
        return positions
    frames = data.get("frames")
    if not isinstance(frames, list):
        return None
    for frame in reversed(frames):
        if isinstance(frame, list):
            return frame
    return None


def _valid_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 < float(value) < 1.0
    )


def _watermark_scores_match(
    attested: Dict[str, Any], recomputed: Dict[str, Any]
) -> bool:
    """Whether counts and score agree across worker and records."""
    if attested.get("green_count") != recomputed["green_count"]:
        return False
    if attested.get("scored_count") != recomputed["scored_count"]:
        return False
    if attested.get("status") != recomputed["status"]:
        return False
    if not _watermark_rate_matches(attested, recomputed):
        return False
    score = attested.get("z_score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return False
    return math.isclose(
        float(score),
        float(recomputed["z_score"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _watermark_rate_matches(
    attested: Dict[str, Any], recomputed: Dict[str, Any]
) -> bool:
    """Match an optional new rate while accepting old attestations."""
    rate = attested.get("green_rate")
    if rate is None:
        return True
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return False
    return math.isclose(
        float(rate),
        float(recomputed["green_rate"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _watermark_display_threshold(
    metadata: Dict[str, Any],
) -> float:
    """The saved display choice, or its stable default."""
    params = metadata.get("params")
    value = (
        params.get(WATERMARK_THRESHOLD_PARAM)
        if isinstance(params, dict)
        else None
    )
    entry = REGISTRY.get(str(metadata.get("backend", "")))
    if entry is not None:
        for spec in entry.param_specs:
            if spec.name == WATERMARK_THRESHOLD_PARAM:
                coerced = _watermark_threshold_value(spec, value)
                return float(coerced)
    if _valid_threshold(value):
        return float(value)
    return KGW_DISPLAY_Z_THRESHOLD_DEFAULT


def _watermark_threshold_value(
    spec: ParamSpec, value: Any
) -> ParamValue:
    """Validate a saved threshold through its registry spec."""
    default = default_of(spec, device=None)
    if value is None:
        return default
    try:
        return coerce(spec, value, device=None, experimental=True)
    except ValueError:
        return default


def _valid_threshold(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _stop_rule(
    metadata: Dict[str, Any],
) -> Optional[Dict[str, ParamValue]]:
    """Return the stopping rule a saved adaptive run used."""
    entry = REGISTRY.get(str(metadata.get("backend", "")))
    if entry is None:
        return None
    if not entry.capabilities.adaptive_stopping:
        return None

    saved = metadata.get("params")
    if not isinstance(saved, dict):
        saved = {}
    specs = {spec.name: spec for spec in entry.param_specs}
    rule: Dict[str, ParamValue] = {}
    for name in STOP_RULE_PARAMS:
        assert name in specs, (
            f"{entry.id} stops adaptively without {name}"
        )
        rule[name] = _stop_rule_value(specs[name], saved.get(name))
    return rule


def _stop_rule_value(spec: ParamSpec, given: Any) -> ParamValue:
    """Return one valid saved rule value, or its registry default."""
    default = default_of(spec, device=None)
    if given is None:
        return default
    try:
        return coerce(spec, given, device=None, experimental=True)
    except ValueError:
        return default


def _compare_selection(ids: str) -> List[str]:
    """Return trimmed, unique ids in first-occurrence order."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for raw in ids.split(","):
        run_id = raw.strip()
        if not run_id:
            continue
        if run_id in seen:
            continue
        seen.add(run_id)
        ordered.append(run_id)
    return ordered


async def _compare_one(
    *, results_dir: Path, run_id: str
) -> Dict[str, Any]:
    """Return one comparison selection's explicit outcome."""
    try:
        record = await asyncio.to_thread(
            _compute_run_metrics, results_dir, run_id
        )
    except run_store.RunNotFoundError:
        return _compare_error(
            run_id=run_id,
            reason=COMPARE_NOT_FOUND,
            message="This run no longer exists.",
        )
    except run_store.InvalidRunIdError:
        return _compare_error(
            run_id=run_id,
            reason=COMPARE_INVALID_ID,
            message="Not a valid run id.",
        )
    except UnsupportedRunVersionError:
        return _compare_error(
            run_id=run_id,
            reason=COMPARE_UNSUPPORTED,
            message="Saved by a newer version of this app.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("compare failed for %s", run_id)
        return _compare_error(
            run_id=run_id,
            reason=COMPARE_UNREADABLE,
            message=f"Could not be read: {exc}",
        )

    record["status"] = COMPARE_STATUS_DATA
    record["label"] = _compare_label(results_dir, run_id)
    if record.get("model_type") == SAVED_MODEL_TYPE_AUTOREGRESSIVE:
        record["status"] = COMPARE_STATUS_UNAVAILABLE
        record["reason"] = COMPARE_NO_CURVE
        record["message"] = (
            "Autoregressive runs have no convergence curve."
        )
    return record


def _compare_error(
    *, run_id: str, reason: str, message: str
) -> Dict[str, Any]:
    """Build one refused selection for the chart legend."""
    assert reason in COMPARE_REASONS, reason
    return {
        "run_id": run_id,
        "status": COMPARE_STATUS_ERROR,
        "reason": reason,
        "message": message,
        "label": run_id,
    }


def _compare_label(results_dir: Path, run_id: str) -> str:
    """Build a legend label from a model's declared parameters."""
    try:
        run_dir = run_store.resolve_run_dir(results_dir, run_id)
        metadata = load_run_metadata(run_dir)
    except (ValueError, OSError):
        return run_id

    entry = REGISTRY.get(str(metadata.get("backend", "")))
    if entry is None:
        return run_id
    params = metadata.get("params")
    if not isinstance(params, dict):
        return entry.display_name

    parts: List[str] = []
    for spec in entry.param_specs:
        if len(parts) >= COMPARE_LABEL_PARAMS_MAX:
            break
        if spec.name not in params:
            continue
        parts.append(f"{spec.label}={params[spec.name]}")
    if not parts:
        return entry.display_name
    return entry.display_name + " " + " ".join(parts)


def _delete_run_blocking(results_dir: Path, run_id: str) -> None:
    """Delete one saved run directory under the current data root."""
    run_store.delete(results_dir, run_id)
