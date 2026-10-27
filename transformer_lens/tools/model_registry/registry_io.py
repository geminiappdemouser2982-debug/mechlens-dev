"""Shared I/O functions for reading and writing model registry data files.

Consolidates the load-modify-save pattern used by verify_models.py and
main_benchmark.py into a single module that properly uses the
VerificationRecord/VerificationHistory dataclasses.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from .verification import (  # noqa: F401 - status codes re-exported for callers
    RESULT_PASSED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    VALID_RESULTS,
    VerificationHistory,
    VerificationRecord,
    status_to_result,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_SUPPORTED_MODELS_PATH = _DATA_DIR / "supported_models.json"
_VERIFICATION_HISTORY_PATH = _DATA_DIR / "verification_history.json"

# HF-loadable quantization formats. Admitted to the registry; verification gates
# on `required_quant_library_for_model()` at run time.
_HF_LOADABLE_QUANT_PATTERNS = [
    "-awq",
    "_awq",
    "-AWQ",
    "_AWQ",
    "-gptq",
    "_gptq",
    "-GPTQ",
    "_GPTQ",
    "GPTQ",
    "-bnb-",
    "_bnb_",
    "bnb-4bit",
    "bnb-8bit",
    "-4bit",
    "_4bit",
    "-8bit",
    "_8bit",
    "-int4",
    "_int4",
    "-int8",
    "_int8",
    "-w4a16",
    "-w8a8",
    "-W4A16",
    "-W8A8",
    ".w4a16",
    ".W4A16",
    "-hqq",
    "_hqq",
    "-HQQ",
    "_HQQ",
    "-3bit",
    "_3bit",
    "-2bit",
    "_2bit",
    "-5bit",
    "-6bit",
    "-oQ",
    "_oQ",
    "-quantized.",
    "_Quantized",
    "-Quantized",
]

# Formats that need a non-HF loader (GGUF→llama.cpp, MLX→Apple, FP4/FP8→NVIDIA).
_INCOMPATIBLE_QUANT_PATTERNS = [
    "-gguf",
    "_gguf",
    "-GGUF",
    "_GGUF",
    "mlx-community/",
    "-mlx",
    "-MLX",
    "_mlx",
    "_MLX",
    ".mlx",
    ".MLX",
    "-fp8",
    "_fp8",
    "-FP8",
    "_FP8",
    "-nvfp4",
    "_nvfp4",
    "-NVFP4",
    "_NVFP4",
    "-mxfp4",
    "_mxfp4",
    "-MXFP4",
    "_MXFP4",
]

# Values are Python import names, not PyPI package names. Order matters: explicit
# format markers must precede generic bit-width markers (HQQ-4bit IDs match both).
_QUANT_LIBRARY_BY_PATTERN: list[tuple[tuple[str, ...], str]] = [
    (("-hqq", "_hqq", "-HQQ", "_HQQ"), "hqq"),
    (("-gptq", "_gptq", "-GPTQ", "_GPTQ", "GPTQ"), "auto_gptq"),
    (("-awq", "_awq", "-AWQ", "_AWQ"), "awq"),
    (("-w4a16", "-w8a8", "-W4A16", "-W8A8", ".w4a16", ".W4A16"), "auto_gptq"),
    (("-bnb-", "_bnb_", "bnb-4bit", "bnb-8bit"), "bitsandbytes"),
    (("-4bit", "_4bit", "-8bit", "_8bit", "-int4", "_int4", "-int8", "_int8"), "bitsandbytes"),
]

QUANTIZED_NOTE = "Quantized format not loadable by HF transformers"


def is_incompatible_quantized(model_id: str) -> bool:
    """True for quantization formats the bridge can't ingest (GGUF, MLX, FP4/FP8)."""
    return any(pat in model_id for pat in _INCOMPATIBLE_QUANT_PATTERNS)


def is_hf_loadable_quantized(model_id: str) -> bool:
    """True for quantizations loadable by HF transformers + a quant library."""
    return any(pat in model_id for pat in _HF_LOADABLE_QUANT_PATTERNS)


def required_quant_library_for_model(model_id: str) -> Optional[str]:
    """Return the Python import name needed to load this model, or None if unquantized."""
    for patterns, library in _QUANT_LIBRARY_BY_PATTERN:
        if any(pat in model_id for pat in patterns):
            return library
    return None


def is_quantized_model(model_id: str) -> bool:
    """Alias for ``is_incompatible_quantized`` — kept for back-compat with existing call sites."""
    return is_incompatible_quantized(model_id)


def load_supported_models_raw() -> dict:
    """Load supported_models.json as a raw dict."""
    with open(_SUPPORTED_MODELS_PATH) as f:
        return json.load(f)


def save_supported_models_raw(data: dict) -> None:
    """Save raw dict back to supported_models.json."""
    with open(_SUPPORTED_MODELS_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load_verification_history() -> VerificationHistory:
    """Load verification_history.json into a VerificationHistory dataclass."""
    if _VERIFICATION_HISTORY_PATH.exists():
        with open(_VERIFICATION_HISTORY_PATH) as f:
            data = json.load(f)
        return VerificationHistory.from_dict(data)
    return VerificationHistory()


def save_verification_history(history: VerificationHistory) -> None:
    """Save VerificationHistory dataclass to verification_history.json."""
    with open(_VERIFICATION_HISTORY_PATH, "w") as f:
        json.dump(history.to_dict(), f, indent=2)
        f.write("\n")


def _get_tl_version() -> Optional[str]:
    """Get the installed TransformerLens version, or None."""
    try:
        import transformer_lens

        version = getattr(transformer_lens, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("transformer-lens")
    except Exception:
        return None


def update_model_status(
    model_id: str,
    arch_id: str,
    status: Optional[int] = None,
    note: Optional[str] = None,
    phase_scores: Optional[dict[int, Optional[float]]] = None,
    sanitize_fn: Optional[Callable[[Optional[str]], Optional[str]]] = None,
) -> bool:
    """Update a single model entry in supported_models.json.

    If the model is not found in the registry and status == STATUS_VERIFIED,
    a new entry is appended.

    When status is None (partial-phase update), only the provided phase_scores
    are updated — status, note, and other scores are preserved.

    Args:
        model_id: The model to update
        arch_id: Architecture of the model
        status: New status code (0-3), or None for score-only updates
        note: Optional note for skip/fail reason
        phase_scores: Phase score dict {1: float, 2: float, 3: float, 4: float}
        sanitize_fn: Optional callable to sanitize note strings

    Returns:
        True if entry was found/created and updated
    """
    if phase_scores is None:
        phase_scores = {}

    if sanitize_fn and note:
        note = sanitize_fn(note)

    data = load_supported_models_raw()
    updated = False

    for entry in data.get("models", []):
        if entry["model_id"] == model_id and entry["architecture_id"] == arch_id:
            if status is not None:
                entry["status"] = status
                entry["verified_date"] = (
                    date.today().isoformat() if status != STATUS_UNVERIFIED else None
                )
                entry["note"] = note
            elif note is not None:
                # Score-only update with an explicit note — overwrite stale notes
                entry["note"] = note
            elif phase_scores and "exceeds" in (entry.get("note") or "").lower():
                # Writing real scores clears a stale memory-skip note
                entry["note"] = None
            for phase_num in (1, 2, 3, 4, 7, 8):
                key = f"phase{phase_num}_score"
                if phase_num in phase_scores:
                    entry[key] = phase_scores[phase_num]
                elif key not in entry:
                    entry[key] = None
            # Reorder keys so phase scores are always in numerical order
            _KEY_ORDER = [
                "architecture_id",
                "model_id",
                "status",
                "verified_date",
                "metadata",
                "note",
                "phase1_score",
                "phase2_score",
                "phase3_score",
                "phase4_score",
                "phase7_score",
                "phase8_score",
            ]
            reordered = {k: entry[k] for k in _KEY_ORDER if k in entry}
            for k in entry:
                if k not in reordered:
                    reordered[k] = entry[k]
            entry.clear()
            entry.update(reordered)
            updated = True
            break

    if not updated and status == STATUS_VERIFIED:
        # Model not in registry -- add it
        data.get("models", []).append(
            {
                "model_id": model_id,
                "architecture_id": arch_id,
                "status": status,
                "verified_date": date.today().isoformat(),
                "metadata": None,
                "note": note,
                "phase1_score": phase_scores.get(1),
                "phase2_score": phase_scores.get(2),
                "phase3_score": phase_scores.get(3),
                "phase4_score": phase_scores.get(4),
                "phase7_score": phase_scores.get(7),
                "phase8_score": phase_scores.get(8),
            }
        )
        updated = True

    if updated:
        models = data.get("models", [])
        data["total_verified"] = sum(1 for m in models if m.get("status", 0) == STATUS_VERIFIED)
        data["total_models"] = len(models)
        data["total_architectures"] = len(set(m["architecture_id"] for m in models))
        save_supported_models_raw(data)

    return updated


def add_verification_record(
    model_id: str,
    arch_id: str,
    notes: Optional[str] = None,
    verified_by: str = "verify_models",
    sanitize_fn: Optional[Callable[[Optional[str]], Optional[str]]] = None,
    result: str = RESULT_PASSED,
) -> None:
    """Append a VerificationRecord to verification_history.json.

    Args:
        model_id: The model that was tested
        arch_id: Architecture type
        notes: Optional verification notes
        verified_by: Who/what performed the verification
        sanitize_fn: Optional callable to sanitize note strings
        result: ``"passed"`` or ``"failed"``. The history is an audit trail of
            attempts, so failures are recorded too; the result is what lets readers
            tell them apart.
    """
    if result not in VALID_RESULTS:
        raise ValueError(f"result must be one of {VALID_RESULTS}, got {result!r}")
    if sanitize_fn and notes:
        notes = sanitize_fn(notes)

    record = VerificationRecord(
        model_id=model_id,
        architecture_id=arch_id,
        verified_date=date.today(),
        verified_by=verified_by,
        transformerlens_version=_get_tl_version(),
        notes=notes,
        result=result,
    )

    history = load_verification_history()
    history.add_record(record)
    save_verification_history(history)


def record_verification_result(
    model_id: str,
    arch_id: str,
    status: int,
    note: Optional[str] = None,
    phase_scores: Optional[dict[int, Optional[float]]] = None,
    verified_by: str = "verify_models",
    sanitize_fn: Optional[Callable[[Optional[str]], Optional[str]]] = None,
) -> bool:
    """Record a pass/fail verification outcome in both registry files.

    This is the single write path for a completed verification: it updates the
    model's entry in supported_models.json and appends a matching record to
    verification_history.json so the two files cannot drift apart. The history
    record is only written when the registry entry was actually updated; a failure
    for a model that is not in the registry is logged and dropped rather than
    leaving an orphan history record.

    Args:
        model_id: The model that was tested
        arch_id: Architecture of the model
        status: STATUS_VERIFIED or STATUS_FAILED
        note: Note describing the outcome
        phase_scores: Phase score dict to write to the registry entry
        verified_by: Who/what performed the verification
        sanitize_fn: Optional callable to sanitize note strings

    Returns:
        True if the registry entry was updated (and the history record written)
    """
    # Raises ValueError for anything other than VERIFIED / FAILED.
    result = status_to_result(status)

    updated = update_model_status(
        model_id,
        arch_id,
        status=status,
        note=note,
        phase_scores=phase_scores,
        sanitize_fn=sanitize_fn,
    )
    if not updated:
        logger.warning(
            "Not recording %s result for %s (%s): no registry entry to update",
            result,
            model_id,
            arch_id,
        )
        return False

    add_verification_record(
        model_id,
        arch_id,
        notes=note,
        verified_by=verified_by,
        sanitize_fn=sanitize_fn,
        result=result,
    )
    return True
