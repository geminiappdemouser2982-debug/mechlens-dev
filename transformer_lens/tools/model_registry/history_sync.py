"""Read-only consistency check between supported_models.json and verification_history.json.

``supported_models.json`` holds the latest status per model; ``verification_history.json``
is the append-only audit trail of every pass/fail outcome. They agree when, for every
registry entry with a verified/failed status, the most recent non-invalidated history
record for that model and architecture has the matching result.

Usage:
    python -m transformer_lens.tools.model_registry.history_sync [--verbose]

This tool only reports. It never modifies either data file.
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

from .registry_io import (
    STATUS_FAILED,
    STATUS_VERIFIED,
    load_supported_models_raw,
    load_verification_history,
)
from .verification import VerificationHistory, result_to_status

_TRACKED_STATUSES = (STATUS_VERIFIED, STATUS_FAILED)


@dataclass
class HistoryAuditReport:
    """Mismatches between the registry and the verification history.

    Attributes:
        missing_history: (model_id, arch) registry entries with a pass/fail status but no
            non-invalidated history record.
        status_disagreement: (model_id, arch, registry_status, history_status) entries whose
            latest history record has a different outcome than the registry.
        orphan_records: model_ids with history records but no registry entry at all.
            Informational: a model dropped from the registry keeps its audit trail.
        legacy_records: number of records with no explicit ``result`` field.
    """

    missing_history: list[tuple[str, str]] = field(default_factory=list)
    status_disagreement: list[tuple[str, str, int, int]] = field(default_factory=list)
    orphan_records: list[str] = field(default_factory=list)
    legacy_records: int = 0

    @property
    def is_consistent(self) -> bool:
        """True when every tracked registry entry agrees with its history."""
        return not self.missing_history and not self.status_disagreement

    def summary(self) -> str:
        lines = [
            f"Registry entries with no history record:      {len(self.missing_history)}",
            f"Registry/history status disagreements:        {len(self.status_disagreement)}",
            f"History records for models not in registry:   {len(self.orphan_records)}",
            f"Legacy records without explicit result:       {self.legacy_records}",
        ]
        return "\n".join(lines)


def audit_history(models: list[dict], history: VerificationHistory) -> HistoryAuditReport:
    """Compare registry entries against the verification history.

    Args:
        models: The ``models`` list from supported_models.json
        history: The loaded verification history

    Returns:
        A report of every mismatch found
    """
    report = HistoryAuditReport()
    registry_ids = {m.get("model_id") for m in models if isinstance(m, dict)}

    for entry in models:
        if not isinstance(entry, dict) or entry.get("status") not in _TRACKED_STATUSES:
            continue
        model_id = entry.get("model_id")
        if not model_id:
            continue
        arch = entry.get("architecture_id", "Unknown")
        latest = history.get_record(model_id, arch)
        if latest is None:
            report.missing_history.append((model_id, arch))
            continue
        history_status = result_to_status(latest.effective_result)
        if history_status != entry["status"]:
            report.status_disagreement.append((model_id, arch, entry["status"], history_status))

    seen: set[str] = set()
    for record in history.records:
        if record.result is None:
            report.legacy_records += 1
        if record.model_id not in registry_ids and record.model_id not in seen:
            seen.add(record.model_id)
            report.orphan_records.append(record.model_id)

    return report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns 0 when consistent, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description="Report inconsistencies between the model registry and its history"
    )
    parser.add_argument("--verbose", action="store_true", help="List every mismatch")
    args = parser.parse_args(argv)

    models = load_supported_models_raw().get("models", [])
    report = audit_history(models, load_verification_history())

    print(report.summary())
    if args.verbose:
        for model_id, arch in report.missing_history:
            print(f"  MISSING   {model_id} ({arch})")
        for model_id, arch, reg, hist in report.status_disagreement:
            print(f"  MISMATCH  {model_id} ({arch}): registry={reg} history={hist}")
        for model_id in report.orphan_records:
            print(f"  ORPHAN    {model_id}")

    return 0 if report.is_consistent else 1


if __name__ == "__main__":
    sys.exit(main())
