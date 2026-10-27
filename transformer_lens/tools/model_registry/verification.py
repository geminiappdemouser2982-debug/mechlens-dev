"""Verification tracking for model compatibility.

This module provides dataclasses and utilities for tracking which models
have been verified to work with TransformerLens.

``verification_history.json`` is an append-only audit trail: every
``verify_models`` outcome (pass or fail) appends one :class:`VerificationRecord`.
``supported_models.json`` holds only the *latest* status per model, so the two
files agree when the most recent non-invalidated record for a model carries the
same status as the registry entry.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

# Status codes shared with supported_models.json (re-exported by registry_io).
STATUS_UNVERIFIED = 0
STATUS_VERIFIED = 1
STATUS_SKIPPED = 2
STATUS_FAILED = 3

# Outcome values stored in a VerificationRecord's ``result`` field.
RESULT_PASSED = "passed"
RESULT_FAILED = "failed"
VALID_RESULTS = (RESULT_PASSED, RESULT_FAILED)

# Note prefixes written by verify_models / main_benchmark before records carried
# an explicit ``result``. Used only to interpret legacy records.
_LEGACY_FAILURE_PREFIXES: tuple[str, ...] = (
    "Below threshold",
    "Below ",
    "CORE FAILED",
    "Core verification failed",
    "Process crash",
    "OOM",
    "CUDA error",
    "Worker timed out",
    "cannot access local variable",
    "Benchmark failed",
    "Config unavailable",
    "Skipped",
)


def infer_legacy_result(notes: Optional[str]) -> str:
    """Infer the outcome of a record written before ``result`` existed.

    Legacy records were appended for both passing and failing runs with no result
    field, so the only signal is the note text. Anything that does not match a known
    failure prefix is treated as a pass, which matches how such records were read
    historically.
    """
    text = (notes or "").lstrip()
    if any(text.startswith(prefix) for prefix in _LEGACY_FAILURE_PREFIXES):
        return RESULT_FAILED
    return RESULT_PASSED


def result_to_status(result: str) -> int:
    """Map a record outcome to the matching supported_models.json status code."""
    return STATUS_VERIFIED if result == RESULT_PASSED else STATUS_FAILED


def status_to_result(status: int) -> str:
    """Map a registry status code to a record outcome.

    Only STATUS_VERIFIED and STATUS_FAILED describe a completed verification.
    """
    if status == STATUS_VERIFIED:
        return RESULT_PASSED
    if status == STATUS_FAILED:
        return RESULT_FAILED
    raise ValueError(f"No verification result for registry status {status}")


@dataclass
class VerificationRecord:
    """A record of a model verification attempt.

    Attributes:
        model_id: The HuggingFace model ID that was verified
        architecture_id: The architecture type of the model
        verified_date: Date when verification was performed
        verified_by: Who performed the verification (user, CI, etc.)
        transformerlens_version: Version of TransformerLens used
        notes: Optional notes about the verification
        result: Outcome, ``"passed"`` or ``"failed"``. ``None`` only on records written
            before this field existed; see :meth:`effective_result`.
        invalidated: Whether this verification has been invalidated
        invalidation_reason: Reason for invalidation if applicable
    """

    model_id: str
    verified_date: date
    architecture_id: str = "Unknown"
    verified_by: Optional[str] = None
    transformerlens_version: Optional[str] = None
    notes: Optional[str] = None
    result: Optional[str] = None
    invalidated: bool = False
    invalidation_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.result is not None and self.result not in VALID_RESULTS:
            raise ValueError(f"result must be one of {VALID_RESULTS}, got {self.result!r}")

    @property
    def effective_result(self) -> str:
        """Outcome, inferring it from the note for legacy records."""
        if self.result is not None:
            return self.result
        return infer_legacy_result(self.notes)

    @property
    def is_pass(self) -> bool:
        """True if this record documents a successful verification."""
        return self.effective_result == RESULT_PASSED

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary.

        ``result`` is omitted for legacy records (``None``) so re-saving the history
        leaves records written before the field existed byte-for-byte unchanged.
        """
        data: dict = {
            "model_id": self.model_id,
            "architecture_id": self.architecture_id,
        }
        if self.result is not None:
            data["result"] = self.result
        data.update(
            {
                "verified_date": self.verified_date.isoformat(),
                "verified_by": self.verified_by,
                "transformerlens_version": self.transformerlens_version,
                "notes": self.notes,
                "invalidated": self.invalidated,
                "invalidation_reason": self.invalidation_reason,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationRecord":
        """Create from a dictionary."""
        return cls(
            model_id=data["model_id"],
            architecture_id=data.get("architecture_id", "Unknown"),
            verified_date=date.fromisoformat(data["verified_date"]),
            verified_by=data.get("verified_by"),
            transformerlens_version=data.get("transformerlens_version"),
            notes=data.get("notes"),
            result=data.get("result"),
            invalidated=data.get("invalidated", False),
            invalidation_reason=data.get("invalidation_reason"),
        )


@dataclass
class VerificationHistory:
    """History of all model verification attempts.

    Attributes:
        records: List of all verification records
        last_updated: When this history was last updated
    """

    records: list[VerificationRecord] = field(default_factory=list)
    last_updated: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationHistory":
        """Create from a dictionary."""
        last_updated = None
        if data.get("last_updated"):
            last_updated = datetime.fromisoformat(data["last_updated"])
        return cls(
            records=[VerificationRecord.from_dict(r) for r in data.get("records", [])],
            last_updated=last_updated,
        )

    def get_record(
        self, model_id: str, architecture_id: Optional[str] = None
    ) -> Optional[VerificationRecord]:
        """Get the most recent non-invalidated record for a model.

        The returned record may document a failure; check :attr:`VerificationRecord.is_pass`
        or use :meth:`is_verified`.

        Args:
            model_id: The model ID to look up
            architecture_id: If given, only records for this architecture match

        Returns:
            The verification record, or None if not found or invalidated
        """
        for record in reversed(self.records):
            if record.model_id != model_id or record.invalidated:
                continue
            if architecture_id is not None and record.architecture_id != architecture_id:
                continue
            return record
        return None

    def is_verified(self, model_id: str, architecture_id: Optional[str] = None) -> bool:
        """Check if a model's most recent valid record is a successful verification.

        Args:
            model_id: The model ID to check
            architecture_id: If given, only records for this architecture match

        Returns:
            True if the latest non-invalidated record passed
        """
        record = self.get_record(model_id, architecture_id)
        return record is not None and record.is_pass

    def add_record(self, record: VerificationRecord) -> None:
        """Add a new verification record.

        Args:
            record: The verification record to add
        """
        self.records.append(record)
        self.last_updated = datetime.now()

    def invalidate(self, model_id: str, reason: str, architecture_id: Optional[str] = None) -> bool:
        """Invalidate the most recent verification for a model.

        Args:
            model_id: The model ID to invalidate
            reason: Reason for invalidation
            architecture_id: If given, only records for this architecture match

        Returns:
            True if a record was invalidated, False if not found
        """
        record = self.get_record(model_id, architecture_id)
        if record:
            record.invalidated = True
            record.invalidation_reason = reason
            self.last_updated = datetime.now()
            return True
        return False
