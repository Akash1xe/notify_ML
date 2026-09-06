from __future__ import annotations


class NotifyError(Exception):
    """Base class for expected application failures."""

    code = "notify_error"


class JobNotFoundError(NotifyError):
    code = "job_not_found"


class InvalidJobTransitionError(NotifyError):
    code = "invalid_job_transition"


class JobCancelledError(NotifyError):
    code = "job_cancelled"


class StorageError(NotifyError):
    code = "storage_error"


class ProcessingError(NotifyError):
    code = "processing_error"
