"""Auto-imported at interpreter startup when this directory is on PYTHONPATH.

/fast has no file locking, so HuggingFace's filelock-based cache locks hang or
error. Per the MPI cluster guide, swap FileLock for SoftFileLock everywhere when
SOFTFILELOCK=1 is set in the environment.
"""
import os

if os.environ.get("SOFTFILELOCK"):
    try:
        import filelock
        from filelock import SoftFileLock

        filelock.FileLock = SoftFileLock
    except Exception:  # pragma: no cover - best effort
        pass
