"""Clean, deployable core for the second IntentComm protocol.

This package deliberately imports only reusable controller infrastructure from
``sbs824``.  It must never depend on historical ``train_*``, ``phase*`` or
``rolling_cem_*`` experiment scripts.
"""

from .protocol import IntentMode, SYNC_EVENT_V2_DEV, SyncEventProtocol

__all__ = ["IntentMode", "SYNC_EVENT_V2_DEV", "SyncEventProtocol"]
