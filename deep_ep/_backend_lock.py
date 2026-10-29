"""Process-local SHMEM construction lock shared by POD and Clos adapters."""

from threading import RLock

LOCK = RLock()
