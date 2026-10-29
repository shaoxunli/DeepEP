"""Compatibility module; Buffer uses the same machine selection as deep_ep.Buffer."""

from .buffers.legacy import Buffer


def _is_blackwell():
    return False


__all__ = ["Buffer", "_is_blackwell"]
