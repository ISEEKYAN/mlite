# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small compatibility patches for dependency-version gaps in examples."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Any


def _patch_transformers_rope_ignore_keys() -> None:
    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return

    for cls in vars(rope_utils).values():
        if not isinstance(cls, type):
            continue
        if getattr(cls, "_verl_mlite_rope_ignore_keys_patch", False):
            continue
        descriptor = vars(cls).get("_check_received_keys")
        if descriptor is None:
            continue

        is_staticmethod = isinstance(descriptor, staticmethod)
        is_classmethod = isinstance(descriptor, classmethod)
        original = descriptor.__func__ if is_staticmethod or is_classmethod else descriptor

        def build_wrapper(check_received_keys: Any) -> Any:
            @wraps(check_received_keys)
            def patched(*args: Any, **kwargs: Any) -> Any:
                ignore_keys = kwargs.get("ignore_keys")
                if isinstance(ignore_keys, list):
                    kwargs["ignore_keys"] = set(ignore_keys)
                elif ignore_keys is not None and not isinstance(ignore_keys, set):
                    if isinstance(ignore_keys, Iterable) and not isinstance(
                        ignore_keys, (str, bytes)
                    ):
                        kwargs["ignore_keys"] = set(ignore_keys)
                return check_received_keys(*args, **kwargs)

            return patched

        patched = build_wrapper(original)
        if is_staticmethod:
            cls._check_received_keys = staticmethod(patched)
        elif is_classmethod:
            cls._check_received_keys = classmethod(patched)
        else:
            cls._check_received_keys = patched
        cls._verl_mlite_rope_ignore_keys_patch = True


def _patch_transformers_flash_attn_compat() -> None:
    try:
        import transformers.utils as transformers_utils
    except Exception:
        return

    if hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
        return

    version_check = getattr(transformers_utils, "is_flash_attn_greater_or_equal", None)
    if not callable(version_check):
        transformers_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
        return

    def is_flash_attn_greater_or_equal_2_10() -> bool:
        return bool(version_check("2.10.0"))

    transformers_utils.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10


def apply_runtime_patches() -> None:
    _patch_transformers_flash_attn_compat()
    _patch_transformers_rope_ignore_keys()
