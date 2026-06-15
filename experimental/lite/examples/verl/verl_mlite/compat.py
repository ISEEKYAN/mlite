# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small compatibility patches for dependency-version gaps in examples."""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
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


def _patch_transformers_auto_model_aliases() -> None:
    try:
        import transformers
        from transformers.models.auto import modeling_auto
    except Exception:
        return

    image_text = getattr(transformers, "AutoModelForImageTextToText", None)
    vision2seq = getattr(transformers, "AutoModelForVision2Seq", None)
    fallback = image_text or getattr(transformers, "AutoModel", None)
    if vision2seq is None and fallback is not None:
        transformers.AutoModelForVision2Seq = fallback
        modeling_auto.AutoModelForVision2Seq = fallback
    if image_text is None and vision2seq is not None:
        transformers.AutoModelForImageTextToText = vision2seq
        modeling_auto.AutoModelForImageTextToText = vision2seq

    for name, value in (
        ("AutoModelForVision2Seq", getattr(transformers, "AutoModelForVision2Seq", None)),
        (
            "AutoModelForImageTextToText",
            getattr(transformers, "AutoModelForImageTextToText", None),
        ),
    ):
        if value is None:
            continue
        objects = getattr(transformers, "_objects", None)
        if isinstance(objects, dict):
            objects[name] = value
        class_to_module = getattr(transformers, "_class_to_module", None)
        if isinstance(class_to_module, dict):
            class_to_module[name] = "models.auto.modeling_auto"


def apply_runtime_patches() -> None:
    _patch_transformers_auto_model_aliases()
    _patch_transformers_rope_ignore_keys()
    _patch_verl_engine_package_import()


def _load_verl_file(relative_path: str, module_name: str):
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("No module named 'verl'")

    path = Path(next(iter(spec.submodule_search_locations))) / relative_path
    file_spec = importlib.util.spec_from_file_location(module_name, path)
    if file_spec is None or file_spec.loader is None:
        raise ImportError(f"Unable to load VERL module from {path}")

    module = importlib.util.module_from_spec(file_spec)
    sys.modules[module_name] = module
    file_spec.loader.exec_module(module)
    return module


def _patch_verl_engine_package_import() -> None:
    if "verl.workers.engine" in sys.modules:
        return
    try:
        base = _load_verl_file("workers/engine/base.py", "_verl_mlite_verl_engine_base")
    except (FileNotFoundError, ModuleNotFoundError, ImportError):
        return

    module = types.ModuleType("verl.workers.engine")
    for name in ("BaseEngine", "BaseEngineCtx", "EngineRegistry"):
        if hasattr(base, name):
            setattr(module, name, getattr(base, name))
    module.__file__ = getattr(base, "__file__", None)
    module.__package__ = "verl.workers.engine"
    module.__path__ = []
    sys.modules["verl.workers.engine"] = module


def load_verl_engine_api():
    try:
        base = _load_verl_file("workers/engine/base.py", "_verl_mlite_verl_engine_base")
        utils = _load_verl_file("workers/engine/utils.py", "_verl_mlite_verl_engine_utils")
        BaseEngine = base.BaseEngine
        BaseEngineCtx = base.BaseEngineCtx
        EngineRegistry = base.EngineRegistry
        postprocess_batch_func = utils.postprocess_batch_func
        prepare_micro_batches = utils.prepare_micro_batches
    except (FileNotFoundError, ModuleNotFoundError, ImportError):
        from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
        from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches

    return BaseEngine, BaseEngineCtx, EngineRegistry, postprocess_batch_func, prepare_micro_batches
