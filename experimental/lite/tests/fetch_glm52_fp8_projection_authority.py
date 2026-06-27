#!/usr/bin/env python3
"""Fetch only the pinned GLM-5.2-FP8 q_a projection payload ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

FIXTURE = Path(__file__).parent / "unit" / "model" / "glm52_fp8_header_authority.json"
SHARD = "model-00001-of-00141.safetensors"
OUTPUT_NAMES = {
    "model.layers.0.self_attn.q_a_proj.weight": "q_a_proj.bin",
    "model.layers.0.self_attn.q_a_proj.weight_scale_inv": "q_a_proj_scale_inv.bin",
    "model.layers.0.self_attn.q_a_layernorm.weight": "q_a_layernorm.bin",
}


def _download_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(
                f"range request returned HTTP {response.status}, expected 206; "
                "refusing a possible full-shard download"
            )
        content_range = response.headers.get("Content-Range")
        expected_prefix = f"bytes {start}-{end}/"
        if not content_range or not content_range.startswith(expected_prefix):
            raise RuntimeError(
                f"range request returned Content-Range={content_range!r}, "
                f"expected prefix {expected_prefix!r}"
            )
        payload = response.read(end - start + 2)
    expected_bytes = end - start + 1
    if len(payload) != expected_bytes:
        raise RuntimeError(
            f"range request returned {len(payload)} bytes, expected {expected_bytes}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    authority = json.loads(FIXTURE.read_text())
    source = authority["source"]
    shard = source["safetensors"][SHARD]
    base_url = (
        f"https://huggingface.co/{source['repo']}/resolve/"
        f"{source['revision']}/{SHARD}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "repo": source["repo"],
        "revision": source["revision"],
        "shard": SHARD,
        "payloads": {},
    }
    for tensor_name, filename in OUTPUT_NAMES.items():
        contract = shard["payload_ranges"][tensor_name]
        start, end = contract["file_range"]
        payload = _download_range(base_url, start, end)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != contract["sha256"]:
            raise RuntimeError(
                f"payload digest mismatch for {tensor_name}: "
                f"actual={digest}, expected={contract['sha256']}"
            )

        destination = args.output_dir / filename
        temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        manifest["payloads"][tensor_name] = {
            "file": filename,
            "bytes": len(payload),
            "sha256": digest,
        }

    manifest_path = args.output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp.{os.getpid()}"
    )
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    print(
        "GLM52_FP8_PROJECTION_AUTHORITY_READY "
        f"revision={source['revision']} bytes="
        f"{sum(item['bytes'] for item in manifest['payloads'].values())} "
        f"path={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
