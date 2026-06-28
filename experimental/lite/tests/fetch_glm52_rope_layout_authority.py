#!/usr/bin/env python3
"""Fetch and verify the small pinned GLM-5.2 RoPE authority files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

FIXTURE = Path(__file__).parent / "unit" / "model" / "glm52_rope_layout_authority.json"
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mlite-glm52-rope-authority/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(
                f"GET {url} returned HTTP {response.status}, expected 200"
            )
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_SOURCE_BYTES:
            raise RuntimeError(
                f"GET {url} advertised {content_length} bytes; refusing a non-small file"
            )
        payload = response.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise RuntimeError(
            f"GET {url} exceeded the {MAX_SOURCE_BYTES}-byte authority limit"
        )
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_digest(*, label: str, payload: bytes, expected: str) -> str:
    actual = _sha256(payload)
    if actual != expected:
        raise RuntimeError(
            f"digest mismatch for {label}: actual={actual}, expected={expected}"
        )
    return actual


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_release_config(payload: bytes, contract: dict) -> None:
    try:
        config = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("pinned GLM-5.2 config is not valid UTF-8 JSON") from error
    for field, expected in contract["required_values"].items():
        if field not in config:
            raise RuntimeError(f"pinned GLM-5.2 config is missing {field!r}")
        actual = config[field]
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(
                f"pinned GLM-5.2 config field {field!r} is {actual!r}, "
                f"expected {expected!r}"
            )


def _validate_vllm_source(payload: bytes, contract: dict, path: str) -> None:
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"pinned vLLM source {path} is not UTF-8") from error
    for snippet_contract in contract["required_snippets"]:
        snippet = snippet_contract["text"]
        expected_count = snippet_contract["expected_count"]
        if type(expected_count) is not int or expected_count < 1:
            raise RuntimeError(
                f"invalid expected_count for vLLM source {path}: {snippet_contract!r}"
            )
        actual_count = source.count(snippet)
        if actual_count != expected_count:
            raise RuntimeError(
                f"pinned vLLM source {path} contains {actual_count} copies of "
                f"{snippet!r}, expected {expected_count}"
            )


def _validate_remote_revisions(authority: dict) -> None:
    release = authority["release"]
    encoded_repo = urllib.parse.quote(release["repo"], safe="/")
    model_metadata = json.loads(
        _download(
            f"https://huggingface.co/api/models/{encoded_repo}/revision/"
            f"{release['revision']}"
        )
    )
    if model_metadata.get("sha") != release["revision"]:
        raise RuntimeError(
            "Hugging Face revision resolution drifted: "
            f"actual={model_metadata.get('sha')!r}, expected={release['revision']!r}"
        )

    vllm = authority["vllm"]
    encoded_tag = urllib.parse.quote(vllm["release"], safe="")
    tag_metadata = json.loads(
        _download(
            f"https://api.github.com/repos/{vllm['repo']}/git/ref/tags/{encoded_tag}"
        )
    )
    tag_object = tag_metadata.get("object", {})
    if tag_object.get("type") != "commit" or tag_object.get("sha") != vllm["revision"]:
        raise RuntimeError(
            f"vLLM tag {vllm['release']} does not resolve to pinned commit "
            f"{vllm['revision']}: {tag_object!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    authority = json.loads(FIXTURE.read_text())
    _validate_remote_revisions(authority)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    release = authority["release"]
    config_contract = release["config"]
    config_url = (
        f"https://huggingface.co/{release['repo']}/resolve/"
        f"{release['revision']}/{config_contract['path']}"
    )
    config_payload = _download(config_url)
    config_digest = _require_digest(
        label=f"{release['repo']}@{release['revision']}/{config_contract['path']}",
        payload=config_payload,
        expected=config_contract["sha256"],
    )
    _validate_release_config(config_payload, config_contract)
    config_filename = "glm52-release-config.json"
    _write_atomic(args.output_dir / config_filename, config_payload)

    manifest = {
        "release": {
            "file": config_filename,
            "repo": release["repo"],
            "revision": release["revision"],
            "sha256": config_digest,
        },
        "vllm": {
            "files": {},
            "release": authority["vllm"]["release"],
            "repo": authority["vllm"]["repo"],
            "revision": authority["vllm"]["revision"],
        },
    }
    vllm = authority["vllm"]
    for source_path, source_contract in vllm["files"].items():
        source_url = (
            f"https://raw.githubusercontent.com/{vllm['repo']}/"
            f"{vllm['revision']}/{source_path}"
        )
        source_payload = _download(source_url)
        source_digest = _require_digest(
            label=f"{vllm['repo']}@{vllm['revision']}/{source_path}",
            payload=source_payload,
            expected=source_contract["sha256"],
        )
        _validate_vllm_source(source_payload, source_contract, source_path)
        filename = "vllm-" + source_path.replace("/", "-")
        _write_atomic(args.output_dir / filename, source_payload)
        manifest["vllm"]["files"][source_path] = {
            "bytes": len(source_payload),
            "file": filename,
            "sha256": source_digest,
        }

    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write_atomic(args.output_dir / "manifest.json", manifest_payload)
    print(
        "GLM52_ROPE_LAYOUT_AUTHORITY_READY "
        f"release_revision={release['revision']} "
        f"vllm_revision={vllm['revision']} path={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
