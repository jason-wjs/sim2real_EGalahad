from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "artifacts" / "lock.v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_stats(root: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_sha256 = _sha256(path)
        digest.update(f"{file_sha256}  {relative}\n".encode("utf-8"))
        total_bytes += path.stat().st_size
    return len(files), total_bytes, digest.hexdigest()


def _load_lock(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported artifact lock schema: {payload.get('schema_version')}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Artifact lock must contain an artifacts list")
    return artifacts


def _select(
    artifacts: list[dict[str, Any]],
    *,
    profiles: Iterable[str],
    artifact_ids: Iterable[str],
) -> list[dict[str, Any]]:
    selected_profiles = set(profiles)
    selected_ids = set(artifact_ids)
    known_ids = {str(artifact["id"]) for artifact in artifacts}
    missing_ids = selected_ids - known_ids
    if missing_ids:
        raise ValueError(f"Unknown artifact ids: {sorted(missing_ids)}")
    if "all" in selected_profiles:
        return artifacts
    return [
        artifact
        for artifact in artifacts
        if str(artifact["id"]) in selected_ids
        or selected_profiles.intersection(artifact.get("profiles", []))
    ]


def _file_matches(path: Path, artifact: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(artifact["bytes"])
        and _sha256(path) == str(artifact["sha256"])
    )


def _tree_matches(root: Path, artifact: dict[str, Any]) -> bool:
    if not root.is_dir():
        return False
    files, total_bytes, tree_sha256 = _tree_stats(root)
    if files != int(artifact["tree_files"]):
        return False
    if total_bytes != int(artifact["tree_bytes"]):
        return False
    expected_digest = artifact.get("tree_sha256")
    return expected_digest is None or tree_sha256 == str(expected_digest)


def _verify_one(artifact: dict[str, Any]) -> bool:
    kind = str(artifact["kind"])
    if kind == "file":
        return _file_matches(REPO_ROOT / str(artifact["path"]), artifact)
    if kind == "tar.gz":
        return _tree_matches(REPO_ROOT / str(artifact["tree_root"]), artifact)
    if kind == "bos-prefix":
        return _tree_matches(REPO_ROOT / str(artifact["path"]), artifact)
    raise ValueError(f"Unsupported artifact kind: {kind}")


def _bcecmd() -> str:
    executable = shutil.which("bcecmd")
    if executable is None:
        raise RuntimeError("bcecmd is required to fetch BCE BOS artifacts")
    return executable


def _download(uri: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    if temporary.exists():
        temporary.unlink()
    subprocess.run(
        [
            _bcecmd(),
            "bos",
            "cp",
            uri,
            str(temporary),
            "--yes",
            "--disable-bar",
        ],
        check=True,
    )
    temporary.replace(target)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            member_path = (destination / member.name).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise ValueError(f"Archive member escapes extraction root: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
        bundle.extractall(destination)


def _fetch_one(artifact: dict[str, Any], *, force: bool) -> None:
    artifact_id = str(artifact["id"])
    if not force and _verify_one(artifact):
        print(f"[ok] {artifact_id}")
        return

    kind = str(artifact["kind"])
    if kind == "file":
        target = REPO_ROOT / str(artifact["path"])
        cache = (
            REPO_ROOT
            / ".cache"
            / "artifacts"
            / "downloads"
            / f"{artifact_id}-{target.name}"
        )
        _download(str(artifact["uri"]), cache)
        if not _file_matches(cache, artifact):
            raise RuntimeError(f"Downloaded artifact failed checksum: {artifact_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(cache, target)
    elif kind == "tar.gz":
        cache = REPO_ROOT / str(artifact["cache_path"])
        if force or not _file_matches(cache, artifact):
            _download(str(artifact["uri"]), cache)
        if not _file_matches(cache, artifact):
            raise RuntimeError(f"Downloaded archive failed checksum: {artifact_id}")
        _safe_extract(cache, REPO_ROOT / str(artifact["extract_to"]))
    elif kind == "bos-prefix":
        target = REPO_ROOT / str(artifact["path"])
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                _bcecmd(),
                "bos",
                "sync",
                str(artifact["uri"]),
                str(target),
                "--yes",
                "--disable-bar",
            ],
            check=True,
        )
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")

    if not _verify_one(artifact):
        raise RuntimeError(f"Installed artifact failed verification: {artifact_id}")
    print(f"[fetched] {artifact_id}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and verify locked BCE BOS artifacts.")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "verify", "fetch"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--profile",
            action="append",
            choices=("reference", "benchmark", "validation", "legacy", "all"),
            default=[],
        )
        child.add_argument("--artifact", action="append", default=[])
        if command == "fetch":
            child.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts = _load_lock(args.lock.expanduser().resolve())
    profiles = args.profile or ([] if args.artifact else ["reference"])
    selected = _select(artifacts, profiles=profiles, artifact_ids=args.artifact)
    if not selected:
        raise ValueError("No artifacts matched the requested profile or ids")

    if args.command == "list":
        for artifact in selected:
            print(f"{artifact['id']}\t{artifact['kind']}\t{artifact['uri']}")
        return 0

    if args.command == "fetch":
        for artifact in selected:
            _fetch_one(artifact, force=args.force)
        return 0

    failures = []
    for artifact in selected:
        matches = _verify_one(artifact)
        print(f"[{'ok' if matches else 'missing-or-mismatched'}] {artifact['id']}")
        if not matches:
            failures.append(str(artifact["id"]))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
