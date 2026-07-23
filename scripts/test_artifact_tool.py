from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts import artifact_tool


class ArtifactToolTest(unittest.TestCase):
    def test_tree_digest_is_stable_and_path_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "nested" / "b.txt").write_text("beta", encoding="utf-8")

            files, total_bytes, digest = artifact_tool._tree_stats(root)

        expected = hashlib.sha256()
        expected.update(
            (
                f"{hashlib.sha256(b'alpha').hexdigest()}  a.txt\n"
                f"{hashlib.sha256(b'beta').hexdigest()}  nested/b.txt\n"
            ).encode("utf-8")
        )
        self.assertEqual(files, 2)
        self.assertEqual(total_bytes, 9)
        self.assertEqual(digest, expected.hexdigest())

    def test_safe_extract_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                payload = b"escape"
                member = tarfile.TarInfo("../escape.txt")
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(ValueError, "escapes extraction root"):
                artifact_tool._safe_extract(archive, root / "destination")

    def test_profile_selection_supports_explicit_artifact_ids(self) -> None:
        artifacts = [
            {"id": "model", "profiles": ["reference"]},
            {"id": "motion", "profiles": ["benchmark"]},
        ]

        selected = artifact_tool._select(
            artifacts,
            profiles=["reference"],
            artifact_ids=["motion"],
        )

        self.assertEqual([item["id"] for item in selected], ["model", "motion"])

    def test_reference_lock_covers_eight_adjacent_policy_contracts(self) -> None:
        artifacts = artifact_tool._load_lock(artifact_tool.DEFAULT_LOCK)
        policies = [
            artifact
            for artifact in artifacts
            if artifact.get("policy_config") is not None
        ]

        self.assertEqual(len(policies), 8)
        self.assertEqual(len({artifact["id"] for artifact in policies}), 8)
        for artifact in policies:
            config_path = artifact_tool.REPO_ROOT / artifact["policy_config"]
            model_path = artifact_tool.REPO_ROOT / artifact["path"]
            self.assertEqual(model_path, config_path.with_suffix(".onnx"))
            self.assertIsInstance(
                yaml.safe_load(config_path.read_text(encoding="utf-8")),
                dict,
            )


if __name__ == "__main__":
    unittest.main()
