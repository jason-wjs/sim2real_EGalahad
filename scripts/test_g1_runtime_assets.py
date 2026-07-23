from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim2real.config.robots import g1


class G1RuntimeAssetsTest(unittest.TestCase):
    def test_missing_local_asset_does_not_fall_back_to_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.xml"
            selected = g1._select_g1_mjcf_reference(missing_path)
            self.assertEqual(selected, str(missing_path.resolve()))

    def test_matching_local_asset_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "g1.xml"
            content = b"<mujoco/>"
            local_path.write_bytes(content)
            expected_sha256 = hashlib.sha256(content).hexdigest()

            with patch.object(g1, "G1_MJCF_SHA256", expected_sha256):
                selected = g1._select_g1_mjcf_reference(local_path)

            self.assertEqual(selected, str(local_path.resolve()))

    def test_mismatched_local_asset_fails_instead_of_silently_changing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "g1.xml"
            local_path.write_bytes(b"unexpected")

            with (
                patch.object(g1, "G1_MJCF_SHA256", "0" * 64),
                self.assertRaisesRegex(RuntimeError, "does not match"),
            ):
                g1._select_g1_mjcf_reference(local_path)


if __name__ == "__main__":
    unittest.main()
