"""Offline and mocked-network checks for reproducible SPICE provisioning."""

from contextlib import redirect_stderr
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from urllib.error import URLError
from unittest.mock import Mock, patch

from core.lunar_frame_bridge import LunarFrameBridge
from core.spice_engine import SpiceGeometryEngine
from scripts import setup_spice_kernels as setup


class SpiceKernelSetupTest(unittest.TestCase):
    def test_setup_inventory_matches_both_existing_kernel_loaders(self) -> None:
        self.assertEqual(
            {spec.filename for spec in setup.KERNEL_SPECS},
            set(SpiceGeometryEngine.KERNEL_NAMES) | set(LunarFrameBridge.KERNEL_NAMES),
        )
        self.assertTrue(all(spec.url.startswith("https://naif.jpl.nasa.gov/") for spec in setup.KERNEL_SPECS))

    def test_missing_kernels_are_reported_by_offline_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            setup, "urlopen", side_effect=AssertionError("check must not access network")
        ):
            problems = setup.check_kernels(Path(temporary))
        self.assertEqual(len(problems), 7)
        self.assertTrue(all("missing" in problem for problem in problems))

    def test_empty_and_corrupt_kernel_files_are_reported(self) -> None:
        spec = setup.KernelSpec("sample.tls", "https://naif.jpl.nasa.gov/sample.tls")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / spec.filename
            path.write_bytes(b"")
            self.assertEqual(setup.check_kernels(Path(temporary), (spec,)), ["sample.tls: empty file"])
            path.write_bytes(b"not a valid kernel")
            with patch.object(setup, "_validate_spice_file", return_value="SPICE rejected the file"):
                self.assertEqual(
                    setup.check_kernels(Path(temporary), (spec,)),
                    ["sample.tls: SPICE rejected the file"],
                )

    def test_existing_loadable_kernel_is_preserved_without_download(self) -> None:
        spec = setup.KernelSpec("sample.tls", "https://naif.jpl.nasa.gov/sample.tls")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / spec.filename
            original = b"existing valid kernel bytes"
            path.write_bytes(original)
            with patch.object(setup, "_validate_spice_file", return_value=None), patch.object(
                setup, "_download", side_effect=AssertionError("valid file must not be downloaded")
            ):
                messages = setup.provision_kernels(Path(temporary), specs=(spec,))
            self.assertEqual(path.read_bytes(), original)
            self.assertIn("kept existing", messages[0])

    def test_official_checksum_accepts_matching_bytes_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.bpc"
            payload = b"officially published kernel fixture"
            path.write_bytes(payload)
            expected = hashlib.md5(payload).hexdigest()
            self.assertTrue(setup.verify_md5(path, expected))
            self.assertFalse(setup.verify_md5(path, "0" * 32))

    def test_download_requires_checksum_match_before_installing(self) -> None:
        spec = setup.KernelSpec(
            "sample.tf", "https://naif.jpl.nasa.gov/sample.tf", required_md5="0" * 32
        )
        payload = b"downloaded fixture"

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.filename
            with patch.object(setup, "_validate_spice_file", side_effect=lambda path: "missing" if path == destination else None):
                with self.assertRaisesRegex(setup.KernelSetupError, "MD5 verification failed"):
                    setup.provision_kernels(
                        Path(temporary), specs=(spec,), opener=lambda *_args, **_kwargs: Response(payload)
                    )
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(destination.name + ".download").exists())

    def test_offline_check_cli_never_attempts_a_network_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            setup, "urlopen", side_effect=AssertionError("offline check attempted network")
        ), redirect_stderr(io.StringIO()):
            result = setup.main(["--check", "--spice-dir", temporary])
        self.assertEqual(result, 1)

    def test_setup_reports_network_failure_without_leaving_partial_file(self) -> None:
        spec = setup.KernelSpec("sample.tf", "https://naif.jpl.nasa.gov/sample.tf")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(setup.KernelSetupError, "Could not provision sample.tf"):
                setup.provision_kernels(
                    Path(temporary),
                    specs=(spec,),
                    opener=Mock(side_effect=URLError("network unavailable")),
                )
            self.assertFalse((Path(temporary) / spec.filename).exists())
            self.assertFalse((Path(temporary) / (spec.filename + ".download")).exists())


if __name__ == "__main__":
    unittest.main()
