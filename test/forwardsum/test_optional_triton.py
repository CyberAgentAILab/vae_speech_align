import subprocess
import sys


class TestOptionalTriton:
    """Tests that triton stays an optional dependency."""

    def test_forwardsum_import_does_not_require_triton(self) -> None:
        """The CPU backends remain importable when Triton is unavailable."""
        import_script = (
            "import sys; "
            "sys.modules['triton'] = None; "
            "import vae_speech_align.forwardsum"
        )

        subprocess.run([sys.executable, "-c", import_script], check=True)
