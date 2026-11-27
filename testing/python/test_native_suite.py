import subprocess

from conftest import BUILD


def test_google_test_suite_passes():
    """Include the independently runnable C++ tests in the pytest result."""
    result = subprocess.run(
        ["ctest", "--test-dir", str(BUILD), "--output-on-failure"],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
