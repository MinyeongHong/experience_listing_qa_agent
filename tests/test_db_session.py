import os
import subprocess
import sys


def test_app_import_does_not_require_database_configuration():
    environment = os.environ.copy()
    environment.pop("DATABASE_URL", None)
    environment.pop("TEST_DATABASE_URL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dotenv; dotenv.load_dotenv = lambda: None; import app.main",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
