"""Spike: install the docstring tool or inject docs into repaired wheels.

Use --install in before-build; pass the repaired wheel directory during repair.
Both steps skip Python <3.10, which docstring-adder no longer supports.
"""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    if sys.version_info < (3, 10):
        print("Skipping wheel docstrings: docstring-adder requires Python >=3.10")
        return

    if sys.argv[1] == "--install":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "wheel",
                "numpy",
                "git+https://github.com/astral-sh/docstring-adder@8377e23ca0c1533bef3e744209709d6d783782b6",
            ],
            check=True,
        )
        return

    wheel_dir = Path(sys.argv[1]).resolve()
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No repaired wheels found in {wheel_dir}")

    for wheel in wheels:
        with TemporaryDirectory() as temporary_dir:
            unpack_dir = Path(temporary_dir)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wheel",
                    "unpack",
                    str(wheel),
                    "-d",
                    str(unpack_dir),
                ],
                check=True,
            )
            (package_root,) = unpack_dir.iterdir()
            # Import this wheel's extension, including its repaired shared libraries.
            # A fresh process avoids loading another installed copy of PySCIPOpt.
            env = dict(
                os.environ,
                PYTHONPATH=str(package_root),
                PYTHONDONTWRITEBYTECODE="1",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "add_docstrings",
                    "--packages",
                    str(package_root / "pyscipopt"),
                ],
                cwd=temporary_dir,
                env=env,
                check=True,
            )
            # Repacking regenerates RECORD, including the modified stub's hash.
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wheel",
                    "pack",
                    str(package_root),
                    "-d",
                    str(wheel_dir),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
