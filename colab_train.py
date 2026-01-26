from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import shutil


def _run(cmd: str) -> None:
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def _copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab setup and SAC training.")
    parser.add_argument("--requirements", type=Path, default=Path("requirements_local.txt"))
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/unitreego2"))
    args = parser.parse_args()

    if args.requirements.exists():
        _run(f"pip -q install -r {args.requirements}")
    else:
        _run("pip -q install 'gymnasium[mujoco]' stable-baselines3 matplotlib tensorboard")

    cmd = f"python train_sac.py --total-timesteps {args.total_timesteps}"
    if args.run_id is not None:
        cmd += f" --run-id {args.run_id}"
    _run(cmd)

    if args.drive_dir.exists():
        _copy_dir(Path("models"), args.drive_dir / "models")
        _copy_dir(Path("logs"), args.drive_dir / "logs")
        print(f"Saved models and logs to {args.drive_dir}")
    else:
        print(f"Drive dir not found: {args.drive_dir}")


if __name__ == "__main__":
    main()
