from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path


def _run(cmd: str, cwd: Path | None = None) -> None:
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=str(cwd) if cwd else None)


def _copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab setup and SAC training.")
    parser.add_argument(
        "--requirements",
        type=Path,
        default=None,
        help="Path to requirements file (defaults to repo requirements_local.txt).",
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/unitreego2"))
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument(
        "--vec-env",
        choices=["auto", "subproc", "dummy"],
        default="auto",
        help="Vectorized env backend. Use dummy if subprocesses crash.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a short benchmark to estimate steps/sec and max timesteps.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only the benchmark and skip the full training run.",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=20_000,
        help="Timesteps to use for the benchmark run.",
    )
    parser.add_argument(
        "--benchmark-run-id",
        type=int,
        default=None,
        help="Optional run id for the benchmark run.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=12.0,
        help="Runtime budget used to estimate max timesteps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for periodic checkpoints (defaults to Drive if available).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Save a checkpoint every N timesteps (0 to disable).",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    requirements_path = args.requirements or (repo_dir / "requirements_local.txt")

    if requirements_path.exists():
        _run(f"pip -q install -r {requirements_path}")
    else:
        _run("pip -q install 'gymnasium[mujoco]' stable-baselines3 matplotlib tensorboard")

    train_script = repo_dir / "train_sac.py"
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.drive_dir.exists():
        checkpoint_dir = args.drive_dir / "checkpoints"
    base_cmd = (
        f"python {train_script}"
        f" --n-envs {args.n_envs}"
        f" --seed {args.seed}"
        f" --max-episode-steps {args.max_episode_steps}"
        f" --vec-env {args.vec_env}"
    )
    if checkpoint_dir is not None and args.checkpoint_freq > 0:
        base_cmd += f" --checkpoint-dir {checkpoint_dir}"
        base_cmd += f" --checkpoint-freq {args.checkpoint_freq}"

    def _train_cmd(total_timesteps: int, run_id: int | None) -> str:
        cmd = f"{base_cmd} --total-timesteps {total_timesteps}"
        if run_id is not None:
            cmd += f" --run-id {run_id}"
        return cmd

    if args.benchmark or args.benchmark_only:
        benchmark_cmd = _train_cmd(args.benchmark_steps, args.benchmark_run_id)
        start = time.perf_counter()
        _run(benchmark_cmd, cwd=repo_dir)
        elapsed = max(time.perf_counter() - start, 1e-6)
        steps_per_sec = args.benchmark_steps / elapsed
        max_timesteps = int(steps_per_sec * args.max_hours * 3600 * 0.9)
        print(
            "Benchmark results:"
            f" {steps_per_sec:.1f} steps/sec,"
            f" estimated max timesteps for {args.max_hours:.1f}h ≈ {max_timesteps}"
        )

    if not args.benchmark_only:
        train_cmd = _train_cmd(args.total_timesteps, args.run_id)
        _run(train_cmd, cwd=repo_dir)

    if args.drive_dir.exists():
        _copy_dir(repo_dir / "models", args.drive_dir / "models")
        _copy_dir(repo_dir / "logs", args.drive_dir / "logs")
        print(f"Saved models and logs to {args.drive_dir}")
    else:
        print(f"Drive dir not found: {args.drive_dir}")


if __name__ == "__main__":
    main()
