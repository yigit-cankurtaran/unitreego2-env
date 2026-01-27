from __future__ import annotations

import argparse
import os
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
    parser = argparse.ArgumentParser(description="Kaggle setup and SAC training.")
    parser.add_argument(
        "--requirements",
        type=Path,
        default=None,
        help="Path to requirements file (defaults to repo requirements_local.txt).",
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--resume-path", type=Path, default=None)
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
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/unitreego2"),
        help="Where to copy models/logs for Kaggle output.",
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
        default=9.0,
        help="Runtime budget used to estimate max timesteps.",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default="egl",
        help="Sets MUJOCO_GL for headless rendering (e.g. egl, osmesa, or empty).",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    requirements_path = args.requirements or (repo_dir / "requirements_local.txt")

    if args.mujoco_gl:
        os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    if requirements_path.exists():
        _run(f"pip -q install -r {requirements_path}")
    else:
        _run("pip -q install 'gymnasium[mujoco]' stable-baselines3 matplotlib tensorboard")

    train_script = repo_dir / "train_sac.py"
    base_cmd = (
        f"python {train_script}"
        f" --n-envs {args.n_envs}"
        f" --seed {args.seed}"
        f" --max-episode-steps {args.max_episode_steps}"
        f" --vec-env {args.vec_env}"
    )
    if args.resume_path is not None:
        base_cmd += f" --resume-path {args.resume_path}"

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _copy_dir(repo_dir / "models", args.output_dir / "models")
    _copy_dir(repo_dir / "logs", args.output_dir / "logs")
    print(f"Saved models and logs to {args.output_dir}")


if __name__ == "__main__":
    main()
