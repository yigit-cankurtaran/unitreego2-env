from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(event_path: Path, tags: list[str]) -> dict[str, tuple[list[int], list[float]]]:
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()

    scalars = {}
    available = set(ea.Tags().get("scalars", []))
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        scalars[tag] = (steps, values)
    return scalars


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SB3 TensorBoard scalars.")
    parser.add_argument("--logdir", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("training_metrics.png"))
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["rollout/ep_rew_mean", "rollout/ep_len_mean"],
    )
    args = parser.parse_args()

    event_files = sorted(args.logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found under {args.logdir}")

    event_path = event_files[-1]
    scalars = load_scalars(event_path, args.tags)
    if not scalars:
        raise SystemExit("No matching scalar tags found")

    fig, axes = plt.subplots(len(scalars), 1, figsize=(8, 3 * len(scalars)), sharex=True)
    if len(scalars) == 1:
        axes = [axes]

    for ax, (tag, (steps, values)) in zip(axes, scalars.items()):
        ax.plot(steps, values)
        ax.set_title(tag)
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timesteps")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
