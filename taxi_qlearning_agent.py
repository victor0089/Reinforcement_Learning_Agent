"""Train, test, and demonstrate a tabular Q-learning agent in Taxi-v3.

Quick start:
    python -m pip install "gymnasium==1.2.1" "numpy>=1.26" "matplotlib>=3.8"
    python taxi_qlearning_agent.py

The program makes an output folder containing performance data, a Q-table,
learning curves, and a visual demonstration of one completed taxi journey.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # Works on laptops and systems without a display.
import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = ("South", "North", "East", "West", "Pick up", "Drop off")


@dataclass(frozen=True)
class Settings:
    episodes: int = 20_000
    learning_rate: float = 0.15
    discount_factor: float = 0.99
    initial_epsilon: float = 1.0
    minimum_epsilon: float = 0.03
    epsilon_decay: float = 0.9995
    evaluation_episodes: int = 200
    seed: int = 42


def moving_average(values: list[float], window: int = 250) -> np.ndarray:
    """Return a rolling mean, including a partial window at the beginning."""
    raw = np.asarray(values, dtype=float)
    totals = np.cumsum(np.insert(raw, 0, 0.0))
    counts = np.minimum(np.arange(1, len(raw) + 1), window)
    starts = np.maximum(0, np.arange(len(raw)) - window + 1)
    return (totals[1:] - totals[starts]) / counts


def train(settings: Settings) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Learn action values from trial-and-error interaction, without labels."""
    env = gym.make("Taxi-v3")
    rng = np.random.default_rng(settings.seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    history: list[dict[str, float]] = []

    for episode in range(settings.episodes):
        # A separate episode seed makes the experiment reproducible.
        state, _ = env.reset(seed=settings.seed + episode)
        epsilon = max(
            settings.minimum_epsilon,
            settings.initial_epsilon * settings.epsilon_decay**episode,
        )
        total_reward, steps, delivered = 0.0, 0, False

        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)

            # Terminal states have no future rewards to estimate.
            future_value = 0.0 if terminated else float(np.max(q_table[next_state]))
            target = reward + settings.discount_factor * future_value
            q_table[state, action] += settings.learning_rate * (
                target - q_table[state, action]
            )

            total_reward += reward
            steps += 1
            delivered = terminated
            state = next_state
            if terminated or truncated:
                break

        history.append(
            {
                "episode": episode + 1,
                "reward": total_reward,
                "steps": steps,
                "success": int(delivered),
                "epsilon": epsilon,
            }
        )
        if (episode + 1) % 5_000 == 0:
            recent = history[-500:]
            print(
                f"Training: {episode + 1:>5,} episodes | "
                f"last-500 average reward {np.mean([x['reward'] for x in recent]):7.2f} | "
                f"successes {np.mean([x['success'] for x in recent]):6.1%}"
            )

    env.close()
    return q_table, history


def evaluate(
    q_table: np.ndarray | None, settings: Settings, *, random_policy: bool = False
) -> dict[str, float]:
    """Evaluate policies on the same held-out sequence of 200 starting seeds."""
    env = gym.make("Taxi-v3")
    rng = np.random.default_rng(settings.seed + 1_000_000)
    returns, lengths, successes = [], [], []

    for index in range(settings.evaluation_episodes):
        state, _ = env.reset(seed=settings.seed + 100_000 + index)
        total_reward, length, success = 0.0, 0, False

        while True:
            if random_policy:
                action = int(rng.integers(env.action_space.n))
            else:
                if q_table is None:
                    raise ValueError("A trained Q-table is required for greedy evaluation.")
                action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            length += 1
            if terminated or truncated:
                success = terminated
                break

        returns.append(total_reward)
        lengths.append(length)
        successes.append(success)

    env.close()
    return {
        "episodes": settings.evaluation_episodes,
        "mean_reward": round(float(np.mean(returns)), 3),
        "reward_std": round(float(np.std(returns, ddof=1)), 3),
        "mean_steps": round(float(np.mean(lengths)), 3),
        "successful_deliveries": int(sum(successes)),
        "success_rate": round(float(np.mean(successes)), 4),
    }


def demonstration(q_table: np.ndarray, settings: Settings) -> list[dict]:
    """Record actual frames from a representative completed test episode."""
    env = gym.make("Taxi-v3", render_mode="ansi")
    trace = []
    selected_seed = None
    for offset in range(100):
        state, _ = env.reset(seed=settings.seed + 200_000 + offset)
        trial = [
            {"state": int(state), "action": "Start", "reward": 0,
             "image": env.render(), "step": 0}
        ]
        for step in range(1, 201):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            trial.append(
                {"state": int(state), "action": ACTION_NAMES[action],
                 "reward": reward, "image": env.render(), "step": step}
            )
            if terminated or truncated:
                if terminated:
                    trace, selected_seed = trial, settings.seed + 200_000 + offset
                break
        if trace:
            break
    env.close()
    if not trace:
        raise RuntimeError("No successful demonstration found in 100 attempts.")
    print(f"Demo episode seed: {selected_seed}; steps: {len(trace)-1}; "
          f"total reward: {sum(step['reward'] for step in trace):.0f}")
    return trace


def save_results(
    directory: Path, settings: Settings, q_table: np.ndarray,
    history: list[dict[str, float]], baseline: dict, trained: dict,
    trace: list[dict],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "taxi_qtable.npy", q_table)
    with (directory / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "environment": "Gymnasium Taxi-v3", "algorithm": "Tabular Q-learning",
        "settings": asdict(settings), "q_table_shape": list(q_table.shape),
        "random_baseline": baseline, "trained_agent": trained,
        "initial_500_training_mean_reward": round(float(np.mean(
            [x["reward"] for x in history[:500]])), 3),
        "final_500_training_mean_reward": round(float(np.mean(
            [x["reward"] for x in history[-500:]])), 3),
        "initial_500_training_success_rate": round(float(np.mean(
            [x["success"] for x in history[:500]])), 4),
        "final_500_training_success_rate": round(float(np.mean(
            [x["success"] for x in history[-500:]])), 4),
        "demo_steps": len(trace) - 1,
        "demo_reward": int(sum(x["reward"] for x in trace)),
    }
    (directory / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    rewards = [x["reward"] for x in history]
    success = [x["success"] for x in history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    episodes = np.arange(1, len(history) + 1)
    ax1.plot(episodes, moving_average(rewards), color="#1163ab", lw=2)
    ax1.axhline(trained["mean_reward"], ls="--", lw=1.3, color="#137e48",
                label="Final greedy evaluation mean")
    ax1.set(ylabel="Reward (rolling 250 episodes)",
            title="Learning progress in Taxi-v3")
    ax1.legend(loc="lower right")
    ax2.plot(episodes, moving_average(success) * 100, color="#d57228", lw=2)
    ax2.set(xlabel="Training episode", ylabel="Successful deliveries (%)",
            ylim=(-5, 105))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "learning_curve.png", dpi=170)
    plt.close(fig)

    # A visual record built from the episode states and environment grid. The
    # original ANSI display encodes the taxi's position with terminal colors;
    # replace that color-only marker with a visible "T" for a readable PNG.
    pickup_index = next(
        (i for i, item in enumerate(trace) if item["action"] == "Pick up"),
        len(trace) // 2,
    )
    frames = [trace[0], trace[pickup_index], trace[-1]]
    captions = ["Initial state", "After passenger pickup", "Successful drop-off"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    import re
    for ax, frame, caption in zip(axes, frames, captions):
        ascii_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame["image"])
        lines = ascii_frame.splitlines()
        row = frame["state"] // 100
        col = (frame["state"] // 20) % 5
        passenger = (frame["state"] // 4) % 5
        destination = frame["state"] % 4
        line = list(lines[row + 1])
        line[1 + 2 * col] = "T"
        lines[row + 1] = "".join(line)
        ascii_frame = "\n".join(lines)
        ax.text(0.06, 0.88, ascii_frame, fontfamily="monospace", fontsize=13,
                va="top", transform=ax.transAxes, linespacing=1.35)
        landmarks = ("Red", "Green", "Yellow", "Blue")
        passenger_label = "In taxi" if passenger == 4 else landmarks[passenger]
        ax.text(0.06, 0.18,
                f"T = taxi at row {row}, column {col}\n"
                f"Passenger: {passenger_label}\n"
                f"Destination: {landmarks[destination]}",
                fontfamily="sans-serif", fontsize=10, va="top",
                transform=ax.transAxes)
        ax.set(title=f"{caption}\nStep {frame['step']}", xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    fig.suptitle("Agent demonstration: actual Taxi-v3 environment frames", fontsize=14)
    fig.tight_layout()
    fig.savefig(directory / "agent_demonstration.png", dpi=170)
    plt.close(fig)

    with (directory / "demo_trace.txt").open("w", encoding="utf-8") as f:
        f.write("Taxi-v3 trained-agent demonstration (actual environment output)\n")
        f.write(f"Total steps: {len(trace)-1}\n")
        f.write(f"Total reward: {sum(x['reward'] for x in trace)}\n\n")
        for item in trace:
            f.write(f"Step {item['step']}: {item['action']} | "
                    f"reward {item['reward']}\n{item['image']}\n")

    print("\nPOLICY COMPARISON (200 identical starting states)")
    print("Policy          Mean reward   Mean steps   Success")
    print(f"Random          {baseline['mean_reward']:10.2f}   "
          f"{baseline['mean_steps']:10.2f}   {baseline['success_rate']:6.1%}")
    print(f"Trained         {trained['mean_reward']:10.2f}   "
          f"{trained['mean_steps']:10.2f}   {trained['success_rate']:6.1%}")
    print(f"\nFiles saved to: {directory.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("taxi_results"))
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("Episodes and evaluation episodes must both be positive.")

    settings = Settings(
        episodes=args.episodes, evaluation_episodes=args.eval_episodes, seed=args.seed
    )
    q_table, history = train(settings)
    baseline = evaluate(None, settings, random_policy=True)
    trained = evaluate(q_table, settings)
    trace = demonstration(q_table, settings)
    save_results(args.output, settings, q_table, history, baseline, trained, trace)


if __name__ == "__main__":
    main()
