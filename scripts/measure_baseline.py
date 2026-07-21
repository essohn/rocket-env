"""프리셋별 베이스라인 측정.

등급 컷을 눈감고 정하지 않기 위한 도구다. 무행동 정책과 학습된 DQN 의
점수 분포를 나란히 재서 결과를 기록한다.

이 스크립트가 있어야 하는 이유는 단순하다 — 모든 정책이 같은 점수를 받는
환경도 모든 테스트를 통과한다. 변별력은 재봐야 안다.

사용:
    uv run python scripts/measure_baseline.py --preset landing-basic --steps 100000
"""

import argparse
import statistics

import gymnasium as gym
import numpy as np

import rocket_env  # noqa: F401
from rocket_env.config import PRESETS

NOOP = 1
EVAL_SEEDS = range(40)


def evaluate(env, act) -> tuple[float, float]:
    """(평균 점수, 성공률)."""
    scores, wins = [], 0
    for seed in EVAL_SEEDS:
        obs, _ = env.reset(seed=10_000 + seed)
        done = truncated = False
        total = 0.0
        info = {}
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(act(obs))
            total += float(reward)
        scores.append(total)
        wins += int(info["is_success"])
    return statistics.mean(scores), wins / len(EVAL_SEEDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="landing-basic", choices=list(PRESETS))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="독립적인 학습 실행 횟수 — 학습 시드에 따른 편차를 재기 위함")
    args = parser.parse_args()

    env = gym.make("rocket-v0", config=PRESETS[args.preset])

    noop_score, noop_rate = evaluate(env, lambda _obs: NOOP)
    rng = np.random.default_rng(0)
    rand_score, rand_rate = evaluate(
        env, lambda _obs: int(rng.integers(env.action_space.n)))

    from stable_baselines3 import DQN

    dqn_scores: list[float] = []
    dqn_rates: list[float] = []
    for train_seed in range(args.seeds):
        model = DQN("MlpPolicy", env, verbose=0, device="cpu",
                    learning_rate=args.lr, buffer_size=200_000,
                    learning_starts=5_000, policy_kwargs={"net_arch": [256, 256]},
                    seed=train_seed)
        model.learn(total_timesteps=args.steps)
        score, rate = evaluate(
            env, lambda obs: int(model.predict(obs, deterministic=True)[0]))
        dqn_scores.append(score)
        dqn_rates.append(rate)

    mean_score = statistics.mean(dqn_scores)
    mean_rate = statistics.mean(dqn_rates)

    print(f"preset={args.preset} steps={args.steps} lr={args.lr} train_seeds={args.seeds}")
    print(f"  no-op        score={noop_score:8.2f}  success={noop_rate:6.1%}")
    print(f"  random       score={rand_score:8.2f}  success={rand_rate:6.1%}")
    for i, (score, rate) in enumerate(zip(dqn_scores, dqn_rates)):
        print(f"  DQN seed={i:<3} score={score:8.2f}  success={rate:6.1%}")
    print(
        f"  DQN mean     score={mean_score:8.2f}  success={mean_rate:6.1%}"
        f"   (score min {min(dqn_scores):.2f}, max {max(dqn_scores):.2f})"
    )
    print(f"  separation (mean DQN - no-op) = {mean_score - noop_score:+.2f}")
    env.close()


if __name__ == "__main__":
    main()
