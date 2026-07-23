"""학생용 학습 템플릿. 알고리즘을 골라 한 라운드를 학습하고 평가한다.

핵심 선택지 — 알고리즘이 도달할 수 있는 라운드가 다르다:

  --algo dqn   기본 DQN. 초반 라운드(landing-basic, landing-attitude)를
               학습한다. 낮은 고도의 짧은 낙하에는 충분하다.

  --algo ppo   PPO + 관측/보상 정규화(VecNormalize). 후반 고고도 라운드
               (landing-descent 이상)는 바닐라 DQN 이 긴 낙하의 신용
               할당에 실패해 발산하지만, 이 셋업은 학습해낸다. 실측:
               y=800 라운드에서 DQN 0% vs PPO+정규화 ~40%.

정규화를 쓰는 PPO 는 모델과 함께 정규화 통계(VecNormalize)를 저장/복원해야
한다 — 저장 경로 옆에 `<out>-vecnorm.pkl` 로 함께 남긴다.

사용 예:
    uv run python scripts/train.py --preset landing-basic --algo dqn --steps 400000
    uv run python scripts/train.py --preset landing-gust  --algo ppo --steps 1500000
"""

import argparse
import math
from pathlib import Path

import gymnasium as gym

import rocket_env  # noqa: F401  (환경 등록)
from rocket_env.config import PRESETS

EVAL_SEED_BASE = 20000


def _evaluate(predict, cfg, n: int = 30) -> tuple[int, float]:
    """학습된 정책을 평가 시드로 돌려 성공 수와 평균 접지 속도를 낸다."""
    env = gym.make("rocket-v0", config=cfg)
    wins, speeds = 0, []
    for i in range(n):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        while True:
            obs, _, terminated, truncated, info = env.step(int(predict(obs)))
            if terminated or truncated:
                break
        wins += int(info["is_success"])
        st = env.unwrapped.state
        speeds.append(math.hypot(st.vx, st.vy))
    env.close()
    return wins, sum(speeds) / len(speeds)


def train_dqn(cfg, steps: int, seed: int, out: Path):
    from stable_baselines3 import DQN

    env = gym.make("rocket-v0", config=cfg)
    model = DQN("MlpPolicy", env, verbose=0, device="cpu",
                learning_rate=6e-4, gamma=0.99, buffer_size=200_000,
                learning_starts=5_000, policy_kwargs={"net_arch": [256, 256]},
                seed=seed)
    model.learn(total_timesteps=steps)
    if out:
        model.save(out)
    return lambda obs: model.predict(obs, deterministic=True)[0]


def train_ppo(cfg, steps: int, seed: int, out: Path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = VecNormalize(
        DummyVecEnv([lambda: gym.make("rocket-v0", config=cfg)]),
        norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, verbose=0, device="cpu",
                gamma=0.999, gae_lambda=0.95, n_steps=2048, batch_size=256,
                n_epochs=10, ent_coef=0.01, learning_rate=3e-4,
                policy_kwargs={"net_arch": [256, 256]}, seed=seed)
    model.learn(total_timesteps=steps)
    # 평가·배포에는 학습으로 갱신된 정규화 통계가 필요하다. 통계를 고정한다.
    venv.training = False
    venv.norm_reward = False
    if out:
        model.save(out)
        venv.save(str(out) + "-vecnorm.pkl")

    def predict(obs):
        return model.predict(venv.normalize_obs(obs), deterministic=True)[0]

    return predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="landing-basic", choices=list(PRESETS))
    parser.add_argument("--algo", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="모델 저장 경로(확장자 없이). ppo 는 옆에 -vecnorm.pkl 도 남긴다")
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    trainer = train_dqn if args.algo == "dqn" else train_ppo
    print(f"학습: preset={args.preset} algo={args.algo} steps={args.steps} seed={args.seed}")
    predict = trainer(cfg, args.steps, args.seed, args.out)

    wins, mean_speed = _evaluate(predict, cfg, args.eval_episodes)
    rate = wins / args.eval_episodes * 100
    print(f"성공률: {wins}/{args.eval_episodes} ({rate:.0f}%)  평균 접지속도={mean_speed:.1f} m/s")


if __name__ == "__main__":
    main()
