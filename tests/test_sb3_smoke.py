"""stable-baselines3 DQN 통합 확인.

서버 워커와 학생 노트북이 정확히 이 경로를 쓴다. SB3가 설치되어 있지
않으면 건너뛴다 — SB3는 런타임 의존성이 아니다.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

import rocket_env  # noqa: F401,E402
from rocket_env.config import PRESETS  # noqa: E402


@pytest.mark.slow
def test_dqn_trains_and_predicts_without_error(tmp_path):
    env = gym.make("rocket-v0", render_mode="rgb_array",
                   config=PRESETS["landing-basic"])
    model = DQN("MlpPolicy", env, verbose=0, device="cpu",
                learning_starts=200, buffer_size=5_000,
                policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=5_000)

    path = tmp_path / "model.zip"
    model.save(path)
    loaded = DQN.load(path, env=env, device="cpu")

    obs, _ = env.reset(seed=0)
    action, _ = loaded.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))
    env.close()


@pytest.mark.slow
def test_server_evaluation_loop_shape_works():
    """서버 워커의 평가 루프와 동일한 형태로 돌려본다."""
    env = gym.make("rocket-v0", render_mode="rgb_array",
                   config=PRESETS["landing-descent"])
    scores, outcomes = [], []
    rng = np.random.default_rng(0)

    for i in range(3):
        obs, _ = env.reset(seed=1000 + i)
        done = truncated = False
        score = 0.0
        info = {}
        while not (done or truncated):
            action = int(rng.integers(env.action_space.n))
            obs, reward, done, truncated, info = env.step(action)
            score += float(reward)
        scores.append(score)
        outcomes.append(bool(info["is_success"]))

    assert len(scores) == 3
    assert all(isinstance(o, bool) for o in outcomes)
    env.close()
