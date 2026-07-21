"""stable-baselines3 DQN 통합 확인.

서버 워커와 학생 노트북이 정확히 이 경로를 쓴다. SB3가 설치되어 있지
않으면 건너뛴다 — SB3는 런타임 의존성이 아니다.
"""

import math
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

import rocket_env  # noqa: F401,E402
from rocket_env.config import PRESETS  # noqa: E402


@pytest.mark.slow
def test_saved_model_drives_the_full_grading_loop(tmp_path):
    """서버 워커가 실제로 밟는 경로를 그대로 시험한다.

    저장 → 로드 → predict() 출력을 그대로 step() 에 전달 → 점수 누적.
    """
    env = gym.make("rocket-v0", render_mode="rgb_array",
                   config=PRESETS["landing-basic"])
    model = DQN("MlpPolicy", env, verbose=0, device="cpu", seed=0,
                learning_starts=200, buffer_size=5_000,
                policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=2_000)
    path = tmp_path / "model.zip"
    model.save(path)
    loaded = DQN.load(path, env=env, device="cpu")

    scores, outcomes = [], []
    for i in range(2):
        obs, _ = env.reset(seed=1000 + i)
        done = truncated = False
        score = 0.0
        info = {}
        while not (done or truncated):
            action, _ = loaded.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            score += float(reward)
        assert math.isfinite(score)
        scores.append(score)
        outcomes.append(bool(info["is_success"]))

    assert len(scores) == 2
    assert all(isinstance(o, bool) for o in outcomes)
    env.close()
