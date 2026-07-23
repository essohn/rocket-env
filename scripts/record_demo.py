"""학습된 에이전트의 데모 영상을 만든다.

`measure_baseline.py`는 학습 → 평가 → 폐기라, 실제로 에이전트가 무엇을
배웠는지 눈으로 볼 방법이 없었다. 강의 자료로 쓸 영상도, 학생이 자기
에이전트를 디버깅할 때 참고할 영상도 이 스크립트가 만든다.

동작:
    1. `--model`(기본값은 프리셋·시드에서 유도)에 저장된 모델이 있으면
       로드하고, 없으면 `measure_baseline.py`와 동일한 하이퍼파라미터로
       새로 학습해 그 경로에 저장한다 — 다음 실행부터는 학습을 건너뛴다.
    2. `--episodes`개 에피소드를 `measure_baseline.py`와 같은 프로토콜
       (시드 10000+i, deterministic=True)로 평가하면서 매 프레임을 모은다.
    3. 가장 좋은 에피소드 — 성공 우선, 동점이면 점수 — 를 골라 영상으로
       쓴다(전부 실패면 점수가 가장 높은 실패를 대신 고른다).
    4. 캐치 성공이면 젓가락이 닫히는 마무리 연출 프레임을 덧붙이고, 착륙
       성공이면 정지 프레임만 덧붙여 끝이 급하지 않게 한다.
    5. 골라진 에피소드로 mp4와, 그걸 축소 재인코딩한 gif를 각각 쓴다.

인코딩은 새 의존성을 늘리지 않으려고 PATH의 ffmpeg에 프레임을
stdin으로 그대로 흘려보낸다. `subprocess`/`ffmpeg`/`stable_baselines3`는
전부 이 스크립트 안에서만 쓰이며, 패키지의 런타임 의존성
(gymnasium/numpy/pygame)은 늘어나지 않는다.

사용:
    uv run python scripts/record_demo.py --preset landing-basic --steps 1000000 --seed 0
"""

import argparse
import math
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np

import rocket_env  # noqa: F401
from rocket_env.config import PRESETS
from rocket_env.render import EXPLODE_SPEED, GRIP_APPROACH_MAX
from rocket_env.types import Outcome

# 빠른 충돌 폭발 연출 길이(프레임). 버섯구름이 부력으로 끝까지 솟구쳐
# 갓을 형성하도록 넉넉히 잡는다 — 30fps 기준 약 6.7초.
BOOM_FRAMES = 200

EVAL_EPISODE_COUNT_DEFAULT = 20
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# --- 마무리 연출 ---
# 캐치 성공 영상은 팔에 매달린 순간 끝나버리면 "잡았다"는 느낌이 없다.
# 마지막 상태 그대로 젓가락만 닫아가며 몇 프레임 더 그린다.
# 젓가락이 조여지는 구간. 14프레임(0.7초)은 순식간이라 고정되는 느낌이
# 없었다. 34프레임이면 1.7초에 걸쳐 천천히 물린다.
# 집게 조임과 기체 안착을 함께 진행하는 구간. 한 동작이라 예전의 grip+settle
# 두 구간(각 34)을 하나로 합쳤다.
CATCH_SETTLE_FRAMES = 46
HOLD_FRAMES = 24      # 다 물고 정지해 있는 구간 (fps=20 기준 약 1.2초)


def default_model_path(preset: str, seed: int) -> Path:
    return ARTIFACTS_DIR / f"{preset}-seed{seed}.zip"


def train_or_load(env, model_path: Path, *, steps: int, lr: float, seed: int):
    """모델이 있으면 로드, 없으면 학습 후 저장한다.

    `measure_baseline.py`와 정확히 같은 하이퍼파라미터를 써야 이 스크립트가
    만든 영상의 성공률이 `docs/baselines.md` 숫자와 비교 가능하다.
    """
    from stable_baselines3 import DQN

    if model_path.exists():
        print(f"기존 모델 로드: {model_path}")
        return DQN.load(model_path, env=env, device="cpu")

    print(f"모델이 없어 새로 학습한다 ({steps:,} 스텝, seed={seed}) ...")
    model = DQN("MlpPolicy", env, verbose=0, device="cpu",
                learning_rate=lr, buffer_size=200_000,
                learning_starts=5_000, policy_kwargs={"net_arch": [256, 256]},
                seed=seed)
    model.learn(total_timesteps=steps)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"모델 저장: {model_path}")
    return model


def closing_frames(env, outcome: str, is_success: bool) -> list[np.ndarray]:
    """에피소드의 마지막 상태로 마무리 연출 프레임을 만든다.

    캐치 성공이면 젓가락이 조여지며 기체가 안착하는 CATCH_SETTLE_FRAMES 에 이어, 다 문
    채(grip=1.0) 정지한 HOLD_FRAMES를 붙인다. 로켓은 `_catch_draw_state`가
    이미 팔에 매단 위치·자세 0·추력 0으로 그리므로 이 구간 내내 완전히
    정지해 보인다. 착륙 성공은 젓가락이 없으니 HOLD_FRAMES만 붙여 영상
    끝이 급하지 않게 한다. 실패로 끝난 에피소드는 붙일 연출이 없다.

    `env.unwrapped._renderer`는 env.render()가 처음 호출될 때 만들어
    캐시해 두는 내부 필드라 공개 접근자가 없다 — grip을 프레임마다 바꿔
    그려야 해서 env.render()가 고정해 넘기는 grip(catch 논리)을 우회해야
    이 함수가 성립한다.
    """
    base = env.unwrapped
    state = base.state
    target = base.task.target(base.cfg)
    renderer = base._renderer
    is_catch = base.cfg["task"] == "catch"

    # 빠른 충돌은 폭발한다. 성공이 아니어도 폭발 연출은 붙인다 — 화구·버섯
    # 구름·파편이 피어올랐다 흩어지는 boom 0->1 시퀀스.
    speed = math.hypot(state.vx, state.vy)
    if (not is_success and outcome in (Outcome.CRASH, Outcome.MISSED,
                                       Outcome.OUT_OF_FUEL)
            and speed > EXPLODE_SPEED):
        return [renderer.draw(state, target, outcome, boom=(i + 1) / BOOM_FRAMES)
                for i in range(BOOM_FRAMES)]

    if not is_success:
        return []

    frames = []
    if is_catch:
        # 집게가 조여지는 것과 기체가 걸림 구조에 얹히는 것을 한 동작으로
        # 동시에 진행한다. 예전처럼 grip 을 다 닫은 뒤 따로 미끄러뜨리면,
        # "잡은 다음 본체가 내려가는" 두 박자가 되어 점프처럼 보였다.
        # 카메라도 직전 위치에 고정한다(hold_camera) — 안 그러면 멎은 기체를
        # 카메라가 계속 좇아가 화면에서 흘러 점프처럼 보인다.
        for i in range(CATCH_SETTLE_FRAMES):
            t = i / (CATCH_SETTLE_FRAMES - 1)
            grip = GRIP_APPROACH_MAX + (1.0 - GRIP_APPROACH_MAX) * t
            frames.append(renderer.draw(state, target, outcome,
                                        grip=grip, settle=t, hold_camera=True))
        frames.extend(
            renderer.draw(state, target, outcome, grip=1.0, settle=1.0,
                          hold_camera=True)
            for _ in range(HOLD_FRAMES))
    else:
        # 착륙 성공: 젓가락이 없으니 정지한 기체를 몇 프레임 더 보여준다.
        # 접지한 기체가 카메라 드리프트로 흐르지 않게 카메라를 고정한다.
        frames.extend(
            renderer.draw(state, target, outcome, grip=0.0, settle=0.0,
                          hold_camera=True)
            for _ in range(HOLD_FRAMES))
    return frames


def run_episode(env, model, seed: int) -> dict:
    """한 에피소드를 끝까지 돌리며 프레임을 모은다."""
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    done = truncated = False
    score = 0.0
    info: dict = {}
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += float(reward)
        frames.append(env.render())
    is_success = bool(info["is_success"])
    return {
        "seed": seed,
        "frames": frames,
        "closing_frames": closing_frames(env, info["outcome"], is_success),
        "score": score,
        "is_success": is_success,
        "outcome": info["outcome"],
        "impact_speed": info["impact_speed"],
    }


def pick_best(episodes: list[dict]) -> dict:
    """성공 여부를 우선, 동점이면 점수로 최고 에피소드를 고른다."""
    return max(episodes, key=lambda ep: (ep["is_success"], ep["score"]))


def frames_to_mp4(frames: list[np.ndarray], fps: int, out_path: Path) -> None:
    """원본 프레임을 ffmpeg 표준입력으로 흘려보내 mp4로 인코딩한다."""
    height, width, _ = frames[0].shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd, input=b"".join(f.astype(np.uint8).tobytes() for f in frames),
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 인코딩 실패 (mp4): {proc.stderr.decode(errors='replace')}")


def mp4_to_gif(mp4_path: Path, gif_path: Path, *, fps: int, width: int = 320) -> None:
    """mp4를 재인코딩해 폭을 줄인 gif를 만든다."""
    filt = f"fps={fps},scale={width}:-1:flags=lanczos"
    cmd = ["ffmpeg", "-y", "-i", str(mp4_path), "-vf", filt, str(gif_path)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 인코딩 실패 (gif): {proc.stderr.decode(errors='replace')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="landing-basic", choices=list(PRESETS))
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=0, help="학습 시드")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODE_COUNT_DEFAULT)
    parser.add_argument("--model", type=Path, default=None,
                         help="모델 경로. 기본값은 프리셋·시드에서 유도")
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    model_path = args.model or default_model_path(args.preset, args.seed)

    env = gym.make("rocket-v0", config=PRESETS[args.preset],
                   render_mode="rgb_array")
    fps = env.metadata["render_fps"]

    model = train_or_load(env, model_path, steps=args.steps, lr=args.lr,
                           seed=args.seed)

    print(f"{args.episodes}개 에피소드 평가 중 (deterministic=True) ...")
    episodes = []
    successes = 0
    for i in range(args.episodes):
        ep = run_episode(env, model, seed=10_000 + i)
        episodes.append(ep)
        successes += int(ep["is_success"])
        print(f"  episode {i:2d}  outcome={ep['outcome']:<12}  "
              f"score={ep['score']:8.2f}  frames={len(ep['frames'])}")
    env.close()

    success_rate = successes / len(episodes)
    print(f"성공률: {successes}/{len(episodes)} ({success_rate:.1%})")

    best = pick_best(episodes)
    impact = best["impact_speed"]
    impact_str = f"{impact:.2f} m/s" if impact is not None else "n/a"
    print(f"선택된 에피소드: seed={best['seed']}  outcome={best['outcome']}  "
          f"score={best['score']:.2f}  impact_speed={impact_str}")
    if not best["is_success"]:
        print("경고: 20개 에피소드 중 성공이 하나도 없어 "
              "가장 점수가 높은 실패 에피소드를 대신 기록한다.")

    frames = best["frames"] + best["closing_frames"]
    if best["closing_frames"]:
        print(f"마무리 연출 프레임 {len(best['closing_frames'])}개 추가 "
              f"(총 {len(frames)}프레임)")

    mp4_path = ARTIFACTS_DIR / f"{args.preset}-best.mp4"
    gif_path = ARTIFACTS_DIR / f"{args.preset}-best.gif"
    frames_to_mp4(frames, fps, mp4_path)
    mp4_to_gif(mp4_path, gif_path, fps=fps)
    print(f"mp4: {mp4_path}")
    print(f"gif: {gif_path}")


if __name__ == "__main__":
    main()
