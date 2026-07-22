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
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np

import rocket_env  # noqa: F401
from rocket_env.config import PRESETS
from rocket_env.render import GRIP_APPROACH_MAX

EVAL_EPISODE_COUNT_DEFAULT = 20
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# --- 마무리 연출 ---
# 캐치 성공 영상은 팔에 매달린 순간 끝나버리면 "잡았다"는 느낌이 없다.
# 마지막 상태 그대로 젓가락만 닫아가며 몇 프레임 더 그린다.
# 젓가락이 조여지는 구간. 14프레임(0.7초)은 순식간이라 고정되는 느낌이
# 없었다. 34프레임이면 1.7초에 걸쳐 천천히 물린다.
GRIP_FRAMES = 34
# 걸림 구조가 팔에 얹힐 때까지 미끄러지는 구간. 12프레임(0.6초)은 너무
# 빨라 미끄러지는 동작이 보이지 않았다. 34프레임이면 1.7초에 걸쳐 내려온다.
SETTLE_FRAMES = 34      # grip 0 -> 1로 닫히는 구간
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

    캐치 성공이면 젓가락이 grip 0 -> 1로 닫히는 GRIP_FRAMES에 이어, 다 문
    채(grip=1.0) 정지한 HOLD_FRAMES를 붙인다. 로켓은 `_catch_draw_state`가
    이미 팔에 매단 위치·자세 0·추력 0으로 그리므로 이 구간 내내 완전히
    정지해 보인다. 착륙 성공은 젓가락이 없으니 HOLD_FRAMES만 붙여 영상
    끝이 급하지 않게 한다. 실패로 끝난 에피소드는 붙일 연출이 없다.

    `env.unwrapped._renderer`는 env.render()가 처음 호출될 때 만들어
    캐시해 두는 내부 필드라 공개 접근자가 없다 — grip을 프레임마다 바꿔
    그려야 해서 env.render()가 고정해 넘기는 grip(catch 논리)을 우회해야
    이 함수가 성립한다.
    """
    if not is_success:
        return []
    base = env.unwrapped
    state = base.state
    target = base.task.target(base.cfg)
    renderer = base._renderer
    is_catch = base.cfg["task"] == "catch"

    frames = []
    if is_catch:
        # 1단계: 집게가 마저 조여진다. 접근 중 이미 GRIP_APPROACH_MAX 까지
        # 닫혀 있으므로 0이 아니라 그 값에서 이어받는다 — 0부터 시작하면
        # 젓가락이 한 번 벌어졌다 닫히는 것처럼 보인다.
        for i in range(GRIP_FRAMES):
            t = i / (GRIP_FRAMES - 1)
            grip = GRIP_APPROACH_MAX + (1.0 - GRIP_APPROACH_MAX) * t
            frames.append(renderer.draw(state, target, outcome,
                                        grip=grip, settle=0.0))
        # 2단계: 기체가 미끄러져 내려가 걸림 구조가 팔에 얹힌다.
        for i in range(SETTLE_FRAMES):
            frames.append(renderer.draw(state, target, outcome, grip=1.0,
                                        settle=i / (SETTLE_FRAMES - 1)))
    hold_grip = 1.0 if is_catch else 0.0
    hold_settle = 1.0 if is_catch else 0.0
    frames.extend(renderer.draw(state, target, outcome, grip=hold_grip,
                                settle=hold_settle)
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
