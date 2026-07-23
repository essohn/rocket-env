"""키보드로 직접 로켓을 조종하는 수동 모드.

정책 없이 사람이 착륙/포획을 시도해 본다 — 환경의 난이도와 조작감을 몸으로
느끼기에 좋다. 화면이 있는 로컬 환경에서 실행한다(헤드리스 서버 불가).

조작:
    ↑ / W      추력 단계 올리기 (0 → 4: 0, 0.6, 1.0, 1.6, 2.5 G)
    ↓ / S      추력 단계 내리기
    ← / A      노즐 왼쪽으로 (누르는 동안)
    → / D      노즐 오른쪽으로 (누르는 동안)
    R          에피소드 재시작
    V          방금 에피소드를 영상으로 저장 (imageio 필요)
    Esc / Q    종료

사용:
    uv run python scripts/play.py --preset landing-basic --livery 내별명
"""

import argparse
from collections import deque
from datetime import datetime

import gymnasium as gym
import numpy as np
import pygame

import rocket_env  # noqa: F401  (환경 등록)
from rocket_env.config import PRESETS
from rocket_env.render import EXPLODE_SPEED
from rocket_env.types import Outcome

MAX_VIDEO_FRAMES = 900   # 최근 프레임만 보관(메모리 상한). 한 에피소드에 충분하다.


def _grab_frame(renderer) -> np.ndarray:
    """현재 렌더 서피스를 (H, W, 3) RGB 배열로 복사한다."""
    return np.transpose(pygame.surfarray.array3d(renderer.surface), (1, 0, 2)).copy()


def _save_video(frames, livery: str | None, score: float | None) -> None:
    """모아 둔 프레임을 mp4(없으면 gif)로 저장한다."""
    if not frames:
        print("저장할 프레임이 없습니다.")
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("영상 저장에는 imageio 가 필요합니다: "
              "pip install imageio imageio-ffmpeg")
        return
    tag = f"{(livery or 'play')}_{'' if score is None else round(score)}"
    stamp = datetime.now().strftime("%H%M%S")
    base = f"rocket_{tag}_{stamp}"
    seq = list(frames)
    try:
        path = f"{base}.mp4"
        imageio.mimsave(path, seq, fps=20, macro_block_size=1)
    except Exception:                          # ffmpeg 없거나 실패 → gif 로
        path = f"{base}.gif"
        imageio.mimsave(path, seq, fps=20)
    print(f"영상 저장: {path}  ({len(seq)} 프레임)")

THRUST_KEYS_UP = (pygame.K_UP, pygame.K_w)
THRUST_KEYS_DOWN = (pygame.K_DOWN, pygame.K_s)
QUIT_KEYS = (pygame.K_ESCAPE, pygame.K_q)
CRASH_OUTCOMES = (Outcome.CRASH, Outcome.MISSED, Outcome.OUT_OF_FUEL)
BOOM_FRAMES = 40    # 폭발 연출 길이(프레임)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="landing-basic", choices=list(PRESETS))
    parser.add_argument("--livery", default=None, help="로켓에 새길 별명(문자열)")
    args = parser.parse_args()

    config = dict(PRESETS[args.preset])
    if args.livery is not None:
        config["livery"] = args.livery

    env = gym.make("rocket-v0", config=config, render_mode="human")
    obs, info = env.reset()
    env.render()   # 창을 연다
    print(__doc__)

    unwrapped = env.unwrapped
    thrust = 0          # 현재 추력 단계(0~4). ↑/↓ 로 바꾼다.
    done = False
    boom_t = 0.0        # 폭발 연출 진행도(0=없음). 빠른 충돌에서 0→1 로 오른다.
    score = None        # 종료 시 착륙 점수
    frames = deque(maxlen=MAX_VIDEO_FRAMES)   # 영상 저장용 최근 프레임
    frames.append(_grab_frame(unwrapped._renderer))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in QUIT_KEYS:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    thrust, done, boom_t, score = 0, False, 0.0, None
                    frames.clear()
                elif event.key == pygame.K_v:
                    _save_video(frames, args.livery, score)
                elif event.key in THRUST_KEYS_UP:
                    thrust = min(4, thrust + 1)
                elif event.key in THRUST_KEYS_DOWN:
                    thrust = max(0, thrust - 1)

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                nozzle = 0
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                nozzle = 2
            else:
                nozzle = 1
            obs, reward, terminated, truncated, info = env.step(thrust * 3 + nozzle)
            if terminated or truncated:
                done = True
                outcome = info["outcome"]
                score = info.get("score")
                mark = "성공!" if info["is_success"] else "실패"
                print(f"[{outcome}] {mark}  점수 {score:.0f}  "
                      f"접지속도 {info['impact_speed']}  (R 재시작 · V 영상저장)")
                # 빠른 충돌은 폭발한다.
                if (outcome in CRASH_OUTCOMES
                        and (info["impact_speed"] or 0.0) > EXPLODE_SPEED):
                    boom_t = 1e-6   # 다음 프레임부터 폭발 연출 시작

        if done:
            # 종료: 큰 점수·재시작 안내를 띄우고, 빠른 충돌이면 폭발 연출
            # (boom 0→1)을 이어 그린다. env.render 는 점수·안내가 없으므로
            # 렌더러를 직접 부른다.
            if boom_t > 0.0:
                boom_t = min(1.0, boom_t + 1.0 / BOOM_FRAMES)
            unwrapped._renderer.draw(
                unwrapped.state, unwrapped._target, unwrapped._outcome,
                boom=boom_t, retry_hint=True, score=score)
        else:
            env.render()   # human 모드는 내부에서 20Hz 로 페이싱한다

        frames.append(_grab_frame(unwrapped._renderer))

    env.close()


if __name__ == "__main__":
    main()
