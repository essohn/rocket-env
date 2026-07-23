"""키보드로 직접 로켓을 조종하는 수동 모드.

정책 없이 사람이 착륙/포획을 시도해 본다 — 환경의 난이도와 조작감을 몸으로
느끼기에 좋다. 화면이 있는 로컬 환경에서 실행한다(헤드리스 서버 불가).

조작:
    ↑ / W      추력 단계 올리기 (0 → 4: 0, 0.6, 1.0, 1.6, 2.5 G)
    ↓ / S      추력 단계 내리기
    ← / A      노즐 왼쪽으로 (누르는 동안)
    → / D      노즐 오른쪽으로 (누르는 동안)
    R          에피소드 재시작
    Esc / Q    종료

사용:
    uv run python scripts/play.py --preset landing-basic --livery 내별명
"""

import argparse

import gymnasium as gym
import pygame

import rocket_env  # noqa: F401  (환경 등록)
from rocket_env.config import PRESETS
from rocket_env.types import Outcome

THRUST_KEYS_UP = (pygame.K_UP, pygame.K_w)
THRUST_KEYS_DOWN = (pygame.K_DOWN, pygame.K_s)
QUIT_KEYS = (pygame.K_ESCAPE, pygame.K_q)


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

    thrust = 0          # 현재 추력 단계(0~4). ↑/↓ 로 바꾼다.
    done = False
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
                    thrust, done = 0, False
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
                mark = "성공!" if info["is_success"] else "실패"
                print(f"[{outcome}] {mark}  접지속도 {info['impact_speed']}  "
                      f"(R 로 재시작)")

        env.render()   # human 모드는 내부에서 20Hz 로 페이싱한다

    env.close()


if __name__ == "__main__":
    main()
