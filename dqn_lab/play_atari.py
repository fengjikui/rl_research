"""Keyboard-controlled original ALE games. No model or training is involved.

Run from the repository root: .venv/bin/python dqn_lab/play_atari.py
1/2/3 switch games; arrows or WASD move; Space starts/fires; P pauses;
R resets; Escape closes. Unfocused windows pause automatically.
"""
import argparse
import os
import time

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import ale_py
import gymnasium as gym
import numpy as np
import pygame


GAMES = ('Breakout', 'Pong', 'SpaceInvaders')


def open_game(name):
    # Show original colour frames. Training later preprocesses these to grey stacks.
    # Four emulator frames per decision, 15 decisions/second = nominal 60 Hz play.
    env = gym.make(f'ALE/{name}-v5', render_mode='rgb_array', frameskip=4,
                   repeat_action_probability=0.0, full_action_space=False)
    observation, info = env.reset(seed=0)
    print(f'PLAY {name}: {env.unwrapped.get_action_meanings()}', flush=True)
    return env, observation, info


def action_for(pressed, name, meanings):
    fire = bool(pressed[pygame.K_SPACE])
    if name == 'Pong':
        # Atari Pong calls its two vertical paddle directions RIGHT and LEFT.
        right = bool(pressed[pygame.K_UP] or pressed[pygame.K_w])
        left = bool(pressed[pygame.K_DOWN] or pressed[pygame.K_s])
    else:
        right = bool(pressed[pygame.K_RIGHT] or pressed[pygame.K_d])
        left = bool(pressed[pygame.K_LEFT] or pressed[pygame.K_a])
    movement = 'RIGHT' if right and not left else 'LEFT' if left and not right else ''
    meaning = movement + ('FIRE' if fire else '') if movement else 'FIRE' if fire else 'NOOP'
    if meaning not in meanings:
        # Breakout has no combined move-and-fire action; serve first, then move.
        meaning = 'FIRE' if fire else movement or 'NOOP'
    return meanings.index(meaning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', choices=GAMES, default='Breakout')
    parser.add_argument('--smoke-frames', type=int, default=0,
                        help='Render a bounded number of frames, then exit (for verification).')
    args = parser.parse_args()
    gym.register_envs(ale_py)
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((480, 730))
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    name = args.game
    env, observation, info = open_game(name)
    score = 0.0
    paused = True
    finished = False
    running = True
    next_step = time.monotonic()
    frames = 0
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.WINDOWFOCUSLOST:
                    paused = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        env.close()
                        name = GAMES[event.key - pygame.K_1]
                        env, observation, info = open_game(name)
                        score, paused, finished = 0.0, True, False
                        next_step = time.monotonic()
                    elif event.key == pygame.K_r:
                        observation, info = env.reset()
                        score, paused, finished = 0.0, True, False
                        next_step = time.monotonic()
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_SPACE and not finished:
                        paused = False
            now = time.monotonic()
            if not paused and not finished and now >= next_step:
                action = action_for(pygame.key.get_pressed(), name,
                                    env.unwrapped.get_action_meanings())
                observation, reward, terminated, truncated, info = env.step(action)
                score += reward
                finished = terminated or truncated
                next_step = now + 1 / 15
            status = 'GAME OVER - R to reset' if finished else 'PAUSED - Space to play' if paused else 'PLAYING'
            pygame.display.set_caption(f'Atari DQN Lab | {name} | {status}')
            screen.fill((15, 19, 28))
            frame = pygame.surfarray.make_surface(np.asarray(observation).swapaxes(0, 1))
            screen.blit(pygame.transform.scale(frame, (480, 630)), (0, 100))
            lines = ['1 Breakout   2 Pong   3 Space Invaders',
                     'Arrows / WASD move | Space start/fire',
                     'P pause | R reset | Esc quit',
                     f'Score: {score:g}   Lives: {info.get("lives", "-")}   {status}']
            for row, text in enumerate(lines):
                screen.blit(font.render(text, True, (230, 235, 242)), (8, 5 + row * 23))
            pygame.display.flip()
            clock.tick(60)
            frames += 1
            if args.smoke_frames and frames >= args.smoke_frames:
                break
    finally:
        env.close()
        pygame.quit()


if __name__ == '__main__':
    main()
