"""Fixed-seed Atari evaluation with original scores, raw actions and real video."""
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import torch

from breakout import make_env


@torch.no_grad()
def evaluate_stage(model, directory, seeds, max_steps=4500, epsilon=0.0):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    episodes, previews = [], []
    for episode, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        env = make_env()
        obs, info = env.reset(seed=seed)
        lives = int(info.get('lives',env.unwrapped.ale.lives()))
        serve_pending = False
        score = 0.0
        video = None
        if episode == 0:
            # 完整保存第一个评估回合，不用挑选“表现最好”的一局。
            video = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
                '-f','rawvideo','-pix_fmt','rgb24','-s','320x444','-r','15','-i','-',
                '-an','-c:v','libx264','-threads','1','-preset','ultrafast','-crf','24',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(directory/'play.mp4')],
                stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        terminated = truncated = False
        try:
            with (directory/f'episode_{episode:02d}.jsonl').open('w') as log:
                for step in range(1,max_steps+1):
                    # [4,84,84] -> [1,4,84,84] -> [1,4]，评估不加随机探索。
                    q = model(torch.as_tensor(np.asarray(obs),device=device)[None])[0]
                    forced_serve = serve_pending
                    exploratory = not forced_serve and epsilon>0 and rng.random()<epsilon
                    action = 1 if forced_serve else int(rng.integers(env.action_space.n)) if exploratory else int(q.argmax().item())
                    obs, reward, terminated, truncated, info = env.step(action)
                    score += float(reward)
                    next_lives = int(info.get('lives',env.unwrapped.ale.lives()))
                    # 丢球后自动按一次发球；训练与评估共用同一规则，并记录真实动作。
                    serve_pending = next_lives < lives and not (terminated or truncated)
                    lives = next_lives
                    log.write(json.dumps(dict(step=step,action=action,reward=float(reward),
                        score=score,lives=lives,forced_serve=forced_serve,random_exploration=bool(exploratory),
                        terminated=bool(terminated),truncated=bool(truncated),
                        q_values=q.cpu().tolist()))+'\n')
                    if video is not None:
                        frame = Image.new('RGB',(320,444),'black')
                        frame.paste(Image.fromarray(env.render()).resize((320,420),Image.Resampling.NEAREST),(0,24))
                        ImageDraw.Draw(frame).text((5,5),f'step {step} | score {score:g} | seed {seed}',fill='white')
                        video.stdin.write(frame.tobytes())
                        # 仅 GIF 预览截取前 600 步；MP4 保留整局或完整截断长度。
                        if step <= 600 and step % 4 == 0:
                            previews.append(frame.copy())
                    if terminated or truncated:
                        break
        finally:
            env.close()
            if video is not None:
                video.stdin.close()
                error = video.stderr.read().decode()
                if video.wait(timeout=30):
                    raise RuntimeError(f'Video encoder failed: {error}')
        episodes.append(dict(seed=seed,score=score,steps=step,
            terminated=bool(terminated),truncated=bool(truncated),
            capped=not (terminated or truncated)))
    if previews:
        previews[0].save(directory/'preview.gif',save_all=True,append_images=previews[1:],duration=267,loop=0)
    scores = [e['score'] for e in episodes]
    result = dict(policy='greedy; deterministic re-serve after each lost life' if epsilon==0 else
                  f'epsilon-greedy ({epsilon}); deterministic re-serve after each lost life',epsilon=epsilon,
        max_steps_per_episode=max_steps,episodes=episodes,mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),any_capped=any(e['capped'] for e in episodes),
        video_scope='episode 0 in full; preview GIF shows at most its first 600 decisions')
    (directory/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    model.train(was_training)
    return result
