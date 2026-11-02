"""Fixed-seed visual policy evaluation, raw trajectories and a complete first video."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw
import torch

from envs import make_visual_env
from runtime import atomic_json


def video_frame(env,label):
    raw=np.asarray(env.render())
    if raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[-1] not in (3,4):
        raise RuntimeError(f'Expected RGB render frame, got {raw.shape}, {raw.dtype}')
    image=Image.fromarray(raw[:,:,:3]);image.thumbnail((320,240),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(320,264),'black')
    canvas.paste(image,((320-image.width)//2,24+(240-image.height)//2))
    ImageDraw.Draw(canvas).text((4,5),label,fill='white')
    return canvas


@torch.no_grad()
def evaluate(model,directory,env_id,env_kwargs,seeds,device,size=64,stack=4,
             action_repeat=1,max_episode_steps=None,max_steps=2000,policy='greedy'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    was_training=model.training;model.eval();episodes=[]
    try:
        for episode,seed in enumerate(seeds):
            env=make_visual_env(env_id,env_kwargs,size,stack,action_repeat,max_episode_steps)
            rng=np.random.default_rng(seed);video=None
            score=0.;raw_steps=0;terminated=truncated=False
            try:
                observation,_=env.reset(seed=int(seed))
                if episode==0:
                    np.save(directory/'initial_observation.npy',observation)
                    fps=max(1,min(60,round(env.metadata.get('render_fps',20)/action_repeat)))
                    video=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
                        '-f','rawvideo','-pix_fmt','rgb24','-s','320x264','-r',str(fps),'-i','-',
                        '-an','-c:v','libx264','-threads','1','-preset','ultrafast','-crf','24',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(directory/'play.mp4')],
                        stdin=subprocess.PIPE,stderr=subprocess.PIPE)
                with (directory/f'episode_{episode:02d}.jsonl').open('w') as log:
                    for step in range(1,max_steps+1):
                        logits,value=model(torch.as_tensor(observation,device=device)[None])
                        probabilities=logits.softmax(-1)[0].cpu().numpy().astype(np.float64)
                        probabilities/=probabilities.sum()
                        action=int(np.argmax(probabilities)) if policy=='greedy' else int(rng.choice(len(probabilities),p=probabilities))
                        digest=hashlib.sha256(observation.tobytes()).hexdigest()
                        observation,reward,terminated,truncated,info=env.step(action)
                        score+=float(reward);raw_steps+=int(info.get('policy_lab_raw_steps',1))
                        log.write(json.dumps(dict(step=step,action=action,reward=float(reward),score=score,
                            terminated=bool(terminated),truncated=bool(truncated),
                            action_probabilities=probabilities.tolist(),state_value=float(value[0]),
                            observation_sha256=digest))+'\n')
                        if video is not None:
                            video.stdin.write(video_frame(env,f'seed {seed} | step {step} | score {score:.1f}').tobytes())
                        if terminated or truncated:break
                episodes.append(dict(seed=int(seed),score=score,steps=step,raw_steps=raw_steps,
                    terminated=bool(terminated),truncated=bool(truncated),capped=not (terminated or truncated)))
            finally:
                env.close()
                if video is not None:
                    video.stdin.close();message=video.stderr.read().decode()
                    if video.wait(timeout=30):raise RuntimeError(f'ffmpeg failed: {message}')
        scores=[row['score'] for row in episodes]
        result=dict(environment=env_id,policy=policy,episodes=episodes,mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),max_steps=max_steps,any_capped=any(row['capped'] for row in episodes),
            video_scope='first evaluation episode in full, up to the explicit max_steps cap',
            score_scope='raw summed environment rewards; no reward clipping/normalization')
        atomic_json(directory/'evaluation.json',result)
        return result
    finally:model.train(was_training)
