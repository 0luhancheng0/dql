import numpy as np
from collections import namedtuple
import re
from pathlib import Path
Transition = namedtuple(
    'Transtition', ('state', 'action', 'reward', 'next_state', 'done'))
def create_eps_schedule(eps_start, eps_end, eps_decay):
    eps_s = np.linspace(eps_start, eps_end, eps_decay)

    def eps_schedule(current_t):
        return eps_s[min(eps_decay-1, current_t)]
    return eps_schedule

def get_latest_model_path(hps):
    checkpoint_path = (Path(hps['log_dir']) / hps['envname'] / 'checkpoints').resolve()
    print(checkpoint_path)
    latest_episode_tag = -1
    # ckpt_list = []
    path = None
    for f in checkpoint_path.iterdir():
        if f.is_file():
            matched = re.match('saved_model.ckpt-([0-9]+$)', f.name)
            if matched is not None:
                episode_tag = int(matched.group(1))
                if episode_tag > latest_episode_tag:
                    latest_episode_tag = episode_tag
                    path = checkpoint_path / f
    if path is None:
        raise ValueError('provided log_dir path does not have checkpoints')

    return path, latest_episode_tag
