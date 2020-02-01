import numpy as np
from collections import namedtuple
Transition = namedtuple(
    'Transtition', ('state', 'action', 'reward', 'next_state', 'done'))




def create_eps_schedule(eps_start, eps_end, eps_decay):
    eps_s = np.linspace(eps_start, eps_end, eps_decay)

    def eps_schedule(current_t):
        return eps_s[min(eps_decay-1, current_t)]
    return eps_schedule
