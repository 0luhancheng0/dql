import torch
import numpy as np
import gym
from collections import defaultdict
from tqdm import tqdm
# import re
from dql.helper import Transition
from pathlib import Path
# from dql.model import ReplayMemory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Evaluator:
    def __init__(self, env, model, hps, logger):
        self.logger = logger
        self.env = env
        self.model = model
        self.hps = hps
        # self.running_reward = 0
        return
    def generate_one_step_transition(self, state):
        action = self.model.policy_net.greedy_policy(state.unsqueeze(0)).squeeze()
        _, reward, done, _ = self.env.step(action.item())
        reward = torch.tensor(reward)
        current_screen = self.env.get_screen()
        next_state = torch.cat([state[1:, ...], current_screen])
        transition = Transition(state, action, reward, next_state, done)
        return transition

    def train_single_episode(self, i_episode):
        self.env.reset()
        state = self.env.get_init_state()
        done = False
        while not done:
            transition = self.generate_one_step_transition(state)
            done = transition.done
            state = transition.next_state
            # self.running_loss += loss
            # self.running_reward += transition.reward
        return transition.reward
    def eval(self):
        for i_episode in tqdm(range(self.hps['num_episode'])):
            reward = self.train_single_episode(i_episode)
            self.logger.add_scalar('reward', reward, i_episode)
        self.logger.close()
        return self.model
    # def load_model(self):
    #     checkpoint_path = (Path(hps['log_dir']) / 'runs' / hps['envname']).resolve()
        
    #     return 
