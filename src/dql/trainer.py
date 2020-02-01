import torch 
import numpy as np 
import gym
from collections import defaultdict
from tqdm import tqdm
from dql.helper import create_eps_schedule, Transition
from dql.model import ReplayMemory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, env, model, hps, logger):
        self.logger = logger
        self.env = env 
        self.model = model
        self.hps = hps
        self.eps_schedule = create_eps_schedule(*tuple(self.hps['eps_schedule']))
        self.replay_memory = ReplayMemory(
            self.hps['replay_mem_cap'], self.hps['batch_size'])
        self.running_loss = 0
        self.running_reward = 0
        return 
    def sample_batch(self):
        transitions = self.replay_memory.sample()
        batch = Transition(*zip(*transitions))
        return batch
    def optimize(self):
        return self.model.optimize_model(self.sample_batch())
    
    def populate_replay_memory(self):
        state = self.env.get_init_state()
        for _ in range(self.replay_memory.capacity):
            transition = self.generate_one_step_transition(state, self.hps['eps_schedule'][0])
            if transition.done:
                state = self.env.get_init_state()
            else:
                state = transition.next_state
        return 
    def generate_one_step_transition(self, state, current_eps):
        action = self.model.policy_net.policy(
            state.unsqueeze(0), current_eps).squeeze()
        _, reward, done, _ = self.env.step(action.item())
        reward = torch.tensor(reward)
        current_screen = self.env.get_screen()
        next_state = torch.cat([state[1:, ...], current_screen])
        transition = Transition(state, action, reward, next_state, done)
        self.replay_memory.push(*transition)
        return transition
    def train_single_episode(self, i_episode):
        self.env.reset()
        state = self.env.get_init_state()
        done = False
        while not done:
            transition = self.generate_one_step_transition(
                state, self.eps_schedule(i_episode))
            done = transition.done
            state = transition.next_state
            loss = self.optimize()
            self.running_loss += loss
            self.running_reward += transition.reward
        return 
    def train(self):
        self.populate_replay_memory()
        sample_inputs = torch.stack(self.sample_batch().state)

        self.logger.add_graph(self.model, input_to_model=sample_inputs)
        for i_episode in tqdm(range(self.hps['num_episode'])):
            self.train_single_episode(i_episode)
            if i_episode % self.hps['log_interval'] == 0:
                self.logger.add_scalar(
                    'running_loss', self.running_loss / i_episode, i_episode)
                self.logger.add_scalar(
                    'running_reward', self.running_reward / i_episode, i_episode)
                self.running_reward = 0
                self.running_loss = 0
            if i_episode % self.hps['checkpoint_interval'] == 0:
                self.logger.save_checkpoint(self.model, i_episode)
        self.logger.close()
        return self.model
