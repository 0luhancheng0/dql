import torch 
import numpy as np 
import gym
from itertools import count
from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dql.helper import create_eps_schedule, Transition
from dql.model import Model, ReplayMemory
from dql.environment import Environment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, env, model, hps):
        self.env = env 
        self.model = model
        self.hps = hps
        self.eps_schedule = create_eps_schedule(*tuple(self.hps['eps_schedule']))
        self.replay_memory = ReplayMemory(
            self.hps['replay_mem_cap'], self.hps['batch_size'])
        self.running_loss = 0
        self.running_reward = 0
        return 
    def optimize(self):
        transitions = self.replay_memory.sample()
        batch = Transition(*zip(*transitions))
        return self.model.optimize_model(batch)
    
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
    def train_single_episode(self, global_step):
        self.env.reset()
        state = self.env.get_init_state()
        done = False
        while not done:
            transition = self.generate_one_step_transition(
                state, self.eps_schedule(global_step))
            done = transition.done
            state = transition.next_state
            loss = self.optimize()
            self.running_loss += loss
            # global_step += 1
        return 
    def train(self):
        self.populate_replay_memory()
        global_step = 0
        running_reward = 0
        for i_episode in tqdm(range(self.hps['num_episode'])):
            self.train_single_episode(global_step)
            global_step += 1
            if i_episode % self.hps['target_update'] == 0:
                self.model.update_target()
            if i_episode % self.hps['log_interval'] == 0:
                self.model.summary_writter.add_scalar(
                    'running_loss', self.running_loss / self.hps['log_interval'], i_episode)
                self.model.summary_writter.add_scalar(
                    'running_reward', running_reward / self.hps['log_interval'], i_episode)
                running_reward = 0
                self.running_loss = 0
        return self.model

def parse_argument():
    parser = ArgumentParser('My Implementation of deep q learning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--envname', type=str, nargs='?', default='CartPole-v0', help='Need to be a valid gym atari environment')
    parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=32, help='How many samples to gather from replay memory for each update')
    parser.add_argument('-n', '--num_episode', type=int, nargs='?', default=100, help='number of episode to train on, this does not include the episode generated while populate replay memory')
    parser.add_argument('--frame_length', type=int, nargs='?', default=4, help='this amount of frame with be stacked together to create a state')
    parser.add_argument('--replay_mem_cap', type=int, nargs='?', default=100, help='this argument defines the capacity of replay memory, the number must be strictly greater than the size of batch')

    parser.add_argument('--gamma', type=float, nargs='?', default=0.99, help='discount factor')
    parser.add_argument('--target_update', type=int, nargs='?', default=5, help='This argument specify how frequent target net will be updated')
    parser.add_argument('--log_interval', type=int, nargs='?', default=5, help='log will be written every <interval> episodes')
    parser.add_argument('--log_dir', type=str, nargs='?', default='./runs', help='logs will be placed in <log_dir>/<envname> directory ')
    parser.add_argument('--eps_schedule', type=float, nargs='*', default=[0.9, 0.05, 200], help='consume 3 values which defines the epsilon decay schedule (start, end, steps)')
    parser.add_argument('--image_size', type=int, nargs='*', default=[84, 84], help='Size of cropped image')
    
    args = parser.parse_args()
    if args.batch_size >= args.replay_mem_cap:
        raise ValueError('Capacity of replay memory must be strictly greater than batch size. Otherwise how would you sample from it?')
    if len(args.eps_schedule) != 3:
        raise ValueError('epsilon schedule must consume 3 values (start, end, step)')
    assert len(args.image_size) == 2
    return args
def main():
    args = parse_argument()
    hps = vars(args)
    env = Environment(hps)
    model = Model(env, hps)
    trainer = Trainer(env, model, hps)
    trainer.train()

    
    model.summary_writter.close()
    env.close()
    return 
if __name__ == "__main__":
    main()

