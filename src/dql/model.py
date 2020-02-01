import torch 
import numpy as np 
import gym
from torch import nn
from itertools import count
from PIL import Image
import torch.nn.functional as F 
from torchvision import transforms as T
from collections import namedtuple, defaultdict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser
# BATCH_SIZE = 32
# hps = {
#     'gamma': 
# }
hps = defaultdict(int)

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1

LOGGING_INTERVAL = 1
IMAGE_SIZE = (84, 84)
FRAME_LENGTH = 4
# REPLAY_MEM_CAPACITY = 100
# NUM_EPISODE = 100
# ENVNAME = 'CartPole-v0'
Transition = namedtuple(
    'Transtition', ('state', 'action', 'reward', 'next_state', 'done'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class logger:
#     def __init__(self, agent, hps):
#         self.agent = 
#         return 

# just a implementation of random access queue
class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.batch_size = batch_size
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    # did find a torch equivalent to np.random.choice :(
    def sample(self):
        random_idx = np.random.choice(np.arange(len(self.memory)), size=self.batch_size)
        res = [self.memory[i] for i in random_idx]
        return res
    def __len__(self):
        return len(self.memory)

# the architecture for both policy net and target net
class DQN(nn.Module):
    def __init__(self, h, w, n_actions):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(FRAME_LENGTH, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3  = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride +1 
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.output_layer = nn.Linear(linear_input_size, n_actions)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.output_layer(x.view(x.size(0), -1))
    # Policy is kind of analogue to predict
    def policy(self, state, global_step):
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * global_step / EPS_DECAY)
        # take the greedy policy
        if np.random.sample() > eps:
            with torch.no_grad():
                return self(state).argmax(1)
        # or randomly select an action
        else:
            return torch.tensor([[np.random.choice(np.arange(self.n_actions))]], device=device, dtype=torch.long)

# ScreenProcessor will get you the current screen of envrionment. The image is transformed to graysacle and rescale to smaller size
class ScreenProcessor:
    def __init__(self, env):
        self.env = env
        self.transform = T.Compose([T.ToPILImage(),
                                T.Resize(IMAGE_SIZE, interpolation=Image.CUBIC),
                                T.Lambda(lambda x : T.functional.to_grayscale(x)),
                                T.ToTensor()])
        return 
    def get_screen(self):
        screen = self.env.render(mode='rgb_array')
        return self.transform(screen)
class Model:
    def __init__(self, env, summary_writter, hps=None):
        # define both policy net and target net
        self.policy_net = DQN(*IMAGE_SIZE, env.action_space.n)
        self.target_net = DQN(*IMAGE_SIZE, env.action_space.n)

        # make sure they have the same paremeters
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.loss = torch.empty(1)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.summary_writter = summary_writter

        return 
    
    def optimize_model(self, batch):
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.tensor(batch.done, device=device)

        current_q = self.policy_net(state_batch).gather(1,action_batch.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_state_batch)

        # done batch here is a mask, which activates when current next_state is not terminal state 
        TD_target_batch = reward_batch + (~done_batch).type(torch.float32) * GAMMA * next_q.max(dim=1).values
        self.loss = F.smooth_l1_loss(current_q, TD_target_batch)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        # self.running_loss += self.loss

        return self.loss
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        return 
    # def reset_running_loss(self):
    #     loss = self.loss
    #     self.loss = 0
    #     return loss

class Agent:
    def __init__(self, env, batch_size, replay_memory_capacity, frame_length, model, hps=None):
        self.env = env 
        self.frame_length = frame_length
        self.replay_memory = ReplayMemory(replay_memory_capacity, batch_size)
        self.screen_processor = ScreenProcessor(env)
        self.model = model
        self.running_loss = 0
        # self.stat = {
        #     'avg_loss' : None, 
        #     'avg_reward': None
        # }
        # self.policy = make_policy(self.model.target_net, env.action_space.n)
    def populate_replay_memory(self):
        state = self.get_init_state()
        for _ in range(self.replay_memory.capacity):
            action = self.model.target_net.policy(state.unsqueeze(0), 1).squeeze()
            _, reward, done, _ = self.env.step(action.item())
            reward = torch.tensor(reward, device=device)
            current_screen = self.screen_processor.get_screen()
            next_state = torch.cat([state[1:, ...], current_screen])
            self.replay_memory.push(state, action, reward, next_state, done)
            if done:
                state = self.get_init_state()
            else:
                state = next_state
        return 

    # the initial state is created by stacking same frames on each other
    def get_init_state(self):
        self.env.reset()
        current_screen = self.screen_processor.get_screen()
        state = torch.cat([current_screen] * self.frame_length, dim=0)
        return state
    def optimize(self):
        transitions = self.replay_memory.sample()
        batch = Transition(*zip(*transitions))
        return self.model.optimize_model(batch)
        # return loss
    def train(self, num_episode):
        global_step = 0
        running_reward = 0
        for i_episode in tqdm(range(num_episode)):
            self.env.reset()
            state = self.get_init_state()
            done = False
            while not done:
                action = self.model.policy_net.policy(state.unsqueeze(0), global_step).squeeze()
                global_step += 1
                _, reward, done, _ = self.env.step(action.item())
                running_reward += reward
                reward = torch.tensor(reward, device=device)
                current_screen = self.screen_processor.get_screen()
                next_state = torch.cat([state[1:, ...], current_screen])
                self.replay_memory.push(
                    state, action, reward, next_state, done)
                state = next_state
                loss = self.optimize()
                self.running_loss += loss
                
            if i_episode % TARGET_UPDATE == 0:
                self.model.update_target()
            if i_episode % LOGGING_INTERVAL == 0:
                self.model.summary_writter.add_scalar(
                    'running_loss', self.running_loss / LOGGING_INTERVAL, i_episode)
                self.model.summary_writter.add_scalar('running_reward', running_reward / LOGGING_INTERVAL, i_episode)
                running_reward = 0
                self.running_loss = 0
        return self.model
def parse_argument():
    parser = ArgumentParser('My Implementation of deep q learning')
    parser.add_argument('--envname', type=str, nargs='?', default='CartPole-v0', help='Need to be a valid gym atari environment')
    parser.add_argument('--batch_size', type=int, nargs='?', default=32, help='How many samples to gather from replay memory for each update')
    parser.add_argument('--num_episode', type=int, nargs='?', default=100, help='number of episode to train on, this does not include the episode generated while populate replay memory')
    parser.add_argument('--frame_length', type=int, nargs='?', default=4, help='this amount of frame with be stacked together to create a state')
    parser.add_argument('--replay_mem_cap', type=int, nargs='?', default=100, help='this argument defines the capacity of replay memory, the number must be strictly greater than the size of batch')

    parser.add_argument('--gamma', type=int, nargs='?', default=0.99, help='discount factor')
    parser.add_argument('--target_update', type=int, nargs='?', default=5, help='This argument specify how frequent target net will be updated')
    parser.add_argument('--log_interval', type=int, nargs='?', default=5, help='log will be written every <interval> episodes')
    parser.add_argument('--log_dir', type=str, nargs='?', default='./runs', help='logs will be placed in <log_dir>/<envname> directory ')
    # parser.add_argument('--image_size', type)
    # parser.add_argument('--ep', type=int, nargs=3, default=)


    
    args = parser.parse_args()
    if args.batch_size >= args.replay_mem_cap:
        raise ValueError('Capacity of replay memory must be strictly greater than batch size. Otherwise how would you sample from it?')
    return args
def main():
    args = parse_argument()
    # hps['gamma'] = args.gamma
    # hps['target_update'] = args.target_update
    # hps['']
    writter = SummaryWriter(log_dir='runs/{}'.format(args.envname))

    env = gym.make(args.envname)
    model = Model(env, writter)


    agent = Agent(env, args.batch_size, args.replay_mem_cap, args.frame_length, model)
    agent.populate_replay_memory()
    agent.train(args.num_episode)

    
    writter.close()
    env.close()
    return 
if __name__ == "__main__":
    main()

