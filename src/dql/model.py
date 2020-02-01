import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dql.helper import Transition
# from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        random_idx = np.random.choice(
            np.arange(len(self.memory)), size=self.batch_size)
        res = [self.memory[i] for i in random_idx]
        return res

    def __len__(self):
        return len(self.memory)

# the architecture for both policy net and target net


class DQN(nn.Module):
    def __init__(self, h, w, n_actions, frame_length):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(frame_length, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.output_layer = nn.Linear(linear_input_size, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.output_layer(x.view(x.size(0), -1))

    def policy(self, state, eps):
        if np.random.sample() > eps:
            with torch.no_grad():
                return self(state).argmax(1)
        else:
            return torch.tensor([[np.random.choice(np.arange(self.n_actions))]], device=device, dtype=torch.long)

class Model:
    def __init__(self, env, hps):
        self.hps = hps
        # self.summary_writter = SummaryWriter(
        #     log_dir='runs/{}'.format(self.hps['envname']))
        self.policy_net = DQN(
            *self.hps['image_size'], env.action_space.n, self.hps['frame_length'])
        self.target_net = DQN(
            *self.hps['image_size'], env.action_space.n, self.hps['frame_length'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.loss = torch.empty(1)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.target_net.eval()
        self.policy_net.train()
        return

    def optimize_model(self, batch):
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.tensor(batch.done, device=device)
        current_q = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_state_batch)
        TD_target_batch = reward_batch + \
            (~done_batch).type(torch.float32) * \
            self.hps['gamma'] * next_q.max(dim=1).values
        self.loss = F.smooth_l1_loss(current_q, TD_target_batch)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        return
