import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple
from functools import reduce

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Qnetwork(nn.Module):
    def __init__(self, in_size, out_size, device, hidden_size=40):
        super(Qnetwork, self).__init__()
        self.in_layer = nn.Linear(in_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        return self.out_layer(x)

class RandomAgent():
    def __init__(self,
                 num_actions,
                 device,
                 batch_size):
        self.num_actions = num_actions
        self.device = device
        self.batch_size = batch_size
        self.optimizer = None

    def store_transitions(self, transitions):
        pass

    def compute_q_loss(self):
        pass

    def select_actions(self, state):
        random_actions = torch.multinomial(
            torch.ones(self.batch_size, self.num_actions) / self.num_actions,
            num_samples=1).reshape(self.batch_size,).to(self.device)

        return F.one_hot(random_actions, self.num_actions)

class DQNAgent():
    def __init__(self,
                 num_actions,
                 in_size,
                 out_size,
                 hidden_size,
                 capacity,
                 gamma,
                 epsilon,
                 decay_eps,
                 eps_start,
                 eps_end,
                 eps_decay,
                 optim_type,
                 loss_type,
                 optim_config,
                 device,
                 batch_size):

        self.steps_done = 0
        self.num_actions = num_actions
        self.capacity = capacity
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_eps = decay_eps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.optim_type = optim_type
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.q_net = Qnetwork(in_size, out_size, device, hidden_size)
        self.t_net = Qnetwork(in_size, out_size, device, hidden_size)
        self.q_net.to(device)
        self.t_net.to(device)
        self.buffer = ReplayMemory(capacity)

        if self.optim_type.lower() == "adam":
            self.optimizer = optim.Adam(list(self.q_net.parameters()),
                                        lr=optim_config["lr"],
                                        weight_decay=optim_config["weight_decay"],
                                        maximize=False)
        elif self.optim_type.lower() == "sgd":
            self.optimizer = optim.SGD(list(self.q_net.parameters()),
                                       lr=optim_config["lr"],
                                       weight_decay=optim_config["weight_decay"],
                                       maximize=False)


    def store_transitions(self, transitions):
        for transition in transitions:
            self.buffer.push(Transition(state=transition[0],
                                        action=transition[1],
                                        next_state=transition[2],
                                        reward=transition[3]))

    def select_actions(self, state):
        """epsilon has to be between 0 and .5"""
        self.steps_done += 1
        greedy_actions = torch.argmax(self.q_net(state), dim=1)
        if self.decay_eps:
            self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)

        mask_actions = (torch.rand(self.batch_size) < 1 - 2 * self.epsilon).to(self.device)

        random_actions = torch.multinomial(
            torch.ones(self.batch_size, self.num_actions) / self.num_actions,
            num_samples=1).reshape(self.batch_size,).to(self.device)

        real_actions = (greedy_actions * mask_actions)
        real_actions += random_actions * (~mask_actions)
        return torch.nn.functional.one_hot(real_actions, self.num_actions)

    def compute_q_loss(self):
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.Tensor(batch.state).to(self.device)
        actions = torch.argmax(torch.Tensor(batch.action).to(self.device), dim=1)
        next_states = torch.Tensor(batch.next_state).to(self.device)
        rewards = torch.Tensor(batch.reward).to(self.device)

        s_a_vals = torch.gather(self.q_net(states), 1, actions.reshape(self.batch_size, 1)).reshape(self.batch_size)
        n_s_a_vals, n_s_a_vals_indices = torch.max(self.t_net(next_states), dim=1)

        # import pdb; pdb.set_trace()
        if self.loss_type.lower() == "huber":
            loss = F.huber_loss
        elif self.loss_type.lower() == "mse":
            loss = F.mse_loss
        return loss(s_a_vals, rewards + self.gamma * n_s_a_vals.detach())
