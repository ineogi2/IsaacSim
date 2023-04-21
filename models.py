import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -4
epsilon = 1e-6

# Initialize weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes, action_space=None, alpha=None):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        self.fc_mean = nn.Linear(hidden_sizes[1], action_size)
        self.fc_log_std = nn.Linear(hidden_sizes[1], action_size)

        self.act_fn = torch.relu
        self.output_act_fn = torch.tanh

        self.apply(weights_init_)

        # action rescaling
        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))

        mean = self.output_act_fn(self.fc_mean(x))
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)

        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)


class Value(nn.Module):
    def __init__(self, state_size, hidden_sizes):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_mean = nn.Linear(hidden_sizes[1], 1)

        self.act_fn = torch.relu

        self.apply(weights_init_)

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc_mean(x)

        value = torch.reshape(x, (-1,))
        return value


class Q_Network(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes):
        super().__init__()

        self.fc1 = nn.Linear(state_size+action_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_mean = nn.Linear(hidden_sizes[1], 1)

        self.act_fn = torch.relu

        self.apply(weights_init_)

    def forward(self, state, action):
        # concat state - action pair
        x = torch.cat([state, action], dim=1)

        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc_mean(x)

        return x
