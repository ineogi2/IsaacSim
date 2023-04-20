import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import random
from matplotlib import pyplot as plt
from collections import deque

ACTIVATION_FUNCTIONS = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

class ReplayBuffer:
    def __init__(self, device, cfg) -> None:
        self.device = device
        self.batch_size = cfg.training.batch_size
        self.buffer = deque(maxlen=int(cfg.model.buffer_size))
    
    def push(self, state, action, reward, next_state, terminal):
        self.buffer.append([torch.tensor(state.reshape(1, -1), dtype=torch.float),
                            torch.tensor(action.reshape(1,-1), dtype=torch.float),
                            torch.tensor([reward], dtype=torch.float),
                            torch.tensor(next_state.reshape(1, -1), dtype=torch.float),
                            torch.tensor([terminal], dtype=torch.float)
                            ])
    
    def sample(self):
        batch_size = self.batch_size
        if len(self.buffer) < batch_size: batch_size=len(self.buffer)

        batch = random.sample(self.buffer, batch_size)

        # Dim = 1 : batch_size x tensor shape
        states = torch.cat([transition[0] for transition in batch], dim=0).to(self.device)
        actions = torch.cat([transition[1] for transition in batch], dim=0).to(self.device)
        rewards = torch.cat([transition[2] for transition in batch], dim=0).to(self.device).unsqueeze(dim=1)
        next_states = torch.cat([transition[3] for transition in batch], dim=0).to(self.device)
        terminals = torch.cat([transition[4] for transition in batch], dim=0).to(self.device).unsqueeze(dim=1)

        return states, actions, rewards, next_states, terminals

# Flattens a list of dicts with torch Tensors
def flatten_list_dicts(list_dicts):
    return {k: torch.cat([d[k] for d in list_dicts], dim=0) for k in list_dicts[-1].keys()}

# Makes a lineplot with scalar x and statistics of vector y
def lineplot(x, y, filename, xaxis='Steps', yaxis='Returns'):
    y = np.array(y)
    y_mean, y_std = y.mean(axis=1), y.std(axis=1)
    sns.lineplot(x=x, y=y_mean, color='coral')
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='coral', alpha=0.3)
    plt.xlim(left=0, right=x[-1])
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.savefig(f'{filename}.png')
    plt.close()

# Dataset that returns transition tuples of the form (s, a, r, s', terminal)
class TransitionDataset(Dataset):
    def __init__(self, transitions):
        super().__init__()
        self.states, self.actions, self.rewards, self.terminals = transitions['states'], transitions['actions'].detach(), transitions['rewards'], transitions['terminals']  # Detach actions

  # Allows string-based access for entire data of one type, or int-based access for single transition
    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == 'states':
                return self.states
            elif idx == 'actions':
                return self.actions
            elif idx == 'terminals':
                return self.terminals
        else:
            return dict(states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx], next_states=self.states[idx + 1], terminals=self.terminals[idx])

    def __len__(self):
        return self.terminals.size(0) - 1  # Need to return state and next state
    





# # Creates a sequential fully-connected network
# def _create_fcnn(input_size, hidden_sizes, output_size, activation_function, dropout=0, final_gain=1.0, conv=False):
#     assert activation_function in ACTIVATION_FUNCTIONS.keys()

#     # for 3D RGB-array state
#     if conv:
#         conv_layer1 = nn.Conv2d(3, 16, 3)
#         conv_layer2 = nn.Conv2d(16, 64, 3)
#         # conv_layer3 = nn.Conv2d(64, 128, 3)
#         # pooling_layer = nn.MaxPool2d(2)
#         adaptive_pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1,1))
#         layers = [conv_layer1, nn.ReLU(), conv_layer2, nn.ReLU(), adaptive_pooling_layer]
        
#         network_dims = [64]
#         for hidden_size in hidden_sizes:
#             network_dims.append(hidden_size)

#         for l in range(len(network_dims)-1):
#             layer = nn.Linear(network_dims[l], network_dims[l + 1])
#             nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation_function))
#             nn.init.constant_(layer.bias, 0)
#             layers.append(layer)
#             if dropout > 0: layers.append(nn.Dropout(p=dropout))
#             # layers.append(ACTIVATION_FUNCTIONS[activation_function]())
#             layers.append(nn.ReLU())
    
#   # for flatten RGB-array state
#     else:
#         network_dims, layers = [input_size], []

#         for hidden_size in hidden_sizes:
#             network_dims.append(hidden_size)

#         for l in range(len(network_dims) - 1):
#             layer = nn.Linear(network_dims[l], network_dims[l + 1])
#             nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation_function))
#             nn.init.constant_(layer.bias, 0)
#             layers.append(layer)
#             if dropout > 0: layers.append(nn.Dropout(p=dropout))
#             # layers.append(ACTIVATION_FUNCTIONS[activation_function]())
#             layers.append(nn.ReLU())

#     final_layer = nn.Linear(network_dims[-1], output_size)
#     final_activation = nn.Tanh()
#     nn.init.orthogonal_(final_layer.weight, gain=final_gain)
#     nn.init.constant_(final_layer.bias, 0)
#     layers.append(final_layer)
#     layers.append(final_activation)

#     return nn.Sequential(*layers)