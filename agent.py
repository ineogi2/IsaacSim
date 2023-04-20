# agent for gym_Carla

import numpy as np
import torch
import tqdm, os
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchsummary import summary
from collections import deque

from models import Actor, Value
from utils import flatten_list_dicts, TransitionDataset


class Agent(nn.Module):
    def __init__(self, env, device, cfg):
        super().__init__()
        self.state_size = env.observation_space['state'].shape[0]
        self.action_size = env.action_space.shape[0]
        self.cfg = cfg
        self.device = device
        self.actor = Actor(self.state_size, self.action_size, cfg.model.hidden_sizes, conv=self.cfg.model.conv).to(device)
        self.critic = Value(self.state_size, cfg.model.hidden_sizes).to(device)
        self.optimizer = torch.optim.RMSprop(list(self.actor.parameters())+list(self.critic.parameters()), lr=cfg.reinforcement.learning_rate, alpha=0.9)
        self.load()

    def get_policy(self, state):
        policy = self.actor(state)
        return policy

    def get_action(self, state):
        policy = self.get_policy(state)
        action = policy.rsample()
        log_prob_action = policy.log_prob(action)
        return action, log_prob_action
    
    def get_value(self, state):
        return self.critic(state)
    
    def compute_advantages_(self, trajectories, next_value, discount, trace_decay):
        reward_to_go, advantage = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        trajectories['rewards_to_go'], trajectories['advantages'] = torch.empty_like(trajectories['rewards'], device=self.device), torch.empty_like(trajectories['rewards'], device=self.device)
        
        for t in reversed(range(trajectories['states'].size(0))):
            reward_to_go = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * (discount * reward_to_go)  # Reward-to-go/value R
            trajectories['rewards_to_go'][t] = reward_to_go
            td_error = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * discount * next_value - trajectories['values'][t]  # TD-error δ
            advantage = td_error + (1 - trajectories['terminals'][t]) * discount * trace_decay * advantage  # Generalised advantage estimate ψ
            trajectories['advantages'][t] = advantage
            next_value = trajectories['values'][t]
        # Normalise the advantages
        trajectories['advantages'] = (trajectories['advantages'] - trajectories['advantages'].mean()) / (trajectories['advantages'].std() + 1e-8)
    
    def ppo_update(self, trajectories, next_state, discount, trace_decay, ppo_clip, value_loss_coeff=1, entropy_loss_coeff=1, max_grad_norm=1):
        traj = dict()
        traj['states'] = trajectories['states'].detach().clone().to(self.device)
        traj['actions'] = trajectories['actions'].detach().clone().to(self.device)
        traj['old_log_prob_actions'] = trajectories['old_log_prob_actions'].detach().clone().to(self.device)
        traj['terminals'] = trajectories['terminals'].detach().clone().to(self.device)
        traj['rewards'] = trajectories['rewards'].detach().clone().to(self.device)
        traj['next_state'] = next_state.detach().clone().to(self.device)
        next_state = next_state.detach().clone().to(self.device)

        policy, traj['values'] = self.get_policy(traj['states']), self.get_value(traj['states'])
        traj['log_prob_actions'] = policy.log_prob(traj['actions'])
        with torch.no_grad():
            next_value = self.get_value(next_state)
            self.compute_advantages_(traj, next_value, discount, trace_decay)

        policy_ratio = (traj['log_prob_actions'] - traj['old_log_prob_actions']).exp()
        policy_loss = -torch.min(policy_ratio * traj['advantages'], torch.clamp(policy_ratio, min=1-ppo_clip, max=1+ppo_clip) * traj['advantages']).mean()
        value_loss = F.mse_loss(traj['values'], traj['rewards_to_go'])
        entropy_loss = -policy.entropy().mean()
        
        self.optimizer.zero_grad(set_to_none=True)
        loss = policy_loss + value_loss_coeff * value_loss + entropy_loss_coeff * entropy_loss
        loss.backward()
        clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_grad_norm)
        self.optimizer.step()
        return loss.detach().cpu().numpy()
    
    def train(self, policy_trajectories, next_state):
        training_loss = 0
        for _ in range(self.cfg.reinforcement.ppo_epochs):
            loss = self.ppo_update(policy_trajectories, next_state, self.cfg.reinforcement.discount,
                                    self.cfg.reinforcement.trace_decay, self.cfg.reinforcement.ppo_clip,
                                    self.cfg.reinforcement.value_loss_coeff, self.cfg.reinforcement.entropy_loss_coeff,
                                    self.cfg.reinforcement.max_grad_norm
                                    )
            training_loss += loss
        training_loss = training_loss/self.cfg.reinforcement.ppo_epochs
        return training_loss
    
    def save(self):
        torch.save({
            'actor' : self.actor.state_dict(),
            'critic' : self.critic.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, f'{self.cfg.training.save_dir}/agent')

    def load(self):
        if os.path.isdir(f'{self.cfg.training.save_dir}'):
            if os.path.isfile(f'{self.cfg.training.save_dir}/agent'):
                checkpoint = torch.load(f'{self.cfg.training.save_dir}/agent')
                self.actor.load_state_dict(checkpoint['actor'])
                self.critic.load_state_dict(checkpoint['critic'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print('[load] Agent')
            else:
                print('[new] Agent')
        else:
            print('[new] Agent')


class Discriminator(nn.Module):
    def __init__(self, env, device, cfg):
        super().__init__()
        # self.state_size = env.observation_space.shape[0]
        self.state_size = 1540
        self.action_size = env.action_space.shape[0]
        self.hidden_size = cfg.model.hidden_sizes
        self.policy_trajectory_replay_buffer = deque(maxlen=cfg.imitation.replay_size)
        self.cfg = cfg
        self.device = device
        self.state_only = cfg.imitation.state_only
        # self.discriminator = _create_fcnn(self.state_size if self.state_only else self.state_size + self.action_size, cfg.model.hidden_sizes, 1, 'tanh', conv=cfg.model.conv).to(device)
        
        self.discriminator = nn.Sequential(
        nn.Linear(self.state_size+self.action_size if not self.state_only else self.state_size, self.hidden_size[0]), nn.Tanh(), 
        nn.Linear(self.hidden_size[0], self.hidden_size[1]), nn.Tanh(),
        nn.Linear(self.hidden_size[1], 1), nn.Sigmoid()
        ).to(device)
        self.optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=cfg.imitation.learning_rate)
        self.load()

    def discriminate(self, state, action):
        D = self.discriminator(state if self.state_only else self._join_state_action(state, action)).squeeze(dim=1)
        return D
  
    def predict_reward(self, state, action):
        # D = torch.sigmoid(self.discriminate(state, action))
        D = self.discriminate(state, action)
        h = torch.log(D + 1e-6) - torch.log1p(-D + 1e-6) # Add epsilon to improve numerical stability given limited floating point precision
        return D
  
    def _join_state_action(self, state, action):
        state = state.reshape(state.shape[0], -1)
        return torch.cat([state, action], dim=1)
  
    def train(self, expert_trajectories, policy_trajectories):
        self.policy_trajectory_replay_buffer.append(policy_trajectories)
        policy_trajectories = TransitionDataset(flatten_list_dicts(self.policy_trajectory_replay_buffer))
        for _ in range(self.cfg.imitation.epochs):
            expert_dataloader = DataLoader(expert_trajectories, batch_size=self.cfg.training.batch_size, shuffle=True, drop_last=True, num_workers=4)
            policy_dataloader = DataLoader(policy_trajectories, batch_size=self.cfg.training.batch_size, shuffle=True, drop_last=True, num_workers=4)

            # Iterate over mininum of expert and policy data
            for expert_transition, policy_transition in zip(expert_dataloader, policy_dataloader):
                expert_state, expert_action = expert_transition['states'].detach().clone().to(self.device), expert_transition['actions'].detach().clone().to(self.device)
                policy_state, policy_action = policy_transition['states'].detach().clone().to(self.device), policy_transition['actions'].detach().clone().to(self.device)

                D_expert = self.discriminate(expert_state, expert_action)
                D_policy = self.discriminate(policy_state, policy_action)
            
                # Binary logistic regression
                self.optimizer.zero_grad(set_to_none=True)
                expert_loss = F.binary_cross_entropy_with_logits(D_expert, torch.ones_like(D_expert, device=self.device))  # Loss on "real" (expert) data
                # autograd.backward(expert_loss, create_graph=True)
                # r1_reg = 0
                # for param in self.discriminator.parameters():
                #   r1_reg += param.grad.norm()  # R1 gradient penalty
                policy_loss = F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy, device=self.device))  # Loss on "fake" (policy) data
                # (policy_loss + self.cfg.imitation.r1_reg_coeff * r1_reg).backward()
                loss = expert_loss + policy_loss
                loss.backward()
                self.optimizer.step()

    def save(self):
        torch.save({
            'discriminator' : self.discriminator.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, f'{self.cfg.training.save_dir}/discriminator')

    def load(self):
        if os.path.isdir(f'{self.cfg.training.save_dir}'):
            if os.path.isfile(f'{self.cfg.training.save_dir}/discriminator'):
                checkpoint = torch.load(f'{self.cfg.training.save_dir}/discriminator')
                self.discriminator.load_state_dict(checkpoint['discriminator'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print('[load] Discriminator')
            else:
                print('[new] Discriminator')
        else:
            print('[new] Discriminator')


class Encoder(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        # self.camera_size = env.observation_space['camera'].shape[2]     # 3 channel
        # self.lidar_size = env.observation_space['lidar'].shape[2]       # 3 channel
        # self.birdeye_size = env.observation_space['birdeye'].shape[2]   # 3 channel
        # self.state_size = env.observation_space['state'].shape[0]       # 4 values
        self.device = device
        self.cfg = cfg
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        self.encoder.fc = Identity()   # erase final linear layer
        self.encoder.eval()
        self.load()

    def forward(self, obs):
        camera = torch.tensor(obs['camera'], dtype=torch.float, device=self.device).reshape(-1, 3, 256, 256)
        lidar = torch.tensor(obs['lidar'], dtype=torch.float, device=self.device).reshape(-1, 3, 256, 256)
        birdeye = torch.tensor(obs['birdeye'], dtype=torch.float, device=self.device).reshape(-1, 3, 256, 256)
        state = torch.tensor(obs['state'], dtype=torch.float, device=self.device).reshape(-1, 4)

        latent_camera = self.encoder(camera).detach().clone()
        latent_lidar = self.encoder(lidar).detach().clone()
        latent_birdeye = self.encoder(birdeye).detach().clone()

        latent_state = torch.cat((latent_camera, latent_lidar, latent_birdeye, state), dim=1).reshape(-1, 1540)
        return latent_state
    
    def save(self):
        torch.save({
            'encoder' : self.encoder.state_dict(),
            }, f'{self.cfg.training.save_dir}/encoder')

    def load(self):
        if os.path.isdir(f'{self.cfg.training.save_dir}'):
            if os.path.isfile(f'{self.cfg.training.save_dir}/encoder'):
                checkpoint = torch.load(f'{self.cfg.training.save_dir}/encoder')
                self.discriminator.load_state_dict(checkpoint['encoder'])
                print('[load] Encoder')
            else:
                print('[new] Encoder')
        else:
            print('[new] Encoder')

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x