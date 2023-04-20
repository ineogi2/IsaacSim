import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from utils import ReplayBuffer
from models import Actor, Q_Network

class SAC(nn.Module):
    def __init__(self, env, device, cfg):
        super().__init__()
        self.state_size = env.observation_space.shape[0]
        # self.state_size = 1540
        self.action_size = env.action_space.shape[0]
        self.gamma = cfg.training.gamma
        self.dir = cfg.training.save_dir
        self.device = device

        self.buffer = ReplayBuffer(device, cfg)
        self.actor = Actor(self.state_size, self.action_size, cfg.model.hidden_sizes, action_space=env.action_space).to(device)

        self.q_1 = Q_Network(self.state_size, self.action_size, cfg.model.hidden_sizes).to(device)
        self.q_2 = Q_Network(self.state_size, self.action_size, cfg.model.hidden_sizes).to(device)
        self.q_1_target = Q_Network(self.state_size, self.action_size, cfg.model.hidden_sizes).to(device)
        self.q_2_target = Q_Network(self.state_size, self.action_size, cfg.model.hidden_sizes).to(device)

        # for tuning alpha
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.tensor(np.log(cfg.model.alpha), requires_grad=True, device=self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.training.learning_rate)
        self.q_optimizer = torch.optim.Adam(list(self.q_1.parameters())+list(self.q_2.parameters()), lr=cfg.training.learning_rate)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.training.learning_rate)

        self.load()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_q_target(self, target, source, tau):  # tau = 1 : hard update
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def get_action(self, state, evaluate=False):    # only for env.step(action)
        # state to Tensor
        if type(state) == torch.Tensor:
            state = state.reshape(-1, self.state_size).to(self.device)
        else:
            state = torch.tensor(state.reshape(-1, self.state_size), dtype=torch.float, device=self.device)

        # if training, rsample()
        # else, mean
        if not evaluate:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def train(self):
        # get batch
        states, actions, rewards, next_states, terminals = self.buffer.sample()

        # update Q-function
        with torch.no_grad():
            next_actions, next_log_prob_actions, _ = self.actor.sample(next_states)
            next_q_1_t, next_q_2_t = self.q_1_target(next_states, next_actions), self.q_2_target(next_states, next_actions)
            min_q_t = torch.min(next_q_1_t, next_q_2_t) - self.alpha.detach() * next_log_prob_actions
            q_target = rewards + self.gamma * (1-terminals) * min_q_t
        q_1, q_2 = self.q_1(states, actions), self.q_2(states, actions)
        q_1_loss = F.mse_loss(q_1, q_target)
        q_2_loss = F.mse_loss(q_2, q_target)
        q_loss = q_1_loss + q_2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # update Policy
        actions, log_prob_actions, _ = self.actor.sample(states)
        q_1, q_2 = self.q_1(states, actions), self.q_2(states, actions)
        current_q_val = torch.min(q_1, q_2)
        policy_loss = (-current_q_val + self.alpha.detach() * log_prob_actions).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        alpha_loss = -(self.log_alpha * (log_prob_actions + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.update_q_target(self.q_1_target, self.q_1, tau=0.01)
        self.update_q_target(self.q_2_target, self.q_2, tau=0.01)

        return q_1_loss.item(), q_2_loss.item(), policy_loss.item(), alpha_loss.item(), current_q_val.mean().item()
    
    def save(self):
        torch.save({
            'actor' : self.actor.state_dict(),
            'q_1' : self.q_1.state_dict(),
            'q_2' : self.q_2.state_dict(),
            'actor_optimizer' : self.actor_optimizer.state_dict(),
            'q_optimizer' : self.q_optimizer.state_dict(),
            'log_alpha_optimizer' : self.log_alpha_optimizer.state_dict()
            }, f'{self.dir}/agent')

    def load(self):
        if os.path.isdir(f'{self.dir}'):
            if os.path.isfile(f'{self.dir}/agent'):
                checkpoint = torch.load(f'{self.dir}/agent')
                self.actor.load_state_dict(checkpoint['actor'])
                self.q_1.load_state_dict(checkpoint['q_1'])
                self.q_2.load_state_dict(checkpoint['q_2'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
                self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
                print('[load] Agent')
            else:
                print('[new] Agent')
        else:
            print('[new] Agent')

        # hard update Q-target
        self.update_q_target(self.q_1_target, self.q_1, tau=1)
        self.update_q_target(self.q_2_target, self.q_2, tau=1)