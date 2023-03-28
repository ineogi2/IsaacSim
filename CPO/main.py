from logger import Logger
from agent import Agent

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import argparse
import pickle
import random
import torch
import wandb
import copy
import time
import gym
import os
import datetime
import sys
sys.path.append('../')

import hydra
from omegaconf import DictConfig

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

@hydra.main(config_name="config", config_path="../omniisaacgymenvs/cfg")
def train(cfg: DictConfig):

    """ ------ Isaac GYM Setting ------ """
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = not headless
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = rank
        cfg.rl_device = f'cuda:{rank}'
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    
    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()    
    
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)

    """---------------------------------"""

    """ ------ CPO implementation Setting ------ """
    algo_idx = 1
    agent_name = 'isaac_CPO'
    env_name = cfg.task_name
    max_ep_len = 500
    max_steps = 1000
    epochs = 2500
    save_freq = 10
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = '_'.join(env_name.split('-')[:-1])
    save_name = "result/{}_{}".format(save_name, algo)
    args = {
        'agent_name':agent_name,
        'save_name': save_name,
        'discount_factor':0.99,
        'hidden1':512,
        'hidden2':512,
        'v_lr':2e-4,
        'cost_v_lr':2e-4,
        'value_epochs':200,
        'batch_size':10000,
        'num_conjugate':10,
        'max_decay_num':10,
        'line_decay':0.8,
        'max_kl':0.001,
        'damping_coeff':0.01,
        'gae_coeff':0.97,
        'cost_d':25.0/1000.0,
    }
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    agent = Agent(env, device, args)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        run_name = f"{cfg.task_name}_{time_str}"

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
        )
    """------------------------------------------"""

    while env._simulation_app.is_running():
        for epoch in range(epochs):
            trajectories = []
            ep_step = 0
            scores = []

            while ep_step < max_steps:
                states = env.reset()
                states = states['obs']
                states_tensor = torch.tensor(states, device=device, dtype=torch.float)
                score = 0
                cv = 0
                step = 0
                while step<max_ep_len:
                    ep_step += 1
                    step += 1
                    # states_tensor = torch.tensor(states, device=device, dtype=torch.float)
                    # states_tensor = states.clone().detach()
                    actions, _ = agent.getAction(states_tensor, is_train=True)
                    actions_numpy = actions.detach().cpu().numpy()

                    next_states, rewards, dones, _ = env.step(actions)
                    next_states = torch.tensor(next_states['obs'], device=device, dtype=torch.float)
                    # print(next_states.size(), rewards.size(), dones.size())
                    cost = -rewards

                    trajectories.append([states_tensor.cpu().numpy(), actions_numpy, rewards.cpu(), cost.cpu(), dones.cpu(), torch.zeros(rewards.size()).cpu(), next_states.cpu().numpy()])
                    states = next_states
                    score += torch.sum(rewards).item()/env.num_envs
            # print(trajectories[0])
            states = np.array([traj[0] for traj in trajectories])
            print(states.shape)
            scores.append(score)
            v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy = agent.train(trajs=trajectories)
            score = np.mean(scores)
            log_data = {"score":score, 'cv':cv, "value loss":v_loss, "cost value loss":cost_v_loss, "objective":objective, "cost surrogate":cost_surrogate, "kl":kl, "entropy":entropy}
            
            print(f'\nEpoch {epoch+1}/{epochs} result :')
            print(log_data)

    env._simulation_app.close()



if __name__ == '__main__':
    train()
