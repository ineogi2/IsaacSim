from collections import deque
import numpy as np
import torch
import wandb
import gym
import os
import sys
import datetime
sys.path.append('omniisaacgymenvs')

import hydra
from omegaconf import DictConfig

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

from sac import SAC

@hydra.main(config_name="config", config_path="omniisaacgymenvs/cfg")
def train(cfg: DictConfig):

    """ ------ Isaac GYM Setting ------ """
    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # headless = cfg.headless
    render = cfg.render
    cfg.headless = not render
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = rank
        cfg.rl_device = f'cuda:{rank}'
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    
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

    """ ------ SAC implementation Setting ------ """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'[{device}] used.')

    # Set gym-carla environment
    agent = SAC(env, device, cfg)

    # wandb & Tensorboard
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    if cfg.wandb and rank == 0:
        wandb.init(project='[torch] IsaacGym', entity='ineogi2', name=f'[{cfg.task_name}] {cfg.algorithm}-{cfg.model_num}', resume=True)

    """------------------------------------------"""

    while env._simulation_app.is_running():
        # main loop
        print("-----------------")
        print("----Main Loop----")
        epoch_bar = range(1, cfg.training.epochs+1)
        for epoch in epoch_bar:
            score = 0
            next_state = env.reset()['obs']
            for _ in range(1, cfg.training.max_steps+1):
                with torch.inference_mode():
                    state = next_state
                    action = agent.get_action(state)
                    next_state, reward, terminal, _ = env.step(torch.tensor([action], dtype=torch.float))
                    next_state = next_state['obs']
                    score += reward
                    agent.buffer.push(state, action, reward, next_state, terminal)

                if terminal:
                    next_state = env.reset()['obs']


            # Update models
            q_1_loss, q_2_loss, policy_loss, alpha_loss, q_val = agent.train()
            log_data = {'metrics/score':score, 'metrics/Q_value':q_val, 'loss/Q_1':q_1_loss, 'loss/Q_2':q_2_loss, 'loss/policy':policy_loss, 'loss/alpha':alpha_loss}
            print(log_data)
            # for wandb / Tensorborad
            if cfg.wandb: wandb.log(log_data)
            
            # save model
            if epoch % cfg.training.save_frequency == 0:
                agent.save()
                print('[success] save.')

    env._simulation_app.close()



if __name__ == '__main__':
    train()
