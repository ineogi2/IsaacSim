# Isaac Sim simulation setting for RL-Lab

## Installation Guide

### Isaac Sim installation
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html

### Isaac Gym installation & Work Space Setting
For conda
```bash
mkdir ~/rl_ws/isaac && cd ~/rl_ws/isaac
conda create -n isaac_sim python=3.8 && conda activate isaac_sim && cd isaac_sim
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git && cd OmniIsaacGymEnvs
```

Open source file
```bash
For bash user : gedit ~/.bashrc
For zsh user : gedit ~/.zshrc
```

To set a Sim_PATH variable in the terminal that links to the python executable, add the following line. 
```bash
alias ‘SIM_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh'
```

Source terminal
```bash
For bash user : source ~/.bashrc
For zsh user : source ~/.zshrc
```

IsaacGym Installation
```bash
SIM_PATH -m pip install -e .
```
The following error may appear during the initial installation. This error is harmless and can be ignored.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
```

## Example Code
```bash
cd omniisaacgymenvs
CartPole : SIM_PATH scripts/rlgames_train.py task=Cartpole
Ant locomotion : SIM_PATH scripts/rlgames_train_mt.py task=Ant
```
```bash
If you want no rendering, add the argument : headless=True
```