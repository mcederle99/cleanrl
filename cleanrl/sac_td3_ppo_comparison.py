from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn
import tyro
from dataclasses import dataclass
import os
import cleanrl.sac_continuous_action
import cleanrl.td3_continuous_action


RIGHT_GOAL = 0.45
LEFT_GOAL = -1.5


@dataclass
class Sac_Args:
    exp_name: str = "sac_continuous_action"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    env_id: str = "MountainCarContinuous-v0"
    """the environment id of the task"""
    gamma: float = 0.99
    """the discount factor gamma"""


def evaluate_sac(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
    goal_position: float = -1000,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 1, 0, capture_video, run_name, goal_position)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _ = agent.get_action(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.detach().cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


sac_args = tyro.cli(Sac_Args)
run_name = f"{sac_args.env_id}__{sac_args.exp_name}__{sac_args.seed}__{1758612565}"
model_path = f"runs/{run_name}/{sac_args.exp_name}.cleanrl_model"
device = torch.device("cuda" if torch.cuda.is_available() and sac_args.cuda else "cpu")

print('====================SAC===================')
episodic_returns = evaluate_sac(
    model_path,
    cleanrl.sac_continuous_action.make_env,
    sac_args.env_id,
    eval_episodes=10,
    run_name=f"{run_name}-eval",
    Model=cleanrl.sac_continuous_action.Actor,
    device=device,
    gamma=sac_args.gamma,
    goal_position=LEFT_GOAL,
    capture_video=False,
)


@dataclass
class Td3_Args:
    exp_name: str = "td3_continuous_action"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    env_id: str = "MountainCarContinuous-v0"
    """the id of the environment"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""


def evaluate_td3(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    exploration_noise: float = 0.1,
    goal_position: float = -1000,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, goal_position)])
    actor = Model[0](envs).to(device)
    qf1 = Model[1](envs).to(device)
    qf2 = Model[1](envs).to(device)
    actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf1.load_state_dict(qf1_params)
    qf2.load_state_dict(qf2_params)
    qf1.eval()
    qf2.eval()
    # note: qf1 and qf2 are not used in this script

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


td3_args = tyro.cli(Td3_Args)
run_name = f"{td3_args.env_id}__{td3_args.exp_name}__{td3_args.seed}__{1758619103}"
model_path = f"runs/{run_name}/{td3_args.exp_name}.cleanrl_model"

print('====================TD3===================')
episodic_returns = evaluate_td3(
    model_path,
    cleanrl.td3_continuous_action.make_env,
    td3_args.env_id,
    eval_episodes=10,
    run_name=f"{run_name}-eval",
    Model=(cleanrl.td3_continuous_action.Actor, cleanrl.td3_continuous_action.QNetwork),
    device=device,
    exploration_noise=td3_args.exploration_noise,
    goal_position=LEFT_GOAL,
)