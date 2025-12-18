import hydra
import random
import sys
import torch
import wandb
import numpy as np

from omegaconf import DictConfig, OmegaConf

from .agents import DQNAgent, RandomAgent
from .algos import *
from .ipd import IPD

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(config_path="../scripts", config_name="config", version_base=None)
def main(args: DictConfig):
    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)

    run = wandb.init(
        project="Cooperation Q-learning",
        dir=config["wandb_dir"],
        config=config
    )

    seed_all(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(config)

    batch_size = config["batch_size"]
    pretrain_vs_random = config["pretrain_vs_random"]
    do_self_play = config["do_self_play"]
    pretrain_vs_supervised = config["supervised_pretrain"]
    use_ipd_g_param = config["use_ipd_g_param"]
    ipd_g = config["ipd_g"]

    env = IPD(device, batch_size, use_ipd_g_param, g=ipd_g)
    random_agent = RandomAgent(config["num_actions"], device, batch_size)
    agent_1 = DQNAgent(**config["agent"], optim_config=config["optim"], device=device, batch_size=batch_size)

    if do_self_play:
        agent_2 = agent_1
    else:
        agent_2 = DQNAgent(**config["agent"], optim_config=config["optim"], device=device, batch_size=batch_size)

    if pretrain_vs_random:
        run_dqn(env,
                agent_1,
                random_agent,
                device,
                batch_size,
                num_iters=config["pretrain_iters"],
                tau=config["tau"],
                enable_wandb=True)
        agent_1.steps_done = 0
    elif pretrain_vs_supervised:
        supervised_pretrain(
            agent_1, config["shift_c"], config["shift_d"],
            config["pretrain_iters"],
            rdd=env.mat[0, 0], rcd=env.mat[0, 1])


    if pretrain_vs_random and not do_self_play:
        run_dqn(env,
                agent_2,
                random_agent,
                device,
                batch_size,
                num_iters=config["pretrain_iters"],
                tau=config["tau"],
                enable_wandb=True)
        agent_2.steps_done = 0
    elif pretrain_vs_supervised and not do_self_play:
        supervised_pretrain(
            agent_2, config["shift_c"], config["shift_d"],
            config["pretrain_iters"],
            rdd=env.mat[0, 0], rcd=env.mat[0, 1])

    run_dqn(env,
            agent_1,
            agent_2,
            device,
            batch_size,
            num_iters=config["num_iters"],
            tau=config["tau"],
            do_self_play=do_self_play)

if __name__ == "__main__":
    main()
