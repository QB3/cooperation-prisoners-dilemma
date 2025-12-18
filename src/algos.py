import hydra
import numpy as np
import random
import torch
import wandb

from tqdm import tqdm
from .utils import *

def optimize_model(optimizer, loss, model):
    optimizer.zero_grad()
    loss.backward()
    norm = get_gradient_norm(model)
    optimizer.step()
    return norm

def supervised_pretrain(
        agent, shift_q_val_c, shift_q_val_d, num_iters, rdd, rcd):
    q_val_star_c = 0
    q_val_star_d = 0
    # q_val_star_d = rdd / (1 - agent.gamma)
    # q_val_star_c = rcd + agent.gamma * rdd / (1 - agent.gamma)
    print("Q values at optimum for defecting:%.2f" % q_val_star_d)
    print("Q values at optimum for cooperating:%.2f" % q_val_star_c)
    for iter in tqdm(
            range(num_iters), desc='Supervised pre-training iterations'):
        states = torch.stack(
            get_all_states_ipd(agent.num_actions, agent.device))
        est_vals = agent.q_net(states)

        supervised_loss_c = torch.nn.functional.mse_loss(
            est_vals[:, :, 0],
            (shift_q_val_c + q_val_star_c) * torch.ones_like(est_vals[:, :, 0]))
        supervised_loss_d = torch.nn.functional.mse_loss(
            est_vals[:, :, 1],
            (shift_q_val_d + q_val_star_d) * torch.ones_like(est_vals[:, :, 1]))

        total_loss = supervised_loss_c + supervised_loss_d
        optimize_model(agent.optimizer, total_loss, agent.q_net)


def run_dqn(env, agent_1, agent_2, device, batch_size, num_iters, tau, enable_wandb=True, do_self_play=False):
    state_1 = torch.Tensor([0, 0, 0, 0]).repeat(batch_size, 1).to(device)
    state_2 = state_1

    wandb_info={}

    grad_norm_1, grad_norm_2 = None, None

    for iter in range(num_iters):
        prev_state_1 = state_1.clone()
        prev_state_2 = state_2.clone()

        actions_1 = agent_1.select_actions(state_1)
        actions_2 = agent_2.select_actions(state_2)

        obs, rewards, _, _ = env.step([actions_1, actions_2])
        state_1, state_2 = obs
        rewards_1, rewards_2 = rewards

        # s, a, s'
        transitions_1 = [list(x) for x in zip(prev_state_1.tolist(), actions_1.tolist(), state_1.tolist(), rewards_1.tolist())]
        transitions_2 = [list(x) for x in zip(prev_state_2.tolist(), actions_2.tolist(), state_2.tolist(), rewards_2.tolist())]

        agent_1.store_transitions(transitions_1)
        agent_2.store_transitions(transitions_2)

        q_loss_1 = agent_1.compute_q_loss()
        q_loss_2 = agent_2.compute_q_loss()

        if agent_1.optimizer:
            grad_norm_1 = optimize_model(agent_1.optimizer, q_loss_1, agent_1.q_net)
            update_state_dict(agent_1.t_net, agent_1.q_net.state_dict(), tau)
            # agent_1.t_net.load_state_dict(agent_1.q_net.state_dict())

        if agent_2.optimizer and not do_self_play:
            grad_norm_2 = optimize_model(agent_2.optimizer, q_loss_2, agent_2.q_net), agent_2.q_net
            update_state_dict(agent_2.t_net, agent_2.q_net.state_dict(), tau)
            # agent_2.t_net.load_state_dict(agent_2.q_net.state_dict())

        if iter != 0:
            compute_empirical_prob(prev_state_1, actions_1, wandb_info, 1, agent_1.device)
            compute_empirical_prob(prev_state_2, actions_2, wandb_info, 2, agent_2.device)

        if enable_wandb:
            set_wandb_info(wandb_info,
                           q_loss_1,
                           q_loss_2,
                           rewards_1,
                           rewards_2,
                           agent_1,
                           agent_2,
                           grad_norm_1,
                           grad_norm_2)
            log_to_wandb(wandb_info)
