import random
import torch
import wandb

from collections import defaultdict

import torch.nn.functional as F

def log_to_wandb(wandb_info):
    wandb.log(wandb_info)

def get_code_(tensor, device):
    coop = torch.tensor([1., 0.]).to(device)
    if torch.equal(tensor, coop):
        return "C"
    else:
        return "D"

def get_code(tensor, device):
    prev_a1 = tensor[0:2]
    prev_a2 = tensor[2:4]
    a1 = tensor[4:6]
    return get_code_(a1, device), get_code_(prev_a1, device), get_code_(prev_a2, device)

def set_wandb_info(wandb_info, q_loss_1, q_loss_2, rewards_1, rewards_2, agent_1, agent_2, gn_1, gn_2):
    wandb_info["a1-q_loss"] = q_loss_1 if q_loss_1==None else q_loss_1.detach()
    wandb_info["a2-q_loss"] = q_loss_2 if q_loss_2==None else q_loss_2.detach()
    wandb_info["a1-avg_reward"] = rewards_1.mean().detach()
    wandb_info["a2-avg_reward"] = rewards_2.mean().detach()
    wandb_info["a1-grad_norm"] = gn_1
    wandb_info["a2-grad_norm"] = gn_2
    compute_q_values(wandb_info, agent_1, agent_2)

def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_all_states_ipd(num_actions, device):
    states = []
    for j in range(2):
        for k in range(2):
            prev_a1 = F.one_hot(torch.tensor([j]), num_actions).float().to(device)
            prev_a2 = F.one_hot(torch.tensor([k]), num_actions).float().to(device)
            state = torch.cat([prev_a1, prev_a2], dim=1).float()
            states.append(state)
    return states

def compute_q_values(wandb_info, agent_1, agent_2):
    device = agent_1.device
    num_actions = agent_1.num_actions
    for i, agent in enumerate([agent_1, agent_2]):
        if agent.optimizer:
            for j in range(2):
                for k in range(2):
                    prev_a1 = F.one_hot(torch.tensor([j]), num_actions).float().to(device)
                    prev_a2 = F.one_hot(torch.tensor([k]), num_actions).float().to(device)
                    state = torch.cat([prev_a1, prev_a2], dim=1).float()
                    qc, qd = agent.q_net(state)[0].detach()
                    wandb_info[f"a{i+1}-q(C|{get_code_(prev_a1[0], device)}{get_code_(prev_a2[0], device)})"] = qc
                    wandb_info[f"a{i+1}-q(D|{get_code_(prev_a1[0], device)}{get_code_(prev_a2[0], device)})"] = qd
                    wandb_info[f"a{i+1}-[q(C|{get_code_(prev_a1[0], device)}{get_code_(prev_a2[0], device)})"
                              + f" - q(D|{get_code_(prev_a1[0], device)}{get_code_(prev_a2[0], device)})]"] = qc - qd

def update_state_dict(model, state_dict, tau=1):
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)

def compute_empirical_prob(prev_state, actions, wandb_info, agent_id, device):
    prev_actions_actions = torch.cat((prev_state, actions), axis=1)
    tensors, counts = prev_actions_actions.unique(return_counts=True, dim=0)

    dict_metrics = {}
    dict_normalization_cst = {}
    dict_normalization_cst = defaultdict(lambda: 0, dict_normalization_cst)
    for idx in range(tensors.shape[0]):
        a1, prev_a1, prev_a2 = get_code(tensors[idx, :], device)
        dict_metrics[a1, prev_a1, prev_a2] = counts[idx].double()
        dict_normalization_cst[prev_a1, prev_a2] += counts[idx].double()
    for key in dict_metrics.keys():
        dict_metrics[key] /= dict_normalization_cst[(key[1], key[2])]

    for key in dict_metrics.keys():
        wandb_info[f"a{agent_id}-p({key[0]}|{key[1]}{key[2]})"] = dict_metrics[key]
