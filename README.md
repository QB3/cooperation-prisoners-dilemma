# Cooperation Prisoners Dilemma
This is the code for the ICML 2025 paper [Self-Play Q-Learners Can Provably Collude in the Iterated Prisoner's Dilemma](https://hal.science/hal-05120296/document)

The [coop_rl_self_play_tabular_qlearning.ipynb](https://github.com/QB3/cooperation-prisoners-dilemma/blob/main/coop_rl_self_play_tabular_qlearning.ipynb) file reproduces Figure 3 of the paper: it shows the influence of the exploration parameter $\epsilon$ on the cooperation.

[run_quentin.sh](https://github.com/QB3/cooperation-prisoners-dilemma/blob/main/run_quentin.sh) implements self-play deep Q-learning with initialization against a random agent to reproduce Figure 5.

[coop_rl_ppo_deep_rl.ipynb](https://github.com/QB3/cooperation-prisoners-dilemma/blob/main/coop_rl_ppo_deep_rl.ipynb) is an additional file that investigates cooperation wth self-play PPO: it does not seem to yield cooperation.
