# HW3: Function Approximation RL for Cart-Pole Stabilization

FRA503 Deep Reinforcement Learning for Robotics

| Name                 | Student ID   |
| -------------------- | ------------ |
| Chantouch Orungrote  | 66340500011  |
| Sasish Kaewsing      | 66340500076  |

## Overview

This homework implements **8 function approximation-based RL algorithms** for the Cart-Pole Stabilization task using Isaac Lab, transitioning from the tabular methods of HW2 to neural and linear function approximators capable of handling continuous state spaces.

### Algorithms Implemented

| Algorithm      | Type          | Policy        | On/Off-Policy | Action Space |
| -------------- | ------------- | ------------- | ------------- | ------------ |
| Linear_QN      | Value-based   | Deterministic (ε-greedy) | Off-policy    | Discrete (11) |
| DQN            | Value-based   | Deterministic (ε-greedy) | Off-policy    | Discrete (11) |
| MC_REINFORCE   | Policy-based  | Stochastic    | On-policy     | Discrete (11) |
| AC             | Actor-Critic  | Stochastic    | On-policy     | Continuous    |
| A2C            | Actor-Critic  | Stochastic    | On-policy     | Continuous    |
| PPO            | Actor-Critic  | Stochastic    | On-policy     | Continuous    |
| TD3            | Actor-Critic  | Deterministic | Off-policy    | Continuous    |
| SAC            | Actor-Critic  | Stochastic    | Off-policy    | Continuous    |

### Class Hierarchy

```
BaseAlgorithm  (RL_base_function.py)
├── OnPolicyAlgorithm  (storage/on_policy.py)
│   ├── AC              — Monte Carlo episodic actor-critic
│   ├── A2C             — TD synchronous advantage actor-critic
│   └── PPO             — Clipped surrogate + GAE
├── OffPolicyAlgorithm  (storage/off_policy.py)
│   ├── DQN             — Deep Q-Network with experience replay
│   ├── TD3             — Twin Delayed DDPG
│   └── SAC             — Soft Actor-Critic (max entropy)
└── (direct)
    ├── Linear_QN       — Linear function approximation Q-Learning
    └── MC_REINFORCE    — Vanilla policy gradient with MC returns
```

---

## Setup

```bash
conda activate env_isaaclab
cd ~/FRA503_Deep_Reinforcement_Learning/CartPole_4.5.0
```

---

## Configuration

### Shared Parameters

| Parameter         | Value        | Description                              |
| ----------------- | ------------ | ---------------------------------------- |
| `n_episodes`      | 10000        | Total training episodes                  |
| `max_steps`       | 500          | Maximum steps per episode                |
| `n_observations`  | 4            | Observation space dimension (x, θ, ẋ, θ̇) |
| `action_range`    | [-10.0, 10.0]| Continuous force range (N)               |
| `learning_rate`   | 3e-4         | Base optimizer learning rate             |
| `discount_factor` | 0.99         | Discount factor (γ)                      |

### Per-Algorithm Hyperparameters

#### Value-Based Methods

| Parameter           | Linear_QN | DQN     |
| ------------------- | --------- | ------- |
| `num_of_action`     | 11        | 11      |
| `learning_rate`     | 1e-3      | 3e-4    |
| `initial_epsilon`   | 1.0       | 1.0     |
| `epsilon_decay`     | 5e-4      | 5e-4    |
| `final_epsilon`     | 0.01      | 0.01    |
| `hidden_dim`        | —         | 128     |
| `dropout`           | —         | 0.1     |
| `tau`               | —         | 0.005   |
| `buffer_size`       | —         | 50000   |
| `batch_size`        | —         | 128     |
| `update_freq`       | —         | 4       |
| `target_update_freq`| —         | 200     |

#### Policy Gradient

| Parameter       | MC_REINFORCE |
| --------------- | ------------ |
| `num_of_action` | 11           |
| `hidden_dim`    | 128          |
| `dropout`       | 0.1          |

#### On-Policy Actor-Critic

| Parameter                  | AC    | A2C   | PPO   |
| -------------------------- | ----- | ----- | ----- |
| `num_of_action`            | 1     | 1     | 1     |
| `hidden_dim`               | 256   | 256   | 256   |
| `entropy_coef`             | 0.01  | 0.01  | 0.01  |
| `gae_lambda`               | —     | 0.95  | 0.95  |
| `value_loss_coef`          | —     | 0.5   | 0.5   |
| `clip_param`               | —     | —     | 0.2   |
| `num_transitions_per_env`  | —     | 64    | 64    |
| `num_learning_epochs`      | —     | —     | 4     |
| `num_mini_batches`         | —     | —     | 4     |

#### Off-Policy Actor-Critic

| Parameter       | TD3       | SAC       |
| --------------- | --------- | --------- |
| `num_of_action` | 1         | 1         |
| `hidden_dim`    | 256       | 256       |
| `tau`           | 0.005     | 0.005     |
| `buffer_size`   | 100,000   | 100,000   |
| `batch_size`    | 256       | 256       |
| `update_freq`   | 4         | 4         |
| `policy_noise`  | 0.2       | —         |
| `noise_clip`    | 0.5       | —         |
| `expl_noise`    | 0.1       | —         |
| `policy_delay`  | 2         | —         |
| `alpha`         | —         | 0.2       |
| `auto_entropy`  | —         | True      |

---

## Train

### Train All Algorithms (Recommended)

Trains all 8 algorithms sequentially and generates all comparison and deployment plots:

```bash
python scripts/Function_based/train_all.py \
    --task Stabilize-Isaac-Cartpole-v0 \
    --headless \
    --num_envs 1
```

Optional arguments:

| Argument            | Default | Description                          |
| ------------------- | ------- | ------------------------------------ |
| `--num_envs`        | 1       | Number of parallel environments      |
| `--max_iterations`  | None    | Override default episode count       |
| `--deploy_episodes` | 100     | Episodes for deployment evaluation   |
| `--seed`            | random  | Random seed for reproducibility      |
| `--video`           | False   | Enable video recording               |

### Train Single Algorithm

```bash
python scripts/Function_based/train.py \
    --task Stabilize-Isaac-Cartpole-v0 \
    --headless --num_envs 1
```

---

## Play (Deploy Trained Policy)

```bash
python scripts/Function_based/play.py \
    --task Stabilize-Isaac-Cartpole-v0 \
    --num_envs 1 --headless
```

Deployment uses deterministic action selection for all algorithms:

| Algorithm      | Inference Mode                               |
| -------------- | -------------------------------------------- |
| Linear_QN, DQN | ε = 0 (greedy argmax Q)                     |
| MC_REINFORCE   | Argmax over softmax probabilities            |
| AC, A2C, PPO   | `act_inference()` (deterministic mean action)|
| TD3            | `select_action(noise=0.0)`                   |
| SAC            | `select_action(deterministic=True)`          |

---

## Results

### Output Structure

```
plots/{task}/Function_based/
├── Linear_QN/                      # Per-algorithm plots
│   ├── learning_curve.png          #   [A] Reward per episode (raw + smoothed)
│   ├── episode_length_curve.png    #   [B] Steps survived per episode
│   ├── reward_with_std.png         #   [C] Smoothed reward ± 1-std band
│   ├── actor_loss_curve.png        #   [D] Actor loss over update steps *
│   ├── critic_loss_curve.png       #   [E] Critic/value loss over steps *
│   ├── epsilon_curve.png           #   [F] Epsilon decay schedule **
│   ├── entropy_curve.png           #   [G] Policy entropy ***
│   └── steps_vs_reward.png         #   [H] Reward vs total env steps
├── DQN/
├── MC_REINFORCE/
├── AC/
├── A2C/
├── PPO/
├── TD3/
├── SAC/
├── comparisons/                    # Cross-algorithm comparisons
│   ├── comparison_reward.png           # [I]  All algos smoothed reward
│   ├── comparison_ep_length.png        # [J]  All algos episode length
│   ├── comparison_steps_reward.png     # [K]  Reward vs env steps (sample efficiency)
│   ├── comparison_actor_loss.png       # [L]  Actor loss (neural algos)
│   ├── comparison_critic_loss.png      # [M]  Critic loss (neural algos)
│   ├── comparison_reward_variance.png  # [R2] Rolling std of reward
│   └── comparison_solved_episode.png   # [R3] First episode to reach solved threshold
└── deployment/                     # Deployment evaluation
    ├── deployment_reward.png               # [N]  Bar: avg reward
    ├── deployment_ep_length.png            # [O]  Bar: avg episode length
    ├── deployment_success_rate.png         # [P]  Bar: % episodes reaching max_steps
    ├── deployment_length_hist.png          # [Q]  Histogram of episode lengths
    ├── deployment_reward_per_ep.png        # [R]  Per-episode reward line
    ├── deployment_pole_mean_angle.png      # [R4] Bar: mean |pole angle|
    ├── deployment_pole_std_angle.png       #      Bar: pole angle std
    ├── deployment_pole_angle_traces.png    # [S]  Pole angle per algorithm (subplots)
    ├── deployment_pole_angle_comparison.png#      All algos pole angle overlay
    ├── deployment_cart_position.png        # [T]  Cart position over time
    ├── deployment_phase_portrait.png       # [U]  Phase portrait (θ vs θ̇)
    └── deployment_action_traces.png        # [V]  Applied force over time

*   D/E: only for AC, A2C, PPO, TD3, SAC (algorithms with neural actor/critic)
**  F:   only for Linear_QN, DQN (ε-greedy algorithms)
*** G:   only for AC, A2C, PPO, SAC (algorithms with entropy tracking)
```

### Model Checkpoints

```
w/{task}/
├── Linear_QN/
│   └── Linear_QN_{episode}.json
├── DQN/
│   └── DQN_{episode}.pt
├── MC_REINFORCE/
│   └── MC_REINFORCE_{episode}.pt
├── AC/
│   └── AC_{episode}.pt
├── A2C/
│   └── A2C_{episode}.pt
├── PPO/
│   └── PPO_{episode}.pt
├── TD3/
│   └── TD3_{episode}.pt
└── SAC/
    └── SAC_{episode}.pt
```

---

## Project Structure

| File / Folder                                         | Purpose                                                        |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| `scripts/Function_based/train_all.py`                 | Train all 8 algorithms + generate all comparison/deploy plots  |
| `scripts/Function_based/train.py`                     | Train single algorithm                                         |
| `scripts/Function_based/play.py`                      | Deployment evaluation (deterministic inference)                |
| `RL_Algorithm/RL_base_function.py`                    | Base class: `scale_action`, `decay_epsilon`, `plot_durations`  |
| `RL_Algorithm/storage/buffers.py`                     | `RolloutBuffer` (on-policy) and `ReplayBuffer` (off-policy)   |
| `RL_Algorithm/storage/on_policy.py`                   | `OnPolicyAlgorithm` base for AC, A2C, PPO                     |
| `RL_Algorithm/storage/off_policy.py`                  | `OffPolicyAlgorithm` base for DQN, TD3, SAC                   |
| `RL_Algorithm/networks/mlp.py`                        | Shared `MLP` backbone (relu / elu / tanh activations)          |
| `RL_Algorithm/Function_based/Linear_Q.py`             | Linear function approximation Q-Learning                      |
| `RL_Algorithm/Function_based/DQN.py`                  | Deep Q-Network with experience replay + target network         |
| `RL_Algorithm/Function_based/MC_REINFORCE.py`         | Monte Carlo REINFORCE policy gradient                          |
| `RL_Algorithm/Function_based/AC.py`                   | Actor-Critic (MC episodic)                                     |
| `RL_Algorithm/Function_based/A2C.py`                  | Advantage Actor-Critic (TD + GAE)                              |
| `RL_Algorithm/Function_based/PPO.py`                  | Proximal Policy Optimization (clipped surrogate + GAE)         |
| `RL_Algorithm/Function_based/TD3.py`                  | Twin Delayed DDPG (twin critics + delayed policy updates)      |
| `RL_Algorithm/Function_based/SAC.py`                  | Soft Actor-Critic (max entropy + auto temperature tuning)      |
| `source/CartPole/CartPole/tasks/`                     | IsaacLab task definition (reward, termination, MDP)            |

---

## Summary of Results

### Training Performance

| Algorithm      | Solved (ep) | Final Avg Reward | Training Stability |
| -------------- | ----------- | ---------------- | ------------------ |
| Linear_QN      | Never       | ~40              | Stable but low     |
| DQN            | Never       | ~10              | Diverged early     |
| MC_REINFORCE   | Never       | ~35              | High variance      |
| AC             | 246         | ~500             | Occasional dips    |
| A2C            | 234         | ~500             | Collapse at ep 2k  |
| **PPO**        | **100**     | **~500**         | **Most stable**    |
| TD3            | 3599        | ~500             | Slow convergence   |
| SAC            | 531         | ~500             | Volatile throughout|

*Solved = first episode where 100-episode rolling average ≥ 450 steps.*

### Deployment Performance (100 episodes, deterministic policy)

| Algorithm      | Avg Reward | Avg Length | Success Rate | Mean \|θ\| (°) | Std θ (°) |
| -------------- | ---------- | ---------- | ------------ | -------------- | --------- |
| Linear_QN      | 7.7        | 11.2       | 0%           | 8.85           | 10.17     |
| DQN            | 6.3        | 9.9        | 0%           | 9.31           | 6.23      |
| MC_REINFORCE   | 22.4       | 26.9       | 0%           | 10.93          | 8.48      |
| AC             | 441.4      | 444.5      | 34%          | 1.73           | 2.50      |
| **A2C**        | **499.9**  | **500.0**  | **100%**     | **0.12**       | **0.50**  |
| **PPO**        | **500.0**  | **500.0**  | **100%**     | 0.23           | **0.42**  |
| TD3            | 499.3      | 500.0      | 100%         | 0.98           | 2.02      |
| SAC            | 497.9      | 500.0      | 100%         | 2.95           | 3.69      |

**Overall Winner: PPO** — fastest to solve (ep 100), most stable training, perfect deployment (500.0 avg reward, 100% success).

**Best Pole Control: A2C** — lowest mean absolute pole angle (0.12°) during deployment.
