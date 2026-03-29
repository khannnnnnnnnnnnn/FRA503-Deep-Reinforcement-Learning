# Homework 2: Cart Pole Model-Free Algorithms

**Authors**  
- Chantouch Orungrote (66340500011)  
- Sasish Kaewsing (66340500076)

## Objectives
Implement and evaluate four model-free control algorithms — Monte Carlo, SARSA, Q-Learning, and Double Q-Learning — on the discretized CartPole environment. Analyze how the resolution of the action space and the granularity of state discretization affect learning performance, convergence stability, and Q-value structure.

---

### Part 1: Setup and Run

1. Open terminal in the project folder
2. Install dependencies:
   ```bash
   pip install numpy matplotlib gymnasium
   ```
3. Run all experiments:
   ```bash
   python scripts/RL_Algorithm/train_all.py --task <TASK_NAME> --num_envs <N>
   ```

**Task Selection**

Use the `--task` flag to select which CartPole task to train on:

| Flag | Task |
|------|------|
| `--task Stabilize-Isaac-Cartpole-v0` | Cart-Pole stabilization (keep pole upright) |
| `--task SwingUp-Isaac-Cartpole-v0` | Cart-Pole swing-up (swing pole from bottom to top) |

Example:
```bash
# Train on Stabilize task
python scripts/RL_Algorithm/train_all.py --task Stabilize-Isaac-Cartpole-v0 --headless

# Train on SwingUp task
python scripts/RL_Algorithm/train_all.py --task SwingUp-Isaac-Cartpole-v0 --headless
```

**Parallel Environment Selection**

Use the `--num_envs` flag to run multiple environments in parallel, which significantly speeds up training. Higher values use more GPU memory but train faster.

```bash
# Run with 8 parallel environments (recommended for most GPUs)
python scripts/RL_Algorithm/train_all.py --task Stabilize-Isaac-Cartpole-v0 --headless --num_envs 8

# Run with 64 parallel environments (for high-end GPUs with more VRAM)
python scripts/RL_Algorithm/train_all.py --task Stabilize-Isaac-Cartpole-v0 --headless --num_envs 64
```

> **Note:** Results are saved in separate folders per task (`plots/Stabilize/` and `plots/SwingUp/`), so running both tasks will not overwrite each other's results.

**Alternative: Run Individual Scripts**
- Train a single algorithm:
  ```bash
  python scripts/RL_Algorithm/train.py
  ```
- Play/deploy a trained agent:
  ```bash
  python scripts/RL_Algorithm/play.py
  ```
- Random action baseline:
  ```bash
  python scripts/RL_Algorithm/random_action.py
  ```

---

### Part 2: Parameter Definition

| Parameter               | Value              | Description |
|-------------------------|--------------------|-------------|
| Episodes                | 10,000             | Total training episodes |
| Discount Factor (γ)     | 0.99               | Future reward discounting |
| Learning Rate (α)       | 0.1                | Step size for value updates |
| Initial ε               | 1.0                | Starting exploration rate |
| Final ε                 | 0.01               | Minimum exploration rate |
| ε-decay                 | 0.001              | Linear decay per episode |
| Action Range            | [−10.0, 10.0]      | Force applied to the cart |

#### **Note: All algorithms use ε-greedy with linear decay as the exploration strategy**

---

### Part 3: Configuration

Experiments are divided into two variations to evaluate sensitivity to action and observation space resolution.

**Variation 1: Action Space Resolution**

| Configuration    | Number of Actions | State Weights  |
|------------------|-------------------|----------------|
| Low Resolution   | 5                 | [2, 10, 1, 2]  |
| Normal Resolution| 11                | [2, 10, 1, 2]  |
| High Resolution  | 21                | [2, 10, 1, 2]  |

**Variation 2: Observation Space Discretization**

| Configuration     | Number of Actions | State Weights  |
|-------------------|-------------------|----------------|
| Low Granularity   | 11                | [1, 5, 1, 1]   |
| Normal Granularity| 11                | [2, 10, 1, 2]  |
| High Granularity  | 11                | [4, 20, 2, 4]  |

The state discretization weights vector `W = [wx, wx_dot, wθ, wθ_dot]` scales the granularity for cart position, cart velocity, pole angle, and pole angular velocity respectively.

---

### Part 4: Results Location

All outputs are saved in the `plots/` and `q_value/` folders, separated by task and algorithm:

```
plots/
├── Stabilize/
│   ├── MC/
│   │   ├── action_low/
│   │   │   ├── learning_curve.png
│   │   │   ├── episode_length_curve.png
│   │   │   └── q_surface.png
│   │   ├── action_normal/
│   │   ├── action_high/
│   │   ├── dsw_low/
│   │   └── dsw_high/
│   ├── Q_Learning/             # Same structure as MC
│   ├── SARSA/                  # Same structure as MC
│   ├── Double_Q_Learning/      # Same structure as MC
│   ├── comparisons/
│   │   ├── action_low/
│   │   │   ├── comparison_reward.png
│   │   │   └── comparison_ep_length.png
│   │   ├── action_normal/
│   │   ├── action_high/
│   │   ├── dsw_low/
│   │   └── dsw_high/
│   ├── deployment/
│   │   ├── deployment_reward.png
│   │   ├── deployment_ep_length.png
│   │   └── deployment_success_rate.png
│   └── resolution_effect/
│       ├── MC_action_resolution.png
│       ├── MC_dsw_resolution.png
│       ├── Q_Learning_action_resolution.png
│       ├── Q_Learning_dsw_resolution.png
│       ├── SARSA_action_resolution.png
│       ├── SARSA_dsw_resolution.png
│       ├── Double_Q_Learning_action_resolution.png
│       └── Double_Q_Learning_dsw_resolution.png
│
└── SwingUp/
    └── (same structure as Stabilize/)

q_value/
├── Stabilize/
│   ├── MC/
│   ├── Q_Learning/
│   ├── SARSA/
│   └── Double_Q_Learning/
└── SwingUp/
    ├── MC/
    ├── Q_Learning/
    ├── SARSA/
    └── Double_Q_Learning/
```

**Metrics per configuration:**
- `learning_curve.png` — Cumulative reward smoothed over 100-episode window
- `episode_length_curve.png` — Episode length (pole stability duration)
- `q_surface.png` — 3D Q-value surface: Cart Position × Pole Angle → max Q

---

### Part 5: Structure

```
.
├── RL_Algorithm/
│   ├── Algorithm/
│   │   ├── MC.py                      # Monte Carlo Control
│   │   ├── Q_Learning.py              # Q-Learning
│   │   ├── SARSA.py                   # SARSA
│   │   └── Double_Q_Learning.py       # Double Q-Learning
│   ├── Function_based/
│   │   ├── DQN.py                     # Deep Q-Network
│   │   ├── Linear_Q.py                # Linear Q approximation
│   │   ├── AC.py                      # Actor-Critic
│   │   └── MC_REINFORCE.py            # REINFORCE policy gradient
│   ├── RL_base.py                     # Base class for tabular RL agents
│   └── RL_base_function.py            # Base class for function approximation agents
├── scripts/
│   ├── RL_Algorithm/
│   │   ├── train.py                   # Train a single agent
│   │   ├── train_all.py               # Run all experiments
│   │   ├── play.py                    # Deploy trained agent
│   │   └── random_action.py           # Random action baseline
│   └── Function_based/
│       ├── train.py
│       ├── play.py
│       └── random_action.py
├── source/
│   └── CartPole/                      # CartPole environment source
├── plots/                             # Generated learning curves and Q-surface plots
├── q_value/                           # Saved Q-tables (JSON)
└── docker/                            # Docker environment setup
    ├── Dockerfile
    └── docker-compose.yaml
```
