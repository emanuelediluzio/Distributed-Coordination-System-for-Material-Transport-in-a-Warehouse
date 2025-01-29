# Distributed-Coordination-System-for-Material-Transport-in-a-Warehouse (Centralized & CTDE)

## Overview
This repository implements a **multi-agent reinforcement learning (MARL) system** for warehouse coordination. The environment models multiple robots that must navigate a grid, pick up tasks, deliver them to drop-off zones, and manage battery levels.

Two learning paradigms are implemented:

1. **Centralized Training & Execution (Centralized PPO)**
   - A single neural network controls all robots.
   - Each step, it receives a **global observation** (concatenation of all robots' states) and outputs **actions for all robots**.
   - Trained with **Stable-Baselines3 PPO**.

2. **Centralized Training with Decentralized Execution (CTDE - MAPPO)**
   - Each robot acts based on its **local observation**.
   - During training, a **centralized critic** (with full state information) helps improve learning.
   - During execution, only the **local actor** is used.
   - Implemented using **Multi-Agent PPO (MAPPO)**.

---

## Project Structure

```
/CTDE
│── /logs                  # Training logs
│── /models                # Saved models (Centralized & MAPPO)
│── /scripts               # Training & testing scripts
│── warehouse_env.py       # Multi-agent warehouse environment (PettingZoo ParallelEnv)
│── wrappers.py            # Wrapper for Gym compatibility
│── centralized_training.py # PPO centralized training script
│── mappo_training.py      # MAPPO CTDE training script
│── mappo_test_script.py   # MAPPO testing script
│── README.md              # Project documentation
```

---

## Environment Details
The warehouse is modeled as a **grid-world** with:
- **Robots**: Move in 4 directions, pick & drop tasks, charge at stations.
- **Tasks**: Must be picked up and delivered to predefined drop zones.
- **Obstacles**: Static obstacles that block movement.
- **Charging Stations**: Locations where robots can recharge.
- **Battery Management**: Robots must balance task completion with recharging.
- **Multi-Agent Communication**: Robots share limited information via a **16-bit message system**.

### Observation Space
- **Centralized PPO**: Concatenation of all robots' states.
- **CTDE (MAPPO)**: Each robot sees its own **local state**.

### Action Space
- **Discrete action space**: Each robot can perform one of the following actions:
  - Move (FORWARD, LEFT, RIGHT)
  - Pick/Drop task
  - Charge
  - Send messages (e.g., battery status, request help, path blocked).

---

## Training Paradigms

### 1. Centralized PPO
#### Description
- Uses a **single PPO policy** to control all robots.
- The policy receives a **global observation** and outputs **actions for all robots**.
- **Advantage**: Simple implementation, good for small numbers of agents.
- **Disadvantage**: Doesn't scale well for many robots (high-dimensional input & output).

#### Training (`centralized_training.py`)
```bash
python centralized_training.py
```
- Uses **Stable-Baselines3 PPO**.
- Trains with **MultiDiscrete action space** (one action per robot).
- Saves model as `ppo_warehouse.zip`.

#### Testing
```bash
python centralized_test_script.py
```
- Loads `ppo_warehouse.zip` and evaluates over 10 episodes.

---

### 2. CTDE - MAPPO (Multi-Agent PPO)
#### Description
- Each robot has its **own policy** (actor).
- During training, a **centralized critic** observes the **global state**.
- During execution, each robot uses **only its local observation**.
- **Advantage**: Scales better, realistic for decentralized systems.
- **Disadvantage**: More complex training.

#### Training (`mappo_training.py`)
```bash
python mappo_training.py
```
- Trains **MAPPO (Multi-Agent PPO)** from scratch.
- Saves actor weights to `mappo_actor.pth`.

#### Testing (`mappo_test_script.py`)
```bash
python mappo_test_script.py
```
- Loads `mappo_actor.pth`.
- Runs multiple test episodes with **decentralized execution**.

---

## Key Differences: Centralized PPO vs CTDE MAPPO
| Feature                 | Centralized PPO | CTDE MAPPO |
|-------------------------|----------------|------------|
| **Observation Space**   | Global (all robots) | Local (per robot) |
| **Critic**              | Centralized | Centralized (during training) |
| **Actor Policy**        | Single shared policy | Separate per robot |
| **Execution**           | Centralized | Decentralized |
| **Training Complexity** | Simpler | More complex |
| **Scalability**         | Limited (high dim.) | Better (per-robot policy) |
| **Best Use Case**       | Small number of robots | Large-scale multi-robot systems |

---

## Dependencies
- **Python 3.8+**
- **Stable-Baselines3** (for centralized PPO)
- **PettingZoo** (multi-agent environment handling)
- **PyTorch** (for MAPPO)
- **Gymnasium** (observation & action spaces)

### Install Requirements
```bash
pip install -r requirements.txt
```

---

## Future Improvements
- **More advanced multi-agent coordination** (learning to share workload efficiently).
- **Better reward shaping** for coordination.
- **Hybrid training**: Combining **centralized and decentralized** learning

Author Emanuele Di Luzio

Your NameContact: emanuelediluzio0@gmail.com
