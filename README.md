# CS188 Project 3: Reinforcement Learning

> UC Berkeley CS188 — Introduction to Artificial Intelligence  
> Project 3: Reinforcement Learning — Value Iteration, Q-Learning & Approximate Q-Learning

## Overview

This project implements several reinforcement learning algorithms and applies them to the Gridworld, Crawler, and Pacman environments.

The completed core tasks in this repository include:

- **Value Iteration** — Computing optimal policies for a known MDP via dynamic programming.
- **Bridge Crossing Analysis** — Tuning MDP parameters (discount, noise, living reward) to produce desired policies.
- **Q-Learning** — Learning action values from experience without a model of the environment.
- **Epsilon-Greedy Exploration** — Balancing exploration and exploitation in Q-Learning.
- **Approximate Q-Learning** — Generalizing Q-Learning with linear function approximation and feature extractors.

These algorithms are tested on:

| Environment | Description |
|---|---|
| **Gridworld** | A discrete grid MDP with configurable rewards, noise, and discount |
| **Crawler** | A simulated robot that learns to crawl using joint angle control |
| **Pacman** | The classic game where Pacman learns to navigate and eat food while avoiding ghosts |

---

## Project Structure

```text
CS188_Project3/
├── valueIterationAgents.py      # ⭐ Value Iteration implementation; Prioritized Sweeping skeleton also present
├── qlearningAgents.py           # ⭐ Q-Learning, Pacman Q-Learning & Approximate Q-Learning agents
├── analysis.py                  # ⭐ Answers to analysis questions (Q2, Q3a–e, Q7)
├── featureExtractors.py         # Feature extractors for Approximate Q-Learning
├── environment.py               # Abstract Environment interface
├── mdp.py                       # Abstract MDP interface
├── gridworld.py                 # Gridworld MDP environment
├── crawler.py                   # Crawler robot environment
├── learningAgents.py            # Base classes for value estimation & reinforcement agents
├── pacman.py                    # Pacman game engine
├── game.py                      # Core game logic (Agents, Actions, Grid, etc.)
├── ghostAgents.py               # Ghost agent behaviors
├── pacmanAgents.py              # Basic Pacman agent behaviors
├── keyboardAgents.py            # Keyboard-controlled Pacman agent
├── graphicsDisplay.py           # Pacman graphics rendering
├── graphicsGridworldDisplay.py  # Gridworld graphics rendering
├── graphicsCrawlerDisplay.py    # Crawler graphics rendering
├── graphicsUtils.py             # Graphics utility functions
├── textDisplay.py               # Text-based display for Pacman
├── textGridworldDisplay.py      # Text-based display for Gridworld
├── layout.py                    # Pacman layout parser
├── util.py                      # Data structures & utilities (Counter, PriorityQueue, etc.)
├── autograder.py                # Autograder entry point
├── submission_autograder.py     # Submission autograder
├── grading.py                   # Grading framework
├── testParser.py                # Test case parser
├── testClasses.py               # Generic test class infrastructure
├── reinforcementTestClasses.py  # RL-specific test classes
├── projectParams.py             # Project parameters
├── layouts/                     # Pacman map layout files
├── test_cases/                  # Autograder test cases
├── VERSION                      # Project version info
└── README.md                    # This file
```

> **Files marked with ⭐ are the primary student-code files for this repository.**

---

## Implemented Algorithms

### 1. Value Iteration (`valueIterationAgents.py`)

Implements the **Value Iteration** algorithm for solving known MDPs:

```text
V_{k+1}(s) = max_a Σ_{s'} T(s, a, s') · [R(s, a, s') + γ · V_k(s')]
```

- **`runValueIteration()`** — Runs batch value iteration for a specified number of iterations, updating every state in each sweep.
- **`computeQValueFromValues(state, action)`** — Computes `Q(s, a)` from the current value function using the Bellman equation.
- **`computeActionFromValues(state)`** — Extracts the optimal policy by selecting the action with the highest Q-value.

### 2. Q-Learning (`qlearningAgents.py`)

Implements **model-free** temporal-difference learning:

```text
Q(s, a) ← (1 - α) · Q(s, a) + α · [R + γ · max_{a'} Q(s', a')]
```

Key methods:

- **`getQValue(state, action)`** — Returns the stored Q-value (`0.0` for unseen state-action pairs).
- **`computeValueFromQValues(state)`** — Returns `max_a Q(s, a)` over legal actions.
- **`computeActionFromQValues(state)`** — Returns the best action, breaking ties randomly.
- **`getAction(state)`** — Implements **ε-greedy** exploration: with probability `ε`, selects a random legal action; otherwise follows the current policy.
- **`update(state, action, nextState, reward)`** — Performs the TD update on the Q-value.

### 3. Approximate Q-Learning (`qlearningAgents.py`)

Uses **linear function approximation** with feature vectors:

```text
Q(s, a) = w · f(s, a)
w_i ← w_i + α · [R + γ · max_{a'} Q(s', a') - Q(s, a)] · f_i(s, a)
```

- **`getQValue(state, action)`** — Computes the Q-value as the dot product of weights and features.
- **`update(state, action, nextState, reward)`** — Updates the weight vector based on the TD error and feature values.
- The agent defaults to **`IdentityExtractor`**, and can also be run with **`SimpleExtractor`** via command-line arguments.

### 4. Analysis Questions (`analysis.py`)

This file contains parameter settings for the written analysis parts of the project:

| Question | Description | Parameters |
|---|---|---|
| **Q2** | Cross the bridge (eliminate noise) | `discount=0.9, noise=0.0` |
| **Q3a** | Prefer close exit, risk cliff | `discount=0.2, noise=0.0, livingReward=-1.0` |
| **Q3b** | Prefer close exit, avoid cliff | `discount=0.2, noise=0.2, livingReward=-1.0` |
| **Q3c** | Prefer distant exit, risk cliff | `discount=0.9, noise=0.0, livingReward=-0.1` |
| **Q3d** | Prefer distant exit, avoid cliff | `discount=0.9, noise=0.2, livingReward=-0.1` |
| **Q3e** | Avoid both exits and cliff (stay alive) | `discount=0.9, noise=0.2, livingReward=2.0` |
| **Q7** | Can Q-Learning converge in ≤50 episodes? | `NOT POSSIBLE` |

---

## Getting Started

### Prerequisites

- **Python 3.x**
- **Tkinter** (usually bundled with Python and needed for graphical display)

### Running Gridworld

```bash
# Default Gridworld with manual control
python gridworld.py -m

# Run Value Iteration agent (100 iterations)
python gridworld.py -a value -i 100 -k 10

# Run Q-Learning agent on Gridworld
python gridworld.py -a q -k 100
```

### Running Pacman

```bash
# Pacman with tabular Q-Learning (2000 training games + 10 evaluation games)
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# Pacman with Approximate Q-Learning using the default extractor
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid

# Pacman with Approximate Q-Learning using SimpleExtractor
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

### Running the Crawler

```bash
# Watch the crawler robot learn to crawl
python crawler.py
```

### Running the Autograder

```bash
# Run all test cases
python autograder.py

# Run a specific question (example: q6)
python autograder.py -q q6
```

---

## Common Command-Line Parameters

### Gridworld (`gridworld.py`)

| Parameter | Flag | Default | Description |
|---|---|---|---|
| Discount (`γ`) | `-d` | `0.9` | Future reward discount factor |
| Living Reward | `-r` | `0.0` | Reward for each non-terminal step |
| Noise | `-n` | `0.2` | Probability of moving in an unintended direction |
| Epsilon (`ε`) | `-e` | `0.3` | Exploration probability in Q-learning |
| Learning Rate (`α`) | `-l` | `0.5` | TD learning rate |
| Iterations | `-i` | `10` | Number of value iteration sweeps |
| Episodes | `-k` | `1` | Number of learning episodes |

### Pacman (`pacman.py`)

| Parameter | Flag | Default | Description |
|---|---|---|---|
| Number of games | `-n` | `1` | Total number of games to play |
| Layout | `-l` | `mediumClassic` | Map layout to load |
| Pacman agent | `-p` | `KeyboardAgent` | Agent class to use |
| Agent arguments | `-a` | — | Comma-separated agent parameters |
| Training episodes | `-x` | `0` | Number of training games |

---

## Architecture

```text
                    ┌──────────────────┐
                    │  ValueEstimation │
                    │      Agent       │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │ ValueIterationAgent│         │ ReinforcementAgent│
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
    ┌─────────▼────────────────┐   ┌────────▼─────────┐
    │ PrioritizedSweepingValue │   │  QLearningAgent  │
    │ IterationAgent (skeleton)│   └────────┬─────────┘
    └──────────────────────────┘            │
                                   ┌────────▼─────────┐
                                   │   PacmanQAgent   │
                                   └────────┬─────────┘
                                            │
                                   ┌────────▼──────────┐
                                   │ ApproximateQAgent │
                                   └───────────────────┘
```

---

## Notes on Current Repository Status

- `ValueIterationAgent`, `QLearningAgent`, `PacmanQAgent`, and `ApproximateQAgent` are implemented.
- `PrioritizedSweepingValueIterationAgent` exists in the file structure, but its `runValueIteration()` method is still left as a skeleton.
- `ApproximateQAgent` uses `IdentityExtractor` by default; `SimpleExtractor` is optional and must be specified explicitly via `-a extractor=SimpleExtractor`.

---

## Acknowledgments

The Pacman AI projects were developed at **UC Berkeley**.

The core projects and autograders were primarily created by:

- **John DeNero** (`denero@cs.berkeley.edu`)
- **Dan Klein** (`klein@cs.berkeley.edu`)

Student-side autograding was added by **Brad Miller**, **Nick Hay**, and **Pieter Abbeel** (`pabbeel@cs.berkeley.edu`).

For more information, visit **http://ai.berkeley.edu**.
