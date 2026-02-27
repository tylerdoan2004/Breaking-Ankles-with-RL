---
layout: default
title: Home
---

## Project Description

A reinforcement learning project where an agent must navigate a two-dimensional gridworld, avoid static obstacles and dynamic seekers, and reach a goal under limited visibility. We use Proximal Policy Optimization (PPO) to train the agent. We evaluate the agent's performance on several metrics as the environment scales in difficulty: the agent's success rate, the agent's collision rate, the seekers' interception rate, the agent's time-to-goal, and cumulative episode return.

![Project Image](images/Project%20Image.jpg)

**Source Code:** [GitHub Repository](https://github.com/tylerdoan2004/Breaking-Ankles-with-RL)

---

## Environment and Tools

We build our environment using [Gymnasium](https://gymnasium.farama.org/), a standard interface for reinforcement learning environments. Our gridworld is implemented as a custom [MiniGrid](https://minigrid.farama.org/) environment, allowing us to create configurable layouts and scenarios. MiniGrid also provides built-in rendering tools that help us visualize trajectories and debug agent behavior.

---

## Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [MiniGrid Documentation](https://minigrid.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)

---

## Reports

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)