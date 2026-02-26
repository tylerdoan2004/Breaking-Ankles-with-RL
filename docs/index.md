---
layout: default
title: Home
---

## Breaking Ankles with RL

A reinforcement learning project where an agent must navigate a 2D gridworld, avoid dynamic obstacles and seekers, and reach a goal under limited visibility. We use PPO to train the agent and evaluate its performance as the environment scales in difficulty.

**Source code:** [GitHub Repository](https://github.com/tylerdoan2004/Breaking-Ankles-with-RL)


---

## Environment and Tools

We build our environment using [Gymnasium](https://gymnasium.farama.org/), a standard interface for reinforcement learning environments. Our gridworld is implemented as a custom [MiniGrid](https://minigrid.farama.org/) environment, allowing us to create configurable layouts and scenarios. MiniGrid also provides built-in rendering tools that help us visualize trajectories and debug agent behavior.

---

## Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [MiniGrid Documentation](https://minigrid.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)

---

## Reports

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)