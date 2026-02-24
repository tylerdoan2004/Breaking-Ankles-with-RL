---
layout: default
title: Home
---

This project is a reinforcement learning project where an agent must reach a goal in a 2D gridworld while avoiding obstacles and seekers. The agent has limited visibility and must learn to balance moving toward the goal with avoiding immediate danger. We use PPO to train the agent and evaluate how well it performs as the environment becomes more difficult.

Source code: [Repository](https://github.com/tylerdoan2004/Breaking-Ankles-with-RL)

Environment and Tools

We build our environment using Gymnasium, a standard interface for reinforcement learning environments. Our gridworld is implemented as a custom MiniGrid environment, allowing us to create configurable layouts and scenarios. MiniGrid also provides built-in rendering tools that help us visualize trajectories and debug agent behavior to shape our desired behaviors. 

Reports:

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)