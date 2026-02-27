# CompSci 175 Reinforcement Learning: Breaking Ankles with RL

## Project Summary
[cite_start]Breaking Ankles with RL is a reinforcement-learning project studying reactive avoidance in a two-dimensional gridworld[cite: 2]. [cite_start]In our problem setup, an agent with partial observability navigates from some starting location to some goal location while avoiding static obstacles and dynamic "seekers"[cite: 3]. [cite_start]Both the agent and the seekers traverse the gridworld at some constant velocity, and each seeker attempts to move in a manner that minimizes its Chebyshev distance to the agent[cite: 4]. [cite_start]A single episode terminates when the agent reaches the objective, when a seeker intercepts the agent, when the agent collides with an obstacle or seeker, or when a time limit is reached[cite: 5]. [cite_start]The training objective is to learn a policy that balances goal-oriented navigation with real-time evasive maneuvers[cite: 6]. [cite_start]The overarching goal of our project is to develop a reinforcement learning framework that facilitates long-term autonomous navigation amidst short-term safety constraints[cite: 7]. 

[cite_start]The environment is implemented in Python using Gymnasium and built on MiniGrid as a simulation environment, operating in discrete time steps[cite: 18]. [cite_start]The agent receives an observation of its position, the relative position of the goal, and the relative positions of nearby obstacles and seekers stacked over the last five frames to provide temporal context[cite: 18]. [cite_start]The agents have 9 actions, to move to any one of the adjacent spaces or to not move at all, and using Proximal Policy Optimization through Stable Baselines 3 over many steps, we learn how to balance efficient goal direction movement and seeker and obstacle avoidance[cite: 19, 20]. [cite_start]This is all to simulate robotic exploration and autonomous navigation within dynamic environments[cite: 20].

## Project Approach
[cite_start]Our training-evaluation pipeline accepts as input four YAML files, each containing several environment hyperparameters that determine a particular environment configuration[cite: 9]. [cite_start]In particular, these hyperparameters include the gridworld dimensions, the static obstacles' locations, the episode time limit, the agent's start and goal locations, the agent's velocity, the agent's visibility radius, the seekers' start locations, and the seekers' velocity[cite: 10]. [cite_start]Given these four YAML files, our training-evaluation pipeline produces four environments: one corresponding to a training environment, one corresponding to a validation environment, and two corresponding to evaluation environments (one in-distribution environment and one out-of-distribution environment)[cite: 11]. 

[cite_start]We implement each environment in Python using Gymnasium, a popular open-source library for implementing reinforcement learning environments[cite: 12]. [cite_start]We use MiniGrid, an extension of the Gymnasium ecosystem with configurable grid environments, to assist in rendering our environments[cite: 13]. [cite_start]Each environment operates in discrete time steps, and at each discrete time step, the agent receives an observation of the environment state and a scalar reward[cite: 14]. [cite_start]At each time step, the environment state consists of the hyperparameters listed above, the agent's current position, and the seekers' current position[cite: 15]. [cite_start]The agent's observation consists of a flattened one-dimensional vector of features, representing the agent's position[cite: 16].

## Algorithm
[cite_start]The agent is trained using Proximal Policy Optimization (PPO) [cite: 21] [cite_start][Schulman et al., 2017], which is an on-policy actor-critic reinforcement learning algorithm[cite: 22]. [cite_start]We use the implementation from Stable Baselines 3 with a MultiInputPolicy network, which automatically builds a neural network suited to the flat observation vector[cite: 23]. 

[cite_start]PPO works by collecting a fresh batch of environment experience, running several gradient update passes over that batch, then discarding it and collecting new data[cite: 24]. [cite_start]This keeps training stable because the policy being optimized never drifts too far from the policy that collected the data[cite: 25]. [cite_start]Stability is further enforced by a clipping mechanism: the policy update is constrained so that the new policy's action probabilities can't change by more than a factor of clip_range, which is 0.2 relative to the old policy[cite: 26]. 

[cite_start]The full loss PPO minimizes combines three terms[cite: 27]:
* [cite_start]Clipped policy loss maximizes expected advantage while preventing large policy shifts[cite: 28].
* [cite_start]Value function loss - trains a critic to predict future returns (mean-squared error)[cite: 29].
* [cite_start]Entropy bonus - encourages exploration by penalizing overconfident action distributions[cite: 30].

[cite_start]Advantages (how much better an action was than expected) are estimated using Generalized Advantage Estimation (GAE) with discount factor gamma = 0.99 and smoothing parameter gae_lambda = 0.95[cite: 31].

## Environment
[cite_start]The environment is a 10x10 MiniGrid gridworld in which the agent (the "runner") must navigate from a fixed start position (0, 0) to a goal at (9, 9) while avoiding 7 diagonal static wall obstacles and 2 pursuing seekers[cite: 32]. [cite_start]Each episode lasts at most 10 steps[cite: 33].

**Observation (State)**: 
[cite_start]Each observation is a flat vector with three parts[cite: 34]:
* [cite_start]1. The agent's current (x, y) position, scaled to the range [-1, 1][cite: 34].
* [cite_start]2. The (dx, dy) vector pointing from the agent to the goal, also scaled to [-1, 1][cite: 35].
* [cite_start]3. A local observation stack: the last 5 frames of what the agent can see[cite: 36]. [cite_start]Each frame covers a 5x5 tile window centered on the agent (visibility radius = 5) with 3 binary channels: one for walls/obstacles, one for seekers, and one for the goal[cite: 37]. [cite_start]This gives the agent memory of recent seeker movements so it can infer their direction of travel[cite: 38].

[cite_start]The total observation vector has length: 2 + 2 + (5 frames x 3 channels x 5 x 5) = 379 values[cite: 39].

**Actions**: 
[cite_start]The agent has 9 discrete actions: stay in place, or move in any of the 8 cardinal/diagonal directions (up, down, left, right, and the four diagonals)[cite: 40].

**Rewards**:

| Event | Reward |
| :--- | :--- |
| Reach the goal | [cite_start]+100 [cite: 44] |
| Hit a wall or get caught by a seeker | [cite_start]-100 [cite: 44] |
| Survive one step without a terminal event | [cite_start]-1 [cite: 44] |

[cite_start]The small per-step penalty pushes the agent to reach the goal quickly without idling[cite: 45]. [cite_start]All rewards are in the [-1, 1] range[cite: 46]. 

**Seekers**: 
[cite_start]Each seeker follows a greedy pursuit policy every step it moves one cell in the direction that minimizes its distance to the agent (Chebyshev distance)[cite: 46]. [cite_start]Two seekers start at (0, 9) and (9, 0)[cite: 47].

## Hyperparameters
[cite_start]All PPO hyperparameters are left at their Stable Baselines 3 defaults (source: SB3 PPO docs)[cite: 49]. [cite_start]No tuning was performed[cite: 49].

| Hyperparameters | Value | Source |
| :--- | :--- | :--- |
| Total training steps | 1,000,000 | [cite_start]Set by us [cite: 50] |
| Rollout buffer size (n_steps) | 2,048 | [cite_start]SB3 default [cite: 50] |
| Mini-batch size | 64 | [cite_start]SB3 default [cite: 50] |
| Update epochs per rollout | 10 | [cite_start]SB3 default [cite: 50] |
| Discount factor (gamma) | 0.99 | [cite_start]SB3 default [cite: 50] |
| GAE lambda | 0.95 | [cite_start]SB3 default [cite: 50] |
| range (epsilon) | 0.2 | [cite_start]SB3 default [cite: 50] |
| Learning rate | 0.0003 | [cite_start]SB3 default [cite: 50] |
| Entropy coefficient | 0.0 | [cite_start]SB3 default [cite: 50] |
| Value function coefficient | 0.5 | [cite_start]SB3 default [cite: 50] |
| Grid size | [cite_start]10x10 | config/default.yaml [cite: 50] |
| Visibility radius | [cite_start]5 | config/default.yaml [cite: 50] |
| Observation stack depth | [cite_start]5 frames | config/default.yaml [cite: 50] |
| Episode time limit | [cite_start]80 steps | config/default.yaml [cite: 50] |
| Number of seekers | [cite_start]1-2 | config/default.yaml [cite: 51, 52, 53] |

## Evaluation
[cite_start]After looking more into our project, we decided that our primary metrics will include success rate (percentage of episodes where the agent reaches the objective), capture rate (percentage of episodes where a seeker intercepts the agent), and time-to-goal (average time steps required for the agent to reach the objective)[cite: 64]. 

[cite_start]The reason for each of the metrics we've gathered is simple: for episode rewards, we want to monitor the amount of reward we give for each episode, seeing if the reward is consistent and not over or under-rewarding (PPO)[cite: 65]. [cite_start]Next, we have episode length[cite: 66]. [cite_start]This is important to see how well our agent is at getting to a terminated state, whether it is running into the seeker or reaching the goal[cite: 66]. [cite_start]The length here shows us that there are next to no scenarios that end up with an infinite loop where both the seeker and agent go back and forth[cite: 67]. [cite_start]Next, we are able to use the goal reach rate and capture rate to see how well the agent actually performs as it learns a better policy[cite: 68]. 

[cite_start]You can see that there's a plateau in the success rate after 25000 or so episodes, which could indicate a couple of things, mainly that the environment is too simple[cite: 69]. [cite_start]As the agent learns that going straight for the goal is beneficial, it only matters that it reaches the goal, making the environment almost a solved problem if the agent can make it or not, making "learning" no longer useful[cite: 70].

## Remaining Goals and Challenges
[cite_start]The current implementation is relatively basic, the agent can navigate a bounded space, avoid basic static and dynamic obstacles, and reach a goal point[cite: 72]. [cite_start]Most of our effort has been focused on tuning the training: timesteps, reward shaping, and core behavioral constraints[cite: 73, 74]. [cite_start]As a result, testing across varied environments has been limited[cite: 75]. 

[cite_start]Going forward, we want to increase the complexity of what the agent has to handle, by introducing more restricted observability so the agent must make decisions under uncertainty, reward structures that encourage balancing risk against reward rather than simply reaching the goal, and consistent self-preservation that holds up across different environment configurations[cite: 76]. 

[cite_start]We will aim to build out a batch testing pipeline with procedural environment generation to stress test the agent at scale and identify niche edge cases that custom tests would miss[cite: 77]. [cite_start]Identifying these failure modes is just as important as improving the model, since they will provide us insight on how we can shape behaviors to our desire[cite: 78].

[cite_start]The most significant challenge we anticipate is making sure the agent achieves reliable and perfect self-preservation[cite: 79]. [cite_start]In the real-world applications our model targets, a single mistake can be fatal, so there is very little margin for error[cite: 80]. [cite_start]The difficulty here lies in reward balancing, if we penalize risk too much, the agent becomes overly conservative and sacrifices efficiency entirely, which is just as problematic as being reckless, both are behaviors we want to avoid[cite: 81]. [cite_start]The goal is to keep the agent on the fine line between risk and reward, where it is making smart decisions rather than defaulting to one extreme[cite: 82]. [cite_start]Getting that balance right through reward shaping alone is genuinely difficult and will likely require significant iteration[cite: 83].

[cite_start]The other major challenge is keeping our training efficient[cite: 84]. [cite_start]As we add complexity, our agents' observation and state space will be bloated with unnecessary information that slows training without meaningfully improving our policy[cite: 84]. [cite_start]We want to make sure everything is purposeful and with reason, trimming anything that slows training without contributing to the behaviors we actually care about[cite: 85]. [cite_start]Keeping training efficient while still providing the agent with everything it needs to make good decisions is a balance we will need to actively manage as the project develops[cite: 86].

## AI Use
[cite_start]We used Al tools to explore possible RL algorithms and to perform grammar checking on this proposal document[cite: 88]. [cite_start]We also used Al when trying to get the environment to train[cite: 89].

## References
* Schulman et al. (2017). [cite_start]Proximal Policy Optimization Algorithms. arXiv:1707.06347. [cite: 55]
* Schulman et al. (2016). [cite_start]High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438. [cite: 55, 56]
* Raffin et al. (2021). Stable-Baselines3. [cite_start]JMLR. https://stable-baselines3.readthedocs.io [cite: 57]
