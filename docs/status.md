# CompSci 175 Reinforcement Learning: Breaking Ankles with RL

## Project Summary
Breaking Ankles with RL is a reinforcement-learning project studying reactive avoidance in a two-dimensional gridworld. In our problem setup, an agent with partial observability navigates from some starting location to some goal location while avoiding static obstacles and dynamic "seekers". Both the agent and the seekers traverse the gridworld at some constant velocity, and each seeker attempts to move in a manner that minimizes its Chebyshev distance to the agent. A single episode terminates when the agent reaches the objective, when a seeker intercepts the agent, when the agent collides with an obstacle or seeker, or when a time limit is reached. The training objective is to learn a policy that balances goal-oriented navigation with real-time evasive maneuvers. The overarching goal of our project is to develop a reinforcement learning framework that facilitates long-term autonomous navigation amidst short-term safety constraints. 

The environment is implemented in Python using Gymnasium and built on MiniGrid as a simulation environment, operating in discrete time steps. The agent receives an observation of its position, the relative position of the goal, and the relative positions of nearby obstacles and seekers stacked over the last five frames to provide temporal context. The agents have 9 actions, to move to any one of the adjacent spaces or to not move at all, and using Proximal Policy Optimization through Stable Baselines 3 over many steps, we learn how to balance efficient goal direction movement and seeker and obstacle avoidance. This is all to simulate robotic exploration and autonomous navigation within dynamic environments.

## Project Approach
Our training-evaluation pipeline accepts as input four YAML files, each containing several environment hyperparameters that determine a particular environment configuration. In particular, these hyperparameters include the gridworld dimensions, the static obstacles' locations, the episode time limit, the agent's start and goal locations, the agent's velocity, the agent's visibility radius, the seekers' start locations, and the seekers' velocity. Given these four YAML files, our training-evaluation pipeline produces four environments: one corresponding to a training environment, one corresponding to a validation environment, and two corresponding to evaluation environments (one in-distribution environment and one out-of-distribution environment). 

We implement each environment in Python using Gymnasium, a popular open-source library for implementing reinforcement learning environments. We use MiniGrid, an extension of the Gymnasium ecosystem with configurable grid environments, to assist in rendering our environments. Each environment operates in discrete time steps, and at each discrete time step, the agent receives an observation of the environment state and a scalar reward. At each time step, the environment state consists of the hyperparameters listed above, the agent's current position, and the seekers' current position. The agent's observation consists of a flattened one-dimensional vector of features, representing the agent's position.

## Algorithm
The agent is trained using Proximal Policy Optimization (PPO) [Schulman et al., 2017], which is an on-policy actor-critic reinforcement learning algorithm. We use the implementation from Stable Baselines 3 with a MultiInputPolicy network, which automatically builds a neural network suited to the flat observation vector. 

PPO works by collecting a fresh batch of environment experience, running several gradient update passes over that batch, then discarding it and collecting new data. This keeps training stable because the policy being optimized never drifts too far from the policy that collected the data. Stability is further enforced by a clipping mechanism: the policy update is constrained so that the new policy's action probabilities can't change by more than a factor of clip_range, which is 0.2 relative to the old policy. 

The full loss PPO minimizes combines three terms:
* Clipped policy loss maximizes expected advantage while preventing large policy shifts.
* Value function loss - trains a critic to predict future returns (mean-squared error).
* Entropy bonus - encourages exploration by penalizing overconfident action distributions.

Advantages (how much better an action was than expected) are estimated using Generalized Advantage Estimation (GAE) with discount factor gamma = 0.99 and smoothing parameter gae_lambda = 0.95.

## Environment
The environment is a 10x10 MiniGrid gridworld in which the agent (the "runner") must navigate from a fixed start position (0, 0) to a goal at (9, 9) while avoiding 7 diagonal static wall obstacles and 2 pursuing seekers. Each episode lasts at most 10 steps.

**Observation (State)**:  
Each observation is a flat vector with three parts:
* 1. The agent's current (x, y) position, scaled to the range [-1, 1].
* 2. The (dx, dy) vector pointing from the agent to the goal, also scaled to [-1, 1].
* 3. A local observation stack: the last 5 frames of what the agent can see. Each frame covers a 5x5 tile window centered on the agent (visibility radius = 5) with 3 binary channels: one for walls/obstacles, one for seekers, and one for the goal. This gives the agent memory of recent seeker movements so it can infer their direction of travel.

The total observation vector has length: 2 + 2 + (5 frames x 3 channels x 5 x 5) = 379 values.

**Actions**:  
The agent has 9 discrete actions: stay in place, or move in any of the 8 cardinal/diagonal directions (up, down, left, right, and the four diagonals).

**Rewards**:

| Event | Reward |
| :--- | :--- |
| Reach the goal | +100 |
| Hit a wall or get caught by a seeker | -100 |
| Survive one step without a terminal event | -1 |

The small per-step penalty pushes the agent to reach the goal quickly without idling. All rewards are in the [-1, 1] range. 

**Seekers**:  
Each seeker follows a greedy pursuit policy every step it moves one cell in the direction that minimizes its distance to the agent (Chebyshev distance). Two seekers start at (0, 9) and (9, 0).

## Hyperparameters
All PPO hyperparameters are left at their Stable Baselines 3 defaults (source: SB3 PPO docs). No tuning was performed.

| Hyperparameters | Value | Source |
| :--- | :--- | :--- |
| Total training steps | 1,000,000 | Set by us |
| Rollout buffer size (n_steps) | 2,048 | SB3 default |
| Mini-batch size | 64 | SB3 default |
| Update epochs per rollout | 10 | SB3 default |
| Discount factor (gamma) | 0.99 | SB3 default |
| GAE lambda | 0.95 | SB3 default |
| range (epsilon) | 0.2 | SB3 default |
| Learning rate | 0.0003 | SB3 default |
| Entropy coefficient | 0.0 | SB3 default |
| Value function coefficient | 0.5 | SB3 default |
| Grid size | 10x10 | config/default.yaml |
| Visibility radius | 5 | config/default.yaml |
| Observation stack depth | 5 frames | config/default.yaml |
| Episode time limit | 80 steps | config/default.yaml |
| Number of seekers | 1-2 | config/default.yaml |

## Evaluation
After looking more into our project, we decided that our primary metrics will include success rate (percentage of episodes where the agent reaches the objective), capture rate (percentage of episodes where a seeker intercepts the agent), and time-to-goal (average time steps required for the agent to reach the objective).

The reason for each of the metrics we've gathered is simple: for episode rewards, we want to monitor the amount of reward we give for each episode, seeing if the reward is consistent and not over or under-rewarding (PPO). Next, we have episode length. This is important to see how well our agent is at getting to a terminated state, whether it is running into the seeker or reaching the goal. The length here shows us that there are next to no scenarios that end up with an infinite loop where both the seeker and agent go back and forth. Next, we are able to use the goal reach rate and capture rate to see how well the agent actually performs as it learns a better policy.

## Remaining Goals and Challenges
The current implementation is relatively basic, the agent can navigate a bounded space, avoid basic static and dynamic obstacles, and reach a goal point. Most of our effort has been focused on tuning the training: timesteps, reward shaping, and core behavioral constraints. As a result, testing across varied environments has been limited. 

Going forward, we want to increase the complexity of what the agent has to handle, by introducing more restricted observability so the agent must make decisions under uncertainty, reward structures that encourage balancing risk against reward rather than simply reaching the goal, and consistent self-preservation that holds up across different environment configurations. 

We will aim to build out a batch testing pipeline with procedural environment generation to stress test the agent at scale and identify niche edge cases that custom tests would miss. Identifying these failure modes is just as important as improving the model, since they will provide us insight on how we can shape behaviors to our desire.

The most significant challenge we anticipate is making sure the agent achieves reliable and perfect self-preservation. In the real-world applications our model targets, a single mistake can be fatal, so there is very little margin for error. The difficulty here lies in reward balancing, if we penalize risk too much, the agent becomes overly conservative and sacrifices efficiency entirely, which is just as problematic as being reckless, both are behaviors we want to avoid. The goal is to keep the agent on the fine line between risk and reward, where it is making smart decisions rather than defaulting to one extreme. Getting that balance right through reward shaping alone is genuinely difficult and will likely require significant iteration.

The other major challenge is keeping our training efficient. As we add complexity, our agents' observation and state space will be bloated with unnecessary information that slows training without meaningfully improving our policy. We want to make sure everything is purposeful and with reason, trimming anything that slows training without contributing to the behaviors we actually care about. Keeping training efficient while still providing the agent with everything it needs to make good decisions is a balance we will need to actively manage as the project develops.

## AI Use
We used AI tools to explore possible RL algorithms and to perform grammar checking on this proposal document. We also used AI when trying to get the environment to train.

## References
* Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.  
* Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438.  
* Raffin et al. (2021). Stable-Baselines3. JMLR. https://stable-baselines3.readthedocs.io  
