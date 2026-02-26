from environment import RLEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt

class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []  # reached goal
        self.episode_captures = []  # caught by seeker
        self._current_reward = 0
        self._current_length = 0

    def _on_step(self):
        self._current_reward += self.locals["rewards"][0]
        self._current_length += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            
            # Determine outcome from reward signal
            last_reward = self.locals["rewards"][0]
            self.episode_successes.append(1 if last_reward >= 100 else 0)
            self.episode_captures.append(1 if last_reward <= -100 else 0)

            self._current_reward = 0
            self._current_length = 0
        return True

    def plot_metrics(self):
        rewards = self.episode_rewards
        lengths = self.episode_lengths
        episodes = np.arange(len(rewards))

        # Smoothing helper
        def smooth(data, window=50):
            return np.convolve(data, np.ones(window)/window, mode='valid')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Metrics", fontsize=16)

        # 1. Reward over episodes
        axes[0,0].plot(episodes, rewards, alpha=0.3, color='steelblue', label='Raw')
        if len(rewards) >= 50:
            axes[0,0].plot(np.arange(49, len(rewards)), smooth(rewards), color='steelblue', label='Smoothed (50)')
        axes[0,0].set_title("Episode Reward")
        axes[0,0].set_xlabel("Episode")
        axes[0,0].set_ylabel("Total Reward")
        axes[0,0].legend()

        # 2. Episode length over episodes
        axes[0,1].plot(episodes, lengths, alpha=0.3, color='darkorange', label='Raw')
        if len(lengths) >= 50:
            axes[0,1].plot(np.arange(49, len(lengths)), smooth(lengths), color='darkorange', label='Smoothed (50)')
        axes[0,1].set_title("Episode Length")
        axes[0,1].set_xlabel("Episode")
        axes[0,1].set_ylabel("Steps")
        axes[0,1].legend()

        # 3. Win rate (rolling %)
        window = 100
        if len(self.episode_successes) >= window:
            win_rate = smooth(self.episode_successes, window) * 100
            axes[1,0].plot(np.arange(window-1, len(self.episode_successes)), win_rate, color='green')
        axes[1,0].set_title(f"Goal Reach Rate (rolling {window} eps)")
        axes[1,0].set_xlabel("Episode")
        axes[1,0].set_ylabel("% Episodes")
        axes[1,0].set_ylim(0, 100)

        # 4. Capture rate (rolling %)
        if len(self.episode_captures) >= window:
            cap_rate = smooth(self.episode_captures, window) * 100
            axes[1,1].plot(np.arange(window-1, len(self.episode_captures)), cap_rate, color='red')
        axes[1,1].set_title(f"Capture Rate (rolling {window} eps)")
        axes[1,1].set_xlabel("Episode")
        axes[1,1].set_ylabel("% Episodes")
        axes[1,1].set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig("training_metrics.png", dpi=150)
        plt.show()
        print("Saved training_metrics.png")


# Train
train_env = RLEnvironment(render_mode=None)
check_env(train_env)

callback = MetricsCallback()
model = PPO("MultiInputPolicy", train_env, verbose=1)
model.learn(total_timesteps=100_000, callback=callback)

callback.plot_metrics()
model.save("ppo_reactive_avoidance")
train_env.close()