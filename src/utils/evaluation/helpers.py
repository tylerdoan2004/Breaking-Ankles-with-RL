"""
This module contains helper functions for evaluation.
"""
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from src.utils.environment.environment_creation import make_env_from_yaml
from src.utils.typing.agent import StableBaselines3Model


def evaluate_agent(*, model: StableBaselines3Model,
                   yaml_file_path: str, environment_name: str, seed: int,
                   n_episodes: int, metrics_directory: Path, info_keywords: tuple[str, ...]) -> None:
    """
    Evaluates the agent on the environment for a given number of episodes.
    
    :param model: The Stable Baselines 3 model used by the agent.
    :param yaml_file_path: The path to the YAML file defining the environment.
    :param environment_name: The name of the environment.
    :param seed: The seed to use for the environment.
    :param n_episodes: The number of episodes to evaluate the agent on.
    :param metrics_directory: The directory used to store the evaluation metrics.
    :param info_keywords: The keywords to use for the evaluation metrics.
    :return: None.
    """
    environment = DummyVecEnv([lambda: make_env_from_yaml(yaml_file_path)])
    environment.seed(seed)
    environment = VecMonitor(
        environment,
        filename = str(metrics_directory / "monitor.csv"),
        info_keywords = info_keywords,
    )
    mean_reward, std_reward = evaluate_policy(
        model, 
        environment,
        n_eval_episodes = n_episodes,
        deterministic = True,
        render = False,
    )
    environment.close()
    print(f"[Evaluation] {environment_name}")
    print(f"Number of episodes: {n_episodes}")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Standard deviation: {std_reward:.3f}")
