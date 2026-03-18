"""
This module provides the multi-agent training loop using PettingZoo + SuperSuit + SB3.
All runners share a single PPO policy (parameter sharing).
"""
import importlib
from datetime import datetime, timezone
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

from src.multi_agent_environment import MultiAgentReactiveAvoidanceEnv
from src.utils.yaml_parser.configuration import SystemConfiguration


def main():
    # Import config (default to multi_agent_ppo if it exists, else ppo)
    try:
        config_module = importlib.import_module("configs.experiments.multi_agent_ppo")
    except ModuleNotFoundError:
        config_module = importlib.import_module("configs.experiments.ppo")

    SEED = config_module.SEED
    MODEL = config_module.MODEL
    TRAINING_METADATA = config_module.TRAINING_METADATA
    SYSTEM_CONFIGURATIONS = config_module.SYSTEM_CONFIGURATIONS

    # Parse the system config
    system_config = SystemConfiguration.parse_config_file(Path(SYSTEM_CONFIGURATIONS.training.path))

    num_runners = getattr(config_module, "NUM_RUNNERS", 2)

    print(f"Creating multi-agent environment with {num_runners} runners...")

    # Create the PettingZoo env
    env = MultiAgentReactiveAvoidanceEnv(config=system_config, num_runners=num_runners)

    # Wrap for SB3 compatibility
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=TRAINING_METADATA.n_envs, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    print("Creating PPO agent (shared policy)...")
    model = PPO(
        **MODEL.hyperparameters,
        env=env,
        seed=SEED,
        verbose=1,
        device="auto"
    )

    print(f"Training for {TRAINING_METADATA.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=TRAINING_METADATA.total_timesteps,
        progress_bar=True
    )

    print("Saving model...")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir = Path("logs") / f"multi_agent_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "final_model"))
    print(f"Model saved to {save_dir / 'final_model'}")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
