from src.utils.logging.experiment_metadata import ConfigPaths


LOGGING_DIRECTORY = "logs"
EXPERIMENT_NAME = "reactive_avoidance"
MODEL_NAME = "PPO"
SEED = 0
# NOTE: Default model hyperparameters obtained from https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
MODEL_HYPERPARAMETERS = {
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "rollout_buffer_class": None,
    "rollout_buffer_kwargs": None,
    "target_kl": None,
    "policy_kwargs": None
}
CONFIG_PATHS: ConfigPaths = {
    "training": "configs/training/training.yaml",
    "evaluation/in_distribution": "configs/evaluation/in_distribution.yaml",
    "evaluation/out_of_distribution": "configs/evaluation/out_of_distribution.yaml",
}
