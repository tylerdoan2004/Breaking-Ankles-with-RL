from environment import RLEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Creates the environment and validates it
train_env = RLEnvironment(render_mode=None)
print("Checking environment...")
check_env(train_env)
print("Environment check passed!")

# Creates the model and trains it using the environment
model = PPO("MultiInputPolicy", train_env, verbose=1)
model.learn(total_timesteps=1000000)
print("Training complete!")

# Saves the trained model
model.save("ppo_reactive_avoidance")
print("Model saved as 'ppo_reactive_avoidance'")

train_env.close()