import time
from pathlib import Path
from stable_baselines3 import PPO
from src.environment import ReactiveAvoidanceEnv
from src.utils.yaml_parser.configuration import SystemConfiguration 


# Loads in the trained model and creates a test environment
print("Loading trained model...")

test_yaml_file = Path("configs/system/evaluation/in_distribution.yaml")
test_env = ReactiveAvoidanceEnv(config = SystemConfiguration.parse_config_file(test_yaml_file) ,render_mode="human")
model = PPO.load("logs/reactive_avoidance/ppo/ppo/best_agent_against_greedy_seekers/models/best_model.zip", env=test_env)

# Defining the number of times to test in this environment
print("Starting testing with visualization...")
num_episodes = 5

for episode in range(num_episodes):
    print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

    # Initialize variables, reward and initial conditions
    total_reward = 0
    obs, info = test_env.reset()
    
    # The maximum number of steps per episode, can change based on grid size
    for step in range(100):

        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)
        action = int(action.item())
        
        # Take step and get environment states
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        
        # Render and slow down for visualization
        test_env.render()
        time.sleep(0.2)
        
        # Print step info
        print(f"Step {step}: action={action}, reward={reward:.1f}, total={total_reward:.1f}")
        
        if terminated or truncated:
            print(f"Episode {episode + 1} ended at step {step}!")
            print(f"Total reward: {total_reward:.1f}")
            time.sleep(2)
            break

print("\n=== Testing Complete ===")
test_env.close()
