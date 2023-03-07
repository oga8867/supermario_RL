import gym_super_mario_bros
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Define the Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# Create a DQN agent and train it on the environment
model = DQN('MlpPolicy', env, buffer_size=100000,exploration_fraction=0.2, verbose=1, learning_rate=0.001)
model.learn(total_timesteps=10000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

model.save("dqn_mario")