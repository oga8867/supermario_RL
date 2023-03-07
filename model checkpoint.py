import gym_super_mario_bros
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
# Define the Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# Create a DQN agent and train it on the environment for 10,000 steps
model = DQN.load("dqn_mario_checkpoint4" ,env=env,exploration_final_eps=0.2) #exploration_fraction=0.2, learning_rate=0.001)
#모델을 로드한것은 이렇게 파라메터를 변경해줘야함



model.learning_rate = 0.001
#model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save the model checkpoint
model.save("dqn_mario_checkpoint5")

# # Continue training the model for another 10,000 steps
# model.learn(total_timesteps=10000)
#
# # Load the saved model checkpoint
# model = DQN.load("dqn_mario_checkpoint")
#
# # Continue training the model from the checkpoint for another 10,000 steps
# model.learn(total_timesteps=10000)