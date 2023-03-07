import gym_super_mario_bros
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


import gym
from gym import spaces

# create the Super Mario environment
env = gym.make('SuperMarioBros-1-1-v0')

# # create the same action space object as the model
# action_space = spaces.Discrete(256)
#
# # set the action space of the environment to be the same as the model's action space
# env.action_space = action_space

# # Define the Super Mario environment
# env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
#
# # Wrap the environment in a vectorized environment
# env = DummyVecEnv([lambda: env])

# Load the trained model
model = DQN.load("dqn_mario_checkpoint5.zip", env=env)

done = True
for step in range(50000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()