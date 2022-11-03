import gym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Play BipedalWalker
env_name = 'BipedalWalker-v3'
env = gym.make(env_name)

# TODO: Load the saved model and evaluate that model.
model = PPO.load("bipedalwalker_policy", print_system_info=True)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()