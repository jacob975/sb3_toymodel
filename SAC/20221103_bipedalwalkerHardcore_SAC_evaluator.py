import gym
import time

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Play BipedalWalker
env_name = 'BipedalWalkerHardcore-v3'
env = gym.make(env_name)

# TODO: Load the saved model and evaluate that model.
model = SAC.load("./BWHv2/best_model", print_system_info=True)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()