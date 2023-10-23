import gymnasium as gym
import torch
from stable_baselines3 import PPO

# Play BipedalWalker
env_name = 'BipedalWalker-v3'
env = gym.make(env_name, render_mode='rgb_array')
# Load weights
model = PPO.load("bipedalwalker_policy", env, device=torch.device("cpu"))
model.set_env(env)

print("Start to evaluate the model")
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render('human')
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

