# This model is deprecated.
import gym
from gym import wrappers

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

print("Start to learn something")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

print("Start to evaluate the model")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()