import gymnasium as gym
from stable_baselines3 import PPO
# Use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = gym.make("CartPole-v1", render_mode='rgb_array')

print("Start to learn something")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

print("Start to evaluate the model")
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render('human')
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()