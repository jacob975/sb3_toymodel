import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder

# Play BipedalWalker
env_name = 'LunarLander-v2'
env = gym.make(env_name, render_mode='rgb_array')
# Load weights
model = PPO.load("LunarLander_policy", env, device=torch.device("cpu"))
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

video_folder = "logs/videos/"
video_length = 1000

obs = vec_env.reset()
# Record the video starting at the first step
vec_env = VecVideoRecorder(
    vec_env, video_folder, 
    record_video_trigger=lambda x: x == 0, 
    video_length=video_length, 
    name_prefix=f"random-agent-{env_name}")

vec_env.reset()
for _ in range(video_length + 1):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
# Save the video
vec_env.close()