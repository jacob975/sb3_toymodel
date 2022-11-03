import gym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# TODO: How do SB3 know the action space is a discrete or continouous?
# Play BipedalWalker
env_name = 'BipedalWalker-v3'
env = gym.make(env_name)
model = PPO("MlpPolicy", env, verbose=1)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5,
    tb_log_name="bipedalwalker_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("bipedalwalker_policy")
