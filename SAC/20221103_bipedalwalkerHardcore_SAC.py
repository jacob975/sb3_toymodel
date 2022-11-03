import gym
import time

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

# TODO: How do SB3 know the action space is a discrete or continouous?
# Play BipedalWalker
env_name = 'BipedalWalkerHardcore-v3'
env = gym.make(env_name)
env = Monitor(env, "./BWHv2", allow_early_resets=True)
check_env(env, warn=True)
model = SAC("MlpPolicy", env, verbose=1)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="./BWHv2",
    log_path="./BWHv2",
    eval_freq=10000,
)

callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5,
    tb_log_name="bipedalwalkerhardcore_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("./BWHv2/policy")
