import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import time


# Use CPU 
device = torch.device("cpu")

if __name__ == "__main__":
    start_time = time.time()
    env = gym.make("BipedalWalker-v3", render_mode='rgb_array')
    # Vectorized environments allow to easily multiprocess training
    # Note that SubprocVecEnv has to be put in a `if __name__ == "__main__":`
    #num_cpu = 4
    #env = SubprocVecEnv([lambda: env for i in range(4)])

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

    print("Start to learn something")
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    model.learn(
        total_timesteps=5e5,
        tb_log_name="bipedalwalker_" + str(time.time()),
    )

    # Save policy weights
    model.save("bipedalwalker_policy")
    print("Time elapsed: ", time.time() - start_time)