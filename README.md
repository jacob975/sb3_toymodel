# sb3_toymodel
I play toy RL models in [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)

# Installation
1. Construct a python environment with `python=3.10`
```bash
conda create -n rl-exp python=3.10 -y
conda activate rl-exp
```
2. Install `torch` following the instructions in [Pytorch](https://pytorch.org/)
3. Install the required packages below
```bash
python -m pip install -r requirements.txt
```

# Usage
1. Train the PPO model on the environment, Bipedalwalker.
```bash
python ./PPO/20231023_bipedalwalker.py
```
2. Evaluate as well as record a video of that trained model.
```bash
python ./PPO/20221103_bipedalwalker_PPO_evaluator.py
```