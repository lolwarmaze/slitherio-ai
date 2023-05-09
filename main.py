from webgame import WebGame
from stable_baselines3.common import env_checker
from callback import TrainAndLoggingCallback
import torch.utils.tensorboard
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import tensorflow
import torch

env = WebGame()
env_checker.check_env(env)

CHECKPOINT_DIR = '.train/'
LOG_DIR = './logs/'
callback = TrainAndLoggingCallback(check_freq=2000, save_path=CHECKPOINT_DIR)

model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=100000, learning_starts=0)
model.learn(total_timesteps=5000, callback=callback)
