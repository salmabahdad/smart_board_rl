import gymnasium as gym
from smart_board_env import AirplaneEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.callbacks import  MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold

import os

model_dir = "models"
log_dir = "logs"

def train():

    env = make_vec_env(AirplaneEnv, n_envs=12, env_kwargs={"rows_num":10, "seats_row":5}, vec_env_cls=SubprocVecEnv)
    
    model = MaskablePPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir, ent_coef=0.05)

    eval_callback = MaskableEvalCallback(
        env,
        eval_freq=10000,
        verbose=1,
        best_model_save_path=os.path.join(model_dir, 'MaskablePPO'),
    )

    """
    total_timesteps: pass in a very large number to train (almost) indefinitely.
    callback: pass in reference to a callback fuction above
    """
    model.learn(total_timesteps=int(1e10), callback=eval_callback)

def test(model_name, render=True):

    env = gym.make('smart-board', rows_num=10, seats_row=5, render_mode='human' if render else None)

    # Load model
    model = MaskablePPO.load(f'models/MaskablePPO/{model_name}', env=env)

    rewards = 0
    # Run a test
    obs, _ = env.reset()
    terminated = False

    while True:
        action_masks = get_action_masks(env)
        action, _ = model.predict(observation=obs, deterministic=True, action_masks=action_masks) # Turn on deterministic, so predict always returns the same behavior
        obs, reward, terminated, _, _ = env.step(action)
        rewards += reward

        if terminated:
            break

    print(f"Total rewards: {rewards}")

if __name__ == '__main__':
    train()


