import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold #For call backs

#Create OpenAI Gym Environment
environment_name = "CartPole-v0"
env = gym.make(environment_name)
#Wrapper for a nonvectorized environment
env = DummyVecEnv([lambda: env])

#Save logs to this path
log_path = os.path.join('Training','logs')

#Call backs (Stop when we reach a reward)
save_path = os.path.join('Training','saved_models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
#Every 100000 steps check if passed the reward threshold, if so save it
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path =save_path,
                             verbose=1)

#PPO algorithm, using the MlpPolicy
model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

#Now just train the model for 20000 steps, also pass in the call back
model.learn(total_timesteps=20000,callback=eval_callback)

#Save the model
PPO_Path = os.path.join('Training','saved_models','PPO_Model_Cartpole')
model.save(PPO_Path)
