# Import Gym Dependencies
import gym
from gym import Env #Super class
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
#Import Helpers
import numpy as np
import random
import os
#Import Stable Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#Build an Environment
#   Build an agent to give us the best shower possible
#   Randomly temperature
#   37 and 39 degrees

class ShowerEnv(Env):
    def __init__(self):
        #Three actions (Tap up, down, unchanged)
        self.action_space = Discrete(3)
        #Observation space is a value between 0 and 100
        self.observation_space = Box(low=0, high=100, shape=(1,))
        #Set the initial state
        self.state = 38 + random.randint(-3,3)
        #How long we will shower for (60 seconds)
        self.shower_length = 60

    def step(self, action):
        #Apply temp adj
        self.state += action-1
        #our actions could be 0 1 or 2
        #but we really want to either subtract by a temp, stay the same, or add one temp

        #Decrease shower time
        self.shower_length-=1

        #Calculate Reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <= 0:
            done = True
        else:
            done = False

        #If we wanted to return more information
        info = {}

        return self.state, reward, done, info

    def render(self):
        #implement visualization
        pass
    def reset(self):
        self.state = np.array([38+random.randint(-3,3)]).astype(float)
        self.shower_length = 60
        return self.state

if __name__ == '__main__':

    env = ShowerEnv()

    see_spaces = False
    #Types of spaces ---------------
    if see_spaces:
        print(Discrete(3).sample()) #three actions
        print(Box(0,1, shape=(3,3)).sample()) #continuous values
        print(Tuple((Discrete(3),Box(0,1, shape=(3,3)))).sample()) #combine discrete and continuous
        print(Dict({'height':Discrete(2),"speed":Box(0,100,shape=(1,))}).sample())
        print(MultiBinary(4).sample())
        print(MultiDiscrete([5,2,2]).sample())

        print(env.observation_space.sample())
        print(env.action_space.sample())

    #View Using Randomly choose action --------------
    random_action = False
    if random_action:
        episodes = 5
        for episodes in range(1,episodes+1):
            obs = env.reset()
            done = False
            score = 0

            while not done:
                env.render()
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                score+=reward

            print('Episode: {} Score: {}'.format(episodes,score))

    #Train Model ----------------------------
    log_path = os.path.join('ShowerTraining','Logs')
    model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
    model.learn(total_timesteps=40000)

    #Save the model
    PPO_Path = os.path.join('ShowerTraining','saved_models','PPO_Model_Shower')
    model.save(PPO_Path)

    env.close()
