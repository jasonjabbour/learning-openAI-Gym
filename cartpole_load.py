import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "CartPole-v0"
#Recreated gym environment
env = gym.make(environment_name)
#Wrapper for a nonvectorized environment
env = DummyVecEnv([lambda: env])

#Get path of saved model
PPO_Path = os.path.join('Training','saved_models','PPO_Model_Cartpole')

#Reload the saved model
model = PPO.load(PPO_Path,env=env)

#Now we can continue to train
#model.learn(total_timesteps=1000)

#Evaluation ------
#we want to test for 10 episodes and we want to view it in real life
#evaluate_policy(model,env,n_eval_episodes=10,render=True)

#Test Model ---------------------
#Test out environment 5 times
episodes = 5
#Loop through each episode
for episodes in range(1,episodes+1):
    #Observations of our observation space
    obs = env.reset()
    done = False
    score = 0

    while not done:
        #View the graphical representation
        env.render()
        #Now using the model to predict the step to take
        action, states = model.predict(obs)
        #Pass our random action to the environment, get back net set of observations, reward, whether our episode is done
        obs, reward, done, info = env.step(action)
        #Count the reward
        score+=reward
    print('Episode:{} Score:{}'.format(episodes,score))

# print(env.action_space)
# print(env.observation_space)

#close environment
env.close()