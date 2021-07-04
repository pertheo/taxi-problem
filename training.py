import gym
import numpy as np
import random
import joblib

def train(q_table,city, alpha,gamma,epsilon,decay,episodes,episode_len):
    for episode in range(episodes):
        state=city.reset()
        done=False
        rewards=0
        for step in range(episode_len):
            r=random.uniform(0,1)
            if r<epsilon:
                action=city.action_space.sample() #exploration
            else:
                action=np.argmax(q_table[state])#exploitation
            next_state,reward,done,info=city.step(action) #do action
            rewards+=reward
            next_max = np.max(q_table[next_state])
        
            q_table[state,action]=q_table[state,action]+alpha*(reward+gamma*next_max)
            q_table[state,action]=(1-alpha)*q_table[state,action]+alpha*(reward+gamma*next_max)
            state=next_state
        
            if done==True:
                break
        print("Reward of episode ",episode+1,": ",rewards)
        epsilon=np.exp(-decay*episode)   
        
def evaluate(q_table, city):
    av_reward,av_penalties,av_timesteps,av_move_reward=0,0,0,0
    for ep in range(100):
        state = city.reset()
        done = False
        rewards,penalties,timesteps = 0,0,0

        for s in range(200):
            action = np.argmax(q_table[state,:])
            new_state, reward, done, info = city.step(action)
            if reward==-10:
                penalties+=1
            rewards += reward
            state = new_state
            timesteps+=1
            if done == True:
                break
        av_reward+=rewards
        av_penalties+=penalties
        av_timesteps+=timesteps
    av_move_reward=av_reward/av_timesteps
    print("Average reward: ", av_reward/100)
    print("Average reward per move: ",av_move_reward)
    print("Average penalties: ",av_penalties/100)
    print("Average timesteps: ",av_timesteps/100)
    print("\n\n")

def main_train():
    city = gym.make("Taxi-v3").env
    q_table = np.zeros([city.observation_space.n, city.action_space.n])
    episode_len=200 #max moves for one episode    
    #HYPERPARAMETERS
    
    #alpha - learning rate
    alpha=float(input("Enter learning rate: \n"))
    while alpha>1.0 or alpha<=0.0:
        alpha=float(input("Incorrect value. Enter learning rate again: \n"))
        
    #gamma - discount rate
    gamma=float(input("Enter discount rate: \n"))
    while gamma>1.0 or gamma<=0.0:
        gamma=float(input("Incorrect value. Enter discount rate again: \n")) 
        
    #epsilon
    epsilon=float(input("Enter epsilon: \n"))
    while epsilon>1.0 or epsilon<=0.0:
        epsilon=float(input("Incorrect value. Enter epsilon again: \n"))
        
    #decay rate
    decay_rate=float(input("Enter decay rate: \n"))
    while decay_rate>1.0 or decay_rate<=0.0:
        decay_rate=float(input("Incorrect value. Enter decay rate again: \n"))
        
    #number of episodes
    episodes=int(input("Enter number of episodes: \n"))
    while episodes<=0.0:
        episodes=int(input("Incorrect value. Enter number of episodes again: \n"))

    train(q_table, city,alpha,gamma,epsilon,decay_rate,episodes,episode_len) 
    evaluate(q_table,city)
    file="Qtable.joblib"
    joblib.dump(q_table,file)
