import gym
import numpy as np
import joblib
import time

def test(q_table,city):
    state = city.reset()
    done = False
    rewards = 0

    for s in range(200):

        print("Step {}".format(s+1))

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = city.step(action)
        rewards += reward
        city.render()
        time.sleep(0.5)
        print(f"Score: {rewards}")
        print(f"Number of state: {new_state}")
        print(f"Action: {action}")
        state = new_state

        if done == True:
            print("Result: TAXI SUCCESSFULLY ARRIVED TO THE DESTINATION WITH THE PASSENGER\n")
            break
    if done == False:
        print("Result: TAXI FAILED TO DELIVER PASSENGER TO THE DESTINATION")
def main_test():
     run,ep=1,1
     agent = gym.make("Taxi-v3").env
     q_table=joblib.load("Qtable.joblib")
     run=int(input("Do you want to run one episode? [no-0/yes-1]\n"))
     while(run):
         print("****** EPISODE ",ep,"******")
         test(q_table,agent)
         ep+=1
         run=int(input("Do you want to run another episode? [no-0/yes-1]\n"))
