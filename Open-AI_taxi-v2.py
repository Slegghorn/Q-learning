import numpy as np
import gym
import random

#create the gym environement
env = gym.make('Taxi-v2')

#create the Q-table
#determine the size of the tables with the following gym function
n_action = env.action_space.n
n_state = env.observation_space.n

#initalize the Q-table
qtable = np.zeros((n_state, n_action))

#parameters
total_episodes = 5000               #number of times we run the AI for training
total_test_episodes = 100           # " for testing
max_steps = 99                      #maximum action per episode

learning_rate = 0.7
gamma = 0.618                       #discount rate

#exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

#training
for episode in range(total_episodes):
    #reset the environement
    state = env.reset()
    step = 0
    done = False
    for step in range(max_steps):
        #choose a random value to choose exploration or exploitation
        exp_exp_tradeoff = random.uniform(0, 1)
        #choose based on epsilon value exploration or exploitation
        if exp_exp_tradeoff > epsilon:
            #the most effective action
            action = np.argmax(qtable[state, :])
        else:
            #a random action
            action = env.action_space.sample()
        #evaluate the action
        new_state, reward, done, info = env.step(action)
        #change the Q-table
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state
        if done == True:
            break
    #change epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(decay_rate * (episode + 1))

#testing
env.reset()
rewards = []
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print('--------------------')
    print('Episode :', episode)
    for step in range(max_steps):
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        if done == True:
            rewards.append(total_rewards)
            print('Score :', total_rewards)
            break
        state = new_state
env.close()
print('average score:', sum(rewards) / total_test_episodes)
