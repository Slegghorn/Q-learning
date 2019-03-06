import tensorflow as tf
import numpy as np
import retro
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque
import random
import warnings

warnings.filterwarnings('ignore') #avoid skimage warnings

#make the environement with gym
env = retro.make(game = 'SpaceInvaders-Atari2600')

#format the possibles actions to one-hot list
possible_actions = np.identity(env.action_space.n, dtype = 'int')

#preprocessing the frames
'''
    convert to grayscale, crop the image (useless infos), normalize it and resize it
'''
def preprocessing_frames(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame

#frame stacking
#initialization with zeros
stacked_frames = deque([np.zeros((110, 84), dtype = 'int') for i in range(stack_size)], maxlen = 4)

def stack_frames(stacked_frames, state, is_new_episode):
    #preprocess the frames
    frame = preprocessing_frames(frame)

    if is_new_episode:
        #clear the stacked_frames
        stacked_frames = deque([np.zeros((110, 84), dtype = 'int') for i in range(stack_size)], maxlen = 4)

        #append the same frame 4 times (new episode)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        #stack the frames
        stacked_state = np.stack(stacked_frames, axis = 2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis = 2)
    return stacked_state, stacked_frames

#parameters
state_size = [110, 84, 4]
action_size = env.action_space.n
learning_rate = 0.00025

total_episodes = 50
max_steps = 50000
batch_size = 64

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001

gamma = 0.9

pretrain_length = batch_size
memory_size = 1000000

stack_size = 4

#show the training agents / not
training = False
#render the env / not
episode_render = False
