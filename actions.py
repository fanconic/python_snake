'''
Autonomous snake which should differentiate between pies and posions
Reinforcement learning with Q-Learn algorithm using mainly Keras as library

This file will give an array of actions, which the snake execute autonomously.

After research the best algorithm to use, seems to be SARSA, based on
http://cs229.stanford.edu/proj2016spr/report/060.pdf

Author: Claudio
'''

# Import dependencies
import config
import tensorboard
import tensorflow
import keras
import numpy

actions = ['r','r','u','u','l','l','d','d']

