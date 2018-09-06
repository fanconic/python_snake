'''
Autonomous snake which should differentiate between pies and posions
Reinforcement learning with Q-Learn algorithm using mainly Keras as library

This file will give an array of actions, which the snake execute autonomously.

After research the best algorithm to use, seems to be SARSA, based on
http://cs229.stanford.edu/proj2016spr/report/060.pdf

Author: Claudio
'''

# Import dependencies
import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

actions = ['r','r','u','u','l','l','d','d']

class SnakeNN:
    def __init__(self, initial_games = 100, test_games = 100, goal_steps = 100, learning_rate = 1e-2):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.learning_rate = learning_rate
        self.actions = ['u','d','l','r']

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            