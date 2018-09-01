'''
Autonomous snake which should differentiate between pies and posions
Reinforcement learning with Q-Learn algorithm using mainly Keras as library

After research the best algorithm to use, seems to be SARSA, based on
http://cs229.stanford.edu/proj2016spr/report/060.pdf

Author: Claudio Fanconi
'''

# Import dependencies
import snake_game
import tensorboard
import tensorflow
import keras
import numpy

