'''
Autonomous snake which should differentiate between pies and posions
Reinforcement learning with Q-Learn algorithm using mainly Keras as library

This file will give an array of actions, which the snake execute autonomously.

After research the best algorithm to use, seems to be SARSA, based on
http://cs229.stanford.edu/proj2016spr/report/060.pdf
Most of the Code can be found from:
https://github.com/korolvs/snake_nn/blob/master/nn_1.py

Author: Claudio
'''

from SnakeGame import SnakeGame
import config

from random import randint
import numpy as np
import math

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, flatten, dropout
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter



# Class of Neural Network for Snake
class SnakeNN:
    def __init__(self, initial_games = 10000, test_games = 1000, goal_steps = 2000, lr = 1e-2, filename = 'snake_nn_2.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[-1, 0], 0],
                [[0, 1], 1],
                [[1, 0], 2],
                [[0, -1], 3]
                ]

    # Make  an initial set of games
    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = SnakeGame()
            _, prev_score, snake, food, poison = game.start()
            prev_observation = self.generate_observation(snake, food, poison)
            prev_food_distance = self.get_food_distance(snake, food)
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, score, snake, food, poison = game.step(game_action)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake, food, poison)
                    prev_food_distance = food_distance
        return training_data

    # generate a random action
    def generate_action(self, snake):
        action = randint(0,2) - 1
        return action, self.get_game_action(snake, action)

    # Get action for current game situation
    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    # Generate an observation of the current state
    def generate_observation(self, snake, food, poison):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        poison_direction = self.get_poison_direction_vector(snake, poison)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    # Add an action to an observation
    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    # Get the direction where the snake is going
    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    # Get the direction where the food is
    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    # Get the direction of the poison
    def get_poison_direction_vector(self, snake, poison):
        return np.array(poison) - np.array(snake[0])

    # Normalize vector / Linear Algebra
    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    # Get distance to next food
    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    # If the direction is blocked by a wall or snake
    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == config.screen_width+1 or point[1] == config.screen_heigth+1

    # Turn snake to the left
    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    # Turn the snake to the right
    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    # Neural network model, DFF Model with 3 hidden layer, each with 25, 50, 25 neurons
    def DNNmodel(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 50, activation='relu')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    # 1D-Convolutional Neural Network model
    def CNNmodel(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = conv_1d(network, 1, (3,3), activation='relu')
        network = max_pool_1d(network, kernel_size=(2,2))
        network = conv_1d(network, 1, (3,3), activation='relu')
        network = max_pool_1d(network, kernel_size=(2,2))
        network = flatten(network)
        network = fully_connected(network, 128, activation= 'tanh')
        network = fully_connected(network, 64, activation='tanh')
        network = fully_connected(network, 1, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    # Recurrent neural network model
    def RNNmodel(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = lstm(network, 128, dropout= 0.8)
        network = fully_connected(network, 1, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    # Train the neural Network
    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    # Test the neural network
    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for game_number in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame()
            _, score, snake, food, poison = game.start()
            prev_observation = self.generate_observation(snake, food, poison)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food, poison  = game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print('Test Game', game_number)
                    print('Pies', score)
                    print('Steps', steps)
                    break
                else:
                    prev_observation = self.generate_observation(snake, food, poison)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    # Visualize the game
    def visualise_game(self, model):
        game = SnakeGame(gui = True)
        _, _, snake, food, poison = game.start()
        prev_observation = self.generate_observation(snake, food, poison)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
               precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food, poison  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food, poison)

    # Training activation
    def train(self):
        training_data = self.initial_population()
        nn_model = self.DNNmodel()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    # Visualization activation
    def visualise(self):
        nn_model = self.DNNmodel()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    # Testing activation
    def test(self):
        nn_model = self.DNNmodel()
        nn_model.load(self.filename)
        self.test_model(nn_model)

# Main instructions
if __name__ == "__main__":
    SnakeNN().train()
    #SnakeNN().visualise()
    #SnakeNN().test()