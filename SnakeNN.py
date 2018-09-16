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

# dependencies
import SnakeGame
import config
import random
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

class SnakeNN:
    # Constructor
    def __init__(self, initial_games = 100, test_games = 100, goal_steps = 100, learning_rate = 1e-2, filename = 'snake_nn.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.learning_rate = learning_rate
        self.vector_actions = [
                            [[-1,0], 0],
                            [[0,1], 1],
                            [[1,0], 2],
                            [[0,-1], 3],
                            ]               
        self.filename = filename

    # Do 100 Initial games with random steps
    def initial_population(self):
        training_data = []

        for _ in range(self.initial_games):
            game = SnakeGame.SnakeGame()
            _, _, snake, _, _ = game.start()
            previous_observation = self.generate_observation(snake)

            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, _, snake, _, _ = game.step(game_action)

                if done:
                    training_data.append([self.add_action_to_observation(previous_observation, action), 0])
                    break

                else:
                    training_data.append([self.add_action_to_observation(previous_observation, action), 1])
                    previous_observation = self.generate_observation(snake)

        print(len(training_data))
        return training_data

    # Generate an action for the snake
    def generate_action(self, snake):
        action = random.randint(0,2) - 1
        return action, self.get_game_action(snake, action)

    # Get the action for a game
    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction

        if action == -1:
            new_direction = self.turn_left(snake_direction)
        
        elif action == 1:
            new_direction = self.turn_right(snake_direction)

        for pair in self.vector_actions:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
    
        return game_action

    # Generating an observation
    def generate_observation(self, snake):
        snake_direction = self.get_snake_direction_vector(snake)
        barrier_left = self.direction_blocked(snake, self.turn_left(snake_direction))
        barrier_right = self.direction_blocked(snake, self.turn_right(snake_direction))
        barrier_front = self.direction_blocked(snake, snake_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right)])

    # add an action to an observation
    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    # Get the direction of the action of the snake
    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    # Boolean if the direction of the snake is blocked
    def direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == config.screen_heigth+1 or point[1] == config.screen_width+1

    # Turn the snake to the right of its direction
    def turn_right(self, vector):
        return np.array([vector[1], -vector[0]])    

    # Turn the snake to the left of its direction
    def turn_left(self, vector):
        return np.array([-vector[1], vector[0]])

    # Model the neural network with tensorflowb
    def model(self):
        network = input_data(shape=[None, 4, 1], name= 'input')
        network = fully_connected(network, 1, activation= 'linear')
        network = regression(network, optimizer= 'adam', learning_rate= self.learning_rate, loss= 'mean_square', name= 'target')
        model = tflearn.DNN(network, tensorboard_dir= 'log')
        return model

    # Train the neural network model
    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1,4,1)
        y = np.array([i[1] for i in training_data]).reshape(-1,1)
        model.fit(X, y, n_epoch = 1, shuffle= True, run_id= self.filename)
        model.save(self.filename)
        return model

    # Visualize the game
    def visuluaize_game(self, model):
        game = SnakeGame.SnakeGame(gui= True)
        _, _, snake, _, _ = game.start()
        previous_observation = self.generate_observation(snake)

        for _ in range(self.goal_steps):
            predictions = []

            for _ in range(-1,2):
                predictions.append(model.predict(self.add_action_to_observation(previous_observation, action).reshape(-1,4,1)))

            action = np.argmax(np.array(predictions))
            game_action = self.get_game_action(snake, action -1)
            done, _, snake, _, _ = game.step(game_action)

            if done:
                break
            else:
                previous_observation = self.generate_observation(snake)
    
    # Test the neural network model
    def test_model(self, model):
        steps_arr = []

        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame.SnakeGame()
            _, _, snake, _, _ = game.start()
            previous_observation = self.generate_observation(snake)

            for _ in range(self.goal_steps):
                predictions = []  
                for action in range(-1,2):
                    predictions.append(model.predict(self.add_action_to_observation(previous_observation,action).reshape(-1,4,1)))
                    action = np.argmax(np.array(predictions))
                    game_action = self.get_game_action(snake, action -1)
                    done, _, snake, _, _ = game.step(game_action)
                    game_memory.append([previous_observation, action])

                    if done:
                        break

                    else:
                        previous_observation = self.generate_observation(snake)
                        steps += 1
                steps_arr.append(steps)

            print('Average steps:',mean(steps_arr))
            print(Counter(steps_arr))
                    
    # Train model
    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    # Visualize the game
    def visualize(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visuluaize_game(nn_model)

    # Test the model
    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

# Start the process from terminal
if __name__ == "__main__":
    SnakeNN().train()