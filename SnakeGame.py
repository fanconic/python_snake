'''
Snake game, originally known from old Nokia phones
The snake is controllable with the arrow keys and is implement as a two dimensional array
Additionally there are is now aswell poisonous food (marked as a diamond), where you lose pies, if you eat one.
Author: Claudio Fanconi
based on: https://www.youtube.com/watch?v=rbasThWVb-c and 
https://github.com/korolvs/snake_nn/blob/master/snake_game.py
'''
import random
import curses
import config

class SnakeGame():
    # initializing game parameters
    def __init__(self, screen_width = config.screen_width, screen_heigth = config.screen_width, gui = False):
        self.pies = 0
        self.screen_width = screen_width
        self.screen_heigth = screen_heigth
        self.done = False
        self.gui = gui

    # initiliazing the screen
    def render_init(self):
        curses.initscr()
        window = curses.newwin(self.screen_heigth, self.screen_width, 0, 0)
        curses.curs_set(0)
        window.nodelay(1)
        window.timeout(config.timeout)
        self.window = window
        self.render()

    # Destroy the rendering
    def render_destroy(self):
        curses.endwin()

    # Rendering the game
    def render(self):
        self.window.clear()
        self.window.boarders(0)
        self.window.addstr(0, 2, 'Pies: ' + str(self.pies))
        self.window.addch(self.food[0], self.food[1], curses.ACS_CKBOARD)
        self.window.addch(self.poison[0], self.poison[1], curses.ACS_DIAMOND)
        self.window.getch()

    # Starting a game of snake
    def start(self):
        self.snake_init()
        self.generate_food()
        self.generate_poison()
        if self.gui:
            self.render_init()
        return self.generate_observations()

    # Ending the game of snake
    def end_game(self):
        if self.gui:
            self.render_destroy()
        raise Exception('Game Over!')

    # initializing the snake in the game
    def snake_init(self):
        self.snake_x = self.screen_width/4
        self.snake_y = self.screen_heigth/2
        self.snake = [
            [self.snake_y, self.snake_x],
            [self.snake_y, self.snake_x-1],
            [self.snake_y, self.snake_x-2]
        ]

    # Generating Food
    def generate_food(self):
        food = []
        while food == []:
            food = [random.randint(1, self.screen_heigth-1), random.randint(1,self.screen_width-1)]
            if food in self.snake:
                food = []
        self.food = food

    # Generating Poison
    def generate_poison(self):
        poison = []
        while poison == []:
            poison = [random.randint(1, self.screen_heigth-1), random.randint(1,self.screen_width-1)]
            if poison in self.snake:
                poison = []
        self.poison = poison

    # Next step to take for the snake
    def step(self, key):
        # Check if game is over
        if self.done:
            self.end_game()

        # eat food
        if self.food_eaten():
            self.pies += 1
            self.generate_food()

        # eat poison
        if self.poison_eaten():
            self.pies -= 1
            self.generate_poison()

    # Depending on were the snake moves the new is formed
    def create_new_head(self, action): 
        new_head = [self.snake[0][0], self.snake[0][1]]

        # Go Downwards
        if action == 'd':
            new_head[0] += 1

        # Go Upwards
        if action == 'u':
            new_head[0] -= 1

        # Go Left
        if action == 'l':
            new_head[1] -= 1

        # Go Right
        if action == 'r':
            new_head[1] += 1
        
        self.snake.insert(0, new_head)

    # Removing the last part of the snake
    def remove_tail(self):
        self.snake.pop()

    # Eat the food
    def food_eaten(self):
        return self.snake[0] == self.food

    # Eat the poison
    def poison_eaten(self):
        return self.snake[0] == self.poison

    # Check for collision or if the ammount of pies is negative
    def collision(self):
        if( self.snake[0][0] in [0,self.screen_heigth-1] or
            self.snake[0][1] in [0,self.screen_width-1] or
            self.snake[0] in self.snake[1:] or
            self.pies < 0 ):
            self.done = True
    
    # Returning observations for the reinforcement learning
    def generate_observations(self):
        return self.done, self.pies, self.snake, self.food, self.poison

if __name__ == "__main__":
    game = SnakeGame(gui = True)
    game.start
    actions = ['u', 'd', 'r', 'l']
    for _ in range (20):
        game.step(actions[random.randint(0,3)])
