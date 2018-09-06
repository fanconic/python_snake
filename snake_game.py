'''
Snake game, originally known from old Nokia phones
The snake is controllable with the arrow keys and is implement as a two dimensional array
Additionally there are is now aswell poisonous food (marked as a diamond), where you lose pies, if you eat one.
Author: Claudio Fanconi
based on: https://www.youtube.com/watch?v=rbasThWVb-c   
'''
import random
import curses
import config
import actions

# Initialize Screen and cursor
# pies is the ammount of pies eaten
# speedset is a boolean used to increase speed after eating pies
# Timeout indicates the milliseconds after the screen is refreshed
pies = 0
speedset = False
timeout = 200
screen = curses.initscr()
curses.curs_set(0)

# initializing a playing field from the config file
screen_heigth = config.screen_heigth
screen_width = config.screen_width

# Create new window (Starting at the top left of the screen)
# It shall accept key inputs
window = curses.newwin(screen_heigth, screen_width, 0, 0)
window.keypad(1)
window.timeout(timeout)

# Show points
window.addstr(0, 0, 'Pies:')
window.addstr(1, 0, str(pies))

# Define the snake starting position and body
snake_x = screen_width/4
snake_y = screen_height/2
snake = [
    [snake_y, snake_x],
    [snake_y, snake_x-1],
    [snake_y, snake_x-2]
]

# Create initial food
food = [screen_height/2, screen_width/2]
window.addch(int(food[0]),int(food[1]), curses.ACS_PI)

# Generate initial poissonous food
poison = [
    random.randint(1,screen_heigth-1), 
    random.randint(1,screen_heigth-1)
    ]
window.addch(poison[0], poison[1], curses.ACS_DIAMOND)

# Action of the snake
action = 'r'
i = 0

# Playing
while True:
    next_key = window.getch()
    executions = actions.actions
    if i == 8:
        i = 0

    action = executions[i]

    # Losing the game (if it touches boarders or itself)
    if snake[0][0] in [0,screen_heigth-1] or snake[0][1] in [0,screen_width-1] or snake[0] in snake[1:] or pies < 0:
        curses.endwin()
        quit()

    # Defining new head, if food is eaten
    new_head = [snake[0][0], snake[0][1]]
        
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
    
    snake.insert(0, new_head)
    i += 1

    # Snake runs into the food
    if snake[0] == food:
        pies += 1
        speedset = True
        window.addstr(1, 0, str(pies))
        food = None
        while food is None:

            # Generate new food at random new position
            new_food = [
                random.randint(1,screen_heigth-1),
                random.randint(1, screen_width -1)
            ]
            
            # Check if food is not created in the snake
            food = new_food if new_food not in snake or not poison else None

        window.addch(food[0], food[1], curses.ACS_PI)

    else:
        tail = snake.pop()
        window.addch(int(tail[0]), int(tail[1]), ' ')

    window.addch(int(snake[0][0]), int(snake[0][1]), curses.ACS_CKBOARD)

    # Snake runs into poisson
    if snake[0] == poison:
        pies -= 1
        speedset = True
        window.addstr(1, 0, str(pies))
        poison = None
        while poison is None:

            # Generate new poison at random new position
            new_poison = [
                random.randint(1,screen_heigth-1),
                random.randint(1, screen_width -1)
            ]
            
            # Check if food is not created in the snake
            poison = new_poison if new_poison not in snake or not food else None

        window.addch(poison[0], poison[1], curses.ACS_DIAMOND)

    # After eating 5 pies the speed gets 25% faster
    if pies % 5 == 0 and speedset == True:
        speedset = False
        timeout = int(timeout*0.75)
        window.timeout(timeout)
