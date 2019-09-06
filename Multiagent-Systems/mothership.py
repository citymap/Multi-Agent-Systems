import numpy as np
import random
import math
import pygame
import time

n =  40             # Number of robots
w1 = np.random.randint(0,10,(n,n), dtype=int)
r = np.random.rand(n,n)

# Define samples, crumbs and obstacles positions
samples = np.where(r>0.99,w1,0)
crumbs = np.zeros((n,n), dtype=int)
obstacles = np.zeros((n,n), dtype=int)

# Position of the base
base = (0,0)

# Helper to get local position of the robot
pos_distribution = np.linspace(-350,350,n)
grid_distribution = np.zeros((n,n,2))

# Iterate over all positions to define the grid positioning
for i in range(0,n):
    for j in range(0,n):
        grid_distribution[j,i] = np.array([pos_distribution[i],pos_distribution[j]])

print(grid_distribution)

# Screen parameters
width = 800
height = 800
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
yellow = (255,255, 0)


# Returns if the rectangle is in the board
def try_pos(pos):
    if 0 <= pos[0] < n and 0 <= pos[1] < n:
        return True
    else:
        return False

# Convert coordinates form cartesian to screen coordinates (used to draw in pygame screen)
def cartesian_to_screen(car_pos):
    factor = 1
    screen_pos = np.array([center[0]*factor+car_pos[0],center[1]*factor+car_pos[1]])/factor
    screen_pos = screen_pos.astype(int)
    return screen_pos

# Drawing Board
def draw():
    pygame.event.get()
    screen.fill((0, 0, 0))

    for robot in robots:
        # print(robot.pos, grid_distribution[robot.pos])
        pygame.draw.circle(screen, green, cartesian_to_screen(grid_distribution[robot.pos]),  5)

    for i in range(n):
        for j in range(n):
            pygame.draw.circle(screen, yellow, cartesian_to_screen(grid_distribution[(i, j)]), samples[i,j])
            pygame.draw.circle(screen, red, cartesian_to_screen(grid_distribution[(i, j)]), int(crumbs[i,j]/2))
    pygame.display.flip()


# Class robot
class Robot:
    def __init__(self):
        self.pos = (0,0)
        self.carrying = False
        self.samples_collected = 0

    # Perform action
    def act(self):
        global crumbs
        global samples

        # All possible directions
        near_pos = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        # If carrying sample and arrived to base
        if self.carrying and self.pos == base:
            # Drop sample and increase counter
            self.carrying = False
            self.samples_collected +=1

        # If carrying sample and not in base
        elif self.carrying and self.pos!= base:
            # Leave 2 crumbs and travel in gradient up
            crumbs[self.pos] = crumbs[self.pos]+2
            distances = []
            for near in near_pos:
                distances.append(math.sqrt((self.pos[0]+near[0])**2 + (self.pos[1]+near[1])**2))
            min_index = np.argmin(np.array(distances))
            new_pos = (self.pos[0]+near_pos[min_index][0], self.pos[1]+ near_pos[min_index][1])
            if try_pos(new_pos):
                self.pos = new_pos

        # If found sample
        elif samples[self.pos] >0:
            # Carry sample
            self.carrying = True
            samples[self.pos] = samples[self.pos]-1

        # If found crumbs
        elif crumbs[self.pos] >0:
            # Pick 1 crumb and travel in gradient down
            crumbs[self.pos] = crumbs[self.pos]-1
            distances = []
            for near in near_pos:
                distances.append(math.sqrt((self.pos[0] + near[0]) ** 2 + (self.pos[1] + near[1]) ** 2))
            max_index = distances.index(max(distances))
            new_pos = (self.pos[0]+near_pos[max_index][0], self.pos[1]+ near_pos[max_index][1])
            if try_pos(new_pos):
                self.pos = new_pos

        # Otherwise
        else:
            # Move random pos
            a = random.randint(0,7)
            new_pos = (self.pos[0]+near_pos[a][0],self.pos[1]+near_pos[a][1])
            if try_pos(new_pos):
                self.pos = new_pos

# Define list of robots

robots = []
for i in range(20):
    # Add 
    robots.append(Robot())

# Start and continue simulation
while True:
    for robot in robots:
        robot.act()
    draw()


