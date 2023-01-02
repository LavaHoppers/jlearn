import time, math
import numpy as np
from functools import partial

# for importing the jlearn package
import sys; sys.path.append("..");
import jlearn as jl
from jlearn import *

# number of genes in a chromosome
d = 92

# how long the simulation runs for
ticks=50

# drag coeficient for the agent
C=.05

# for visualizing the simulation
screen_width = 500 # pixels

def spawn_food():
    "Get the coordinates to a new food somewhere on the screen"
    return np.random.uniform(low=-.8, high=.8, size=(2,1)) * screen_width/2

def fitness(x, visuals=False, samples=1):
    "Calculate the fitness of a chromosome"

    # our fitness function is aproximate... more samples increases acuracy
    if samples > 1:
        return sum([fitness(x, False, 1) for _ in range(samples)])/samples
       
    # init the food
    food = spawn_food()
    
    # init the agent
    score = 0
    W2, W1, b2, b1 = jl.util.multi_reshape(x, [(2,10), (10,6), (2,1), (10,1)])
    pos = np.zeros(shape=(2,1))
    vel = np.zeros(shape=(2,1))
        
    # main simulation loop
    for tick in range(ticks):
        
        # get the time that this tick started
        if visuals:
            start_time = time.time()

        # drag the agent (friction)
        vel = vel - C * vel

        # calculate the agent's move
        x1 = np.vstack((pos, vel, food))
        x2 = jl.func.sigmoid(W1@x1+b1)
        x3 = jl.func.sigmoid(W2@x2+b2) * 2 - 1
        acc = x3

        # accelerate and move
        vel += acc
        pos += vel
        
        # calc the distance squared from the food
        diff = pos-food
        sqr_dist = diff.T@diff
        
        # check if the food was eaten
        if sqr_dist < 100:
            score += 1
            food = spawn_food()
            
    # get the final distance to the food
    dist = math.sqrt(sqr_dist)
    # magic number 1000 just to smooth out the curve
    fitness_score = score + np.exp(-dist/1000)
    
    return fitness_score


# population size
n = 1000

# sub-routines 
Fitness = partial(fitness, samples=5)
Selection = partial(jl.genetics.tournament_select, k=2, p=.9)
Crossover = jl.genetics.crossover
Mutation = partial(jl.genetics.mutate, rate=0.01)

# elite to maintain from one population to the next
elite = 10

# generations to train for
gens = 10

# init the population
pop = [np.random.normal(size=(d,)) for _ in range(n)]

if __name__ == '__main__':
    for gen in range(gens):

        # get the next generation
        new_pop, scores = jl.genetics.step(pop, Fitness, Selection, Crossover, 
        Mutation, n_elite = elite, multiproc=True)
        
        print(scores[0],pop[0].__repr__())

        # update the population
        pop = new_pop