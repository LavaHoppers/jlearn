"""
Shows how to use genetic algorithms from jLearn
"""

import jlearn as jl
from jlearn import *
import numpy as np

# The number of genes in a choromosome, or the dimension of the solutions we
# want to analyze. 
d = 256

# The number of chromosomes in the population. Higher will usually give better
# solutions, but higher is also more expensive. 
n = 1_000

# We create a population of random guesses. You could use a heuristic for part 
# of this, but that generally will decrease diversity.
pop = [np.random.randint(0, 2, size=d) for _ in range(n)]

# Now we define the function we want to optimize; the fitness function. How fast
# this function runs will essentially be the limitting factor for how optimal of
# a solution you can find. The longer this takes, the more compute time the
# whole algorithm will take. This funciton can be _anything_ which is the beauty
# of genetic algorithms.

# For my example, we will try to find the 256 bit binary string that contains
# the most sequential binary integers starting at 0. For example, if the string
# had the substrings "0", "1" and "10" but not "11", that string would get a
# score of 3.
def fitness(x):
    """num of seq binary ints from 0 in x"""
    number = 0
    x = ''.join(map(str, x.tolist())) # bit string
    while True: # don't worry, we will get out
        str_number = f"{number:b}"
        if not str_number in x:
            break
        number += 1
    return number

# The selection pressure function. The more selection pressure the more you 
# will favor elite, but you will sacrifice diversity. We take a selection 
# function and fix the hyper parameters. Idealy you would tune these. I'm 
# choosing somewhat random numbers.
select = jl.genetics.Tournament_Select(3, 0.9)

# Let's setup our mutation function. This function brings diversity that wasn't 
# in the intial population. Higher p means more diversity, but less selection 
# presure.

# For our example, this step _could_ be skipped since our population will
# contain many of every possible bit.
binary = lambda: 1 if .5 < np.random.random() else 0 # only 1's and 0's
mutate = jl.genetics.PMutate(0.001, binary)

# This function creates the new chromosomes by mixing chromosomes that get 
# selected. The higher the number the 'more' mixing occurs.
crossover = jl.genetics.KCrossover(3)

# Create the step function to get the next generation. We simply pass in the 
# routines we defined earlier. We can also choose what percent of elite to 
# maintain. Maintaining more elite leads to less diversity, but may give 
# faster convergence. Generally, 0.1 is the most we would want to do.
step = jl.genetics.Step(fitness, select, crossover, mutate, p_elite = 0.05)

# The number of generations to explore. The more you explore the better your 
# chances of finding a good solution. Each generation costs the same amount, 
# but you usually get diminishing returns or find an optimal solution after
# a certain number. I am just going to use 20. You can also use any other 
# stopping criteria by examining the fitness scores.
gens = 100

# Putting the training loop in a main block is important if you want to use 
# multiprocessing for your fitness function. Our example is small enough that
# multiproc might slow the convergence down.
if __name__ == '__main__':

    # fitness scores over time
    best_scores = [None] * gens
    best_solution = [None] * gens

    # step through the generations
    for gen in range(gens):

        new_pop, scores = step(pop)

        # stats
        if not gen % 10:
            print(f'gen {gen} best fitness: {max(scores)}')
        
        best_scores[gen] = max(scores) # save results
        best_solution[gen] = pop[scores.index(max(scores))]

        pop = new_pop           # move to next gen!

    # graphically show results
    #import matplotlib.pyplot as plt
    #plt.plot(best_scores)
    #plt.xlabel('generation #')
    #plt.ylabel('fitness score')
    #plt.show()

    # With the hyperparemters as-is, the best fitness I saw was a 165. Tuning 
    # these parameters will almost certainly give a better solution.