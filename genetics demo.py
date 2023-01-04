"""
Shows how to use genetic algorithms from jLearn
"""

import jlearn as jl
from jlearn import *
import numpy as np
from functools import partial

# protection for the multiprocessing lib
if __name__ == '__main__':

    # number of genes in a choromosome
    d = 20
    # number of chromosomes in a population
    n = 200

    # create a population of random guesses. Can use a heuristic to try to speed up
    # convergence.
    pop = [np.random.randint(0, 2, size=d) for _ in range(n)]

    # some function that meassures the fitness of a chromosome. this is the function
    # you are optimizing. Higher is better. Generally, do not return negative values
    def fitness(x):
        """Encode 1337 using powers of 2"""
        approx = 0
        for i, gene in enumerate(x):
            if gene:
                approx += 2 ** i
        diff = np.abs(1337 - approx)
        return 1 / diff if diff else 2

    # The selection pressure function. The more selection pressure the more you will
    # favor elite, but you will sacrifice diversity. We take a selection function and 
    # fix the hyper parameters. Idealy you would tune these.
    select = partial(
        jl.genetics.tournament_select, k=3, p=1
    )

    # setup our mutation function. for each off, they will be a combo of their
    # parents and a small amount of random
    def binary():
        return 1 if .5 < np.random.random() else 0
    mutate = partial(
        jl.genetics.mutate, p=.01, val=binary
    )

    # finally, the crossover function
    crossover = partial(
        jl.genetics.k_crossover, k=3
    )

    # number of generations to explore. the more you explore the better your chances
    # of finding a good solution
    gens = 20

    # step through the generations. you could stop if you reach a certain
    # fitness
    for gen in range(gens):

        new_pop, scores = jl.genetics.step(pop, fitness, select, crossover, mutate,
            n_elite = 2)

        print(f'gen {gen} best fitness: {max(scores)}')
        print(pop[scores.index(max(scores))])

        pop = new_pop

    