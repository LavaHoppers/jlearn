''' 
Functions for genetic algorithms
'''

import multiprocessing
import numpy as np
from typing import Callable

# typing
Chromosome = np.ndarray
Population = list[Chromosome]

def step(pop: Population, fitness_func: Callable[[Chromosome], float],
         selection_func: Callable[[Population, list[float]], Chromosome],
         crossover_func: Callable[[Chromosome, Chromosome], 
                                  tuple[Chromosome, Chromosome]],
         mutation_func: Callable[[Chromosome], Chromosome],
         multiproc: bool = False, fitness_samples: int = 1, n_elite: int = 0
        ) -> tuple[Population, list[float]] :
    """
    Get the next population

    This function should be called in a loop with it's output population 
    being fed back into pop. Each step should produce a more fit population. 

    Parameters
    ----------
    pop
        The population to improve the fitness of. The more fit a population is, 
        the more fit the output will be.
    fitness_func
        Evaluates the fitnes of a single chromosome. The return value should be 
        >= 0. Higher return value should mean better (more desireable) fitness. 
    selection_func
        Given a set of chromosomes and their matching fitness scores, select and 
        return a single chromosome for breeding.
    crossover_func
        Given two chromosomes, return two new combined chromosomes.
    mutation_func
        Returns a randomly modified chromosome.
    multiproc
        Whether to use multiprocessing to evalute fitness. This may not work in 
        an IPython enviornment.
    fitness_samples
        Number of times to evalute the fitness of a chromosome. The chromosome 
        is assigned its average score.
    n_elite
        The number of elite chromosomes to maintain from input population to the 
        output. 
    
    Returns
    -------
    Returns the output population of chromosomes and the fitness scores of the 
    input population. 
    """
    
    # the population to return
    out = [None] * len(pop)
    
    # evaluate the scores of each chromosome
    if multiproc:
        with multiprocessing.Pool() as pool:
            scores = pool.map(fitness_func, pop)
    else:
        scores = [fitness_func(x) for x in pop]
        
    # rank the chromos from high to low fitness
    paired = zip(pop, scores)
    ranked = sorted(paired, key = lambda x: x[1], reverse = True)

    # maintain the number of elite
    if n_elite:
        for i in range(n_elite):
            out[i] = ranked[i][0]
        
    # breed to fill in the rest of the output
    for i in range(n_elite, len(pop), 2):

        # select two chromose for breeding
        A = selection_func(ranked)
        B = selection_func(ranked)

        # breeing
        C, D = crossover_func(A, B)
        out[i] = mutation_func(C)
        if i != (len(pop)-1):
            out[i+1] = mutation_func(D)
            
    return out, scores

def k_crossover(A: Chromosome, B: Chromosome, k=1) -> tuple[Chromosome]:
    """
    k point cross over

    Parameters
    ----------
    A
        A parent chromosome
    B
        A parent chromosome
    k
        number of crossovers

    Returns
    -------
    The two crossed over chromosomes
    """
    assert A.shape == B.shape, "Cannot crossover differently shaped DNA"
    a = A.flatten()
    b = B.flatten()
    inx_crosses =  sorted(np.random.choice(len(a), size=k))
    for c in inx_crosses:
        a[:c], b[:c] = b[:c].copy(), a[:c].copy()
    return a.reshape(A.shape), b.reshape(B.shape)


def mutate(A: Chromosome, p=0.05, val=np.random.normal) -> Chromosome:
    """
    Mutate a chromosome

    | Param | Desc |
    | -     | -    |
    | A     | The chromosome the mutate |
    | p  | the chance for any one gene to mutate | 
    | val | returns the new value for the gene |

    ### returns 
    the mutated chromome

    """
    a = A.flatten()
    hits = np.random.binomial(len(a), p)
    for _ in range(hits):
        c = np.random.randint(1, len(a))
        a[c] = val()
    return a.reshape(A.shape)

def tournament_select(ranked: list[tuple[Chromosome, float]], k: int=2, 
    p: float=1) -> Chromosome:
    """
    Select a chromosome from the population

    """
    # get the sorted subset of the population
    idx_comps = np.random.choice(len(ranked), size=k, replace=False)
    contestants = [ranked[i][0] for i in sorted(idx_comps)]

    # get the index of the tournament winner
    # https://en.wikipedia.org/wiki/Geometric_distribution
    idx_winner = (np.random.geometric(p)-1) % k if p < 1 else 0

    return contestants[idx_winner]

def roulette_select(ranked: list[tuple[Chromosome, float]]) -> Chromosome:
    """
    Select a chromosome from the population
    """
    total_fitness = sum([x[1] for x in ranked])
    probs = [x[1]/total_fitness for x in ranked]

    idx_winner = np.random.choice(len(ranked), p=probs)

    return ranked[idx_winner][0]
