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
        
    # maintain the number of elite
    if n_elite:
        paired = zip(pop, scores)
        ranked = sorted(paired, key = lambda x: x[1], reverse = True)
        for i in range(n_elite):
            out[i] = ranked[i][0]
        
    # breed to fill in the rest of the output
    for i in range(n_elite, len(pop), 2):
        A = selection_func(pop, scores)
        B = selection_func(pop, scores)
        C, D = crossover_func(A, B)
        out[i] = mutation_func(C)
        if i != (len(pop)-1):
            out[i+1] = mutation_func(D)
            
    return out, scores

def crossover(A: Chromosome, B: Chromosome) -> tuple[Chromosome]:
    """
    Single point cross over

    Parameters
    ----------
    A
        A parent chromosome
    B
        A parent chromosome

    Returns
    -------
    The two crossed over chromosomes
    """
    assert A.shape == B.shape, "Cannot crossover differently shaped DNA"
    a = A.flatten()
    b = B.flatten()
    c = np.random.randint(1, len(a))
    a[:c], b[:c] = b[:c].copy(), a[:c].copy()
    return a.reshape(A.shape), b.reshape(B.shape)


def mutate(A: Chromosome, rate=0.05) -> Chromosome:
    """
    Mutate a chromosome

    | Param | Desc |
    | -     | -    |
    | A     | The chromosome the mutate |
    | rate  | the chance for any one gene to mutate | 

    ### returns 
    the mutated chromome

    """
    a = A.flatten()
    hits = np.random.binomial(len(a), rate)
    for _ in range(hits):
        c = np.random.randint(1, len(a))
        a[c] = np.random.normal()
    return a.reshape(A.shape)

def select(X: list[Chromosome], scores: list[float], k: int=2, 
    p: float=1) -> Chromosome:
    """
    Select a chromosome from the population


    """
    c = np.random.choice(len(scores), size=k, replace=False)
    selected = [scores[i] for i in c]
    return X[c[selected.index(max(selected))]]
    # TODO add support for P