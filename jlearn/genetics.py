''' 
Functions for genetic algorithms
'''

import concurrent.futures
import numpy as np
from typing import Callable

# typing
Chromosome = np.ndarray
Population = list[Chromosome]

class Step():

    def __init__(self, fitness_func: Callable[[Chromosome], float],
        selection_func: Callable[[Population, list[float]], Chromosome],
        crossover_func: Callable[[Chromosome, Chromosome], 
                                tuple[Chromosome, Chromosome]],
        mutation_func: Callable[[Chromosome], Chromosome],
        multiproc: bool = False, fitness_samples: int = 1, p_elite: float = 0.05):
        """Configure a step function for getting the next population

        Args:
            fitness_func (Callable[[Chromosome], float]): Evaluates the fitnes of a single chromosome. The return value should be >= 0. Higher return value should mean better (more desireable) fitness. 
            selection_func (Callable[[Population, list[float]], Chromosome]): Given a set of chromosomes and their matching fitness scores, select and 
            return a single chromosome for breeding.
            crossover_func (Callable[[Chromosome, Chromosome], tuple[Chromosome, Chromosome]]): Given two chromosomes, return two new combined chromosomes.
            mutation_func (Callable[[Chromosome], Chromosome]): Returns a randomly modified chromosome.
            multiproc (bool, optional): run using multiprocessing. only use this if your population is large or if you have an expensive fitness function. Defaults to False.
            fitness_samples (int, optional): resample the fitness function and average the score for non-deterministic fitness functions. Defaults to 1.
            p_elite (float, optional): the percent of elite to maintain in each generation. Defaults to 0.05.
        """
        self.fitness = fitness_func
        self.selection = selection_func
        self.crossover = crossover_func
        self.mutation = mutation_func
        self.multiproc = multiproc
        self.fitness_samples = fitness_samples
        self.p_elite = p_elite
        # can use ThreadPoolExecutor if you are I/O bound for fitness
        self.exe = concurrent.futures.ProcessPoolExecutor()

    def __call__(self, pop: Population) -> tuple[Population, list[float]]:
        """Get the next generation of the population

        Args:
            pop (Population): a population to evolve

        Returns:
            tuple[Population, list[float]]: the new population and the input population's fitness scores
        """
        
        # the population to return
        out = [None] * len(pop)
        
        # evaluate the scores of each chromosome
        if self.multiproc:
            scores = list(self.exe.map(self.fitness, pop))
        else:
            scores = [self.fitness(x) for x in pop]
            
        # rank the chromos from high to low fitness
        paired = zip(pop, scores)
        ranked = sorted(paired, key = lambda x: x[1], reverse = True)

        n_elite = int(self.p_elite * len(pop))

        # maintain the number of elite
        if n_elite:
            for i in range(n_elite):
                out[i] = ranked[i][0]
            
        # breed to fill in the rest of the output
        for i in range(n_elite, len(pop), 2):

            # select two chromose for breeding
            x = self.selection(ranked)
            y = self.selection(ranked)

            # breeing
            x, y = self.crossover(x, y)
            out[i] = self.mutation(x)
            if i != (len(pop)-1):
                out[i+1] = self.mutation(y)
                
        return out, scores

    def end(self):
        """Call this if you use multiproc
        """
        self.exe.shutdown()

class KCrossover():
    """
    Crossover routine that does a set amount of crosses
    """

    def __init__(self, k: int = 1):
        """Returns a new crossover routine

        Args:
            k (int, optional): The number of crossover points. Defaults to 1.

        Raises:
            Exception: K must be a positive integer
        """

        if not isinstance(k, int) or k < 1:
            raise Exception('k must be a positive integer')

        self.k = k

    def __call__(self, x: Chromosome, y: Chromosome) -> tuple[Chromosome]:
        """Crossover two chromosomes at self.k points

        Args:
            x (Chromosome): The first chromosome
            y (Chromosome): The second chromosome

        Raises:
            TypeError: Inputs must be numpy.ndarrays
            Exception: The inputs must have the same shape
            Exception: The inputs must not contain more elements than self.k

        Returns:
            tuple[Chromosome]: The two new chromosomes
        """
         
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError(f"Crossover takes two np.ndarray not "\
                f"{type(x)} and {type(y)}")

        if not x.shape == y.shape:
            raise Exception("Cannot crossover differently shaped DNA")

        a = x.flatten()
        b = y.flatten()

        k = self.k

        if len(a) < k:
            raise Exception(f"cannot crossover {k} times with {len(a)} genes")

        # sorted to avoid "switching back" the same genes
        idx_crosses =  sorted(np.random.choice(len(a), size=k))

        # at each cross, swtich the tails
        for c in idx_crosses:
            a[:c], b[:c] = b[:c].copy(), a[:c].copy()

        return a.reshape(x.shape), b.reshape(y.shape)
   

class PMutate():
    """Mutates a chromosome
    """

    def __init__(self, p: float = 0.01, value: Callable = np.random.normal):
        """Create a new mutation routine

        Args:
            p (float, optional): The chance for any one gene to mutate. Defaults to 0.01.
            value (Callable, optional): The routine to return a new gene. Defaults to np.random.normal.

        Raises:
            Exception: P should be a positve float
        """
        if not isinstance(p, float):
            raise Exception("p should be a float")
        if p <= 0: 
            raise Exception("p should be positive")

        self.p = p

        # TODO add checks for the value
        self.value = value

    def __call__(self, x: Chromosome) -> Chromosome:
        """mutate a chromosome

        each gene in the chromosome has chance self.p to be replaced

        Args:
            x (Chromosome): the chromosome to mutate

        Returns:
            Chromosome: the new mutated chromosome
        """
        a = x.flatten()
        n_hits = np.random.binomial(len(a), self.p)
        idx_hits = np.random.choice(len(a), size=n_hits, replace=False)
        for i in idx_hits:
            a[i] = self.value()
        return a.reshape(x.shape)

class Tournament_Select():

    def __init__(self, k:int=2, p:float=1):
        """Create a new selection pressure routine

        Args:
            k (int, optional): The number of chromosomes to pick from. Defaults to 2.
            p (float, optional): The chance for selection. Defaults to 1.
        """
        self.k = k
        self.p = p
    
    def __call__(self, ranked: list[tuple[Chromosome, float]]) -> Chromosome:
        """Get a chromosome from a population based on selection pressure

        Args:
            ranked (list[tuple[Chromosome, float]]): The chromosomes and their fitness values in order of fitness

        Returns:
            Chromosome: the selected chromosome
        """
        # get the sorted subset of the population
        idx_comps = np.random.choice(len(ranked), size=self.k, replace=False)
        contestants = [ranked[i][0] for i in sorted(idx_comps)]

        # get the index of the tournament winner
        # https://en.wikipedia.org/wiki/Geometric_distribution
        i = (np.random.geometric(self.p)-1) % self.k if self.p < 1 else 0

        return contestants[i]

def roulette_select(ranked: list[tuple[Chromosome, float]]) -> Chromosome:
    """
    Select a chromosome from the population
    """
    total_fitness = sum([x[1] for x in ranked])
    probs = [x[1]/total_fitness for x in ranked]

    idx_winner = np.random.choice(len(ranked), p=probs)

    return ranked[idx_winner][0]
