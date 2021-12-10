import copy

import linear_regression_ol as lro
import numpy as np
from scipy.special import softmax
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class GeneticSolver:
    def __init__(self, population_size: int = 2000, mutation_rate: float = 0.05, num_gens: int = 50,
                 cook_distance: bool = True, dataset: str = 'animals') -> None:
        self.dataset = lro.load_dataset(dataset)
        self.dataset_type = dataset
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_gens = num_gens
        self.cook_distance = cook_distance

    def make_chromosomes(self, string_length) -> np.ndarray:
        chromosomes = np.ndarray(shape=(self.population_size, string_length),
                                 dtype=int)
        for i in range(self.population_size):
            indices = np.random.choice(np.arange(self.dataset.shape[0]), size=string_length,
                                       replace=False)
            indices.sort()
            chromosomes[i] = indices
        return chromosomes

    def make_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[List[int], List[int]]:
        child1 = [parent1[0], parent1[1]]
        i = 2
        while len(child1) < 4:
            if parent2[i] not in child1:
                child1.append(parent2[i])
            i += 1
            i %= 4

        child2 = [parent2[0], parent2[1]]
        i = 2
        while len(child2) < 4:
            if parent1[i] not in child2:
                child2.append(parent1[i])
            i += 1
            i %= 4
        return child1, child2

    def assess_fitness(self, chromosomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        fitnesses = lro.remove_outliers(chromosomes, self.dataset_type, cook_distance=self.cook_distance)
        total_fitness = np.sum(fitnesses)
        print(total_fitness)
        normalized_fitnesses = (((1 - (fitnesses / total_fitness)) * 1000) % 1 * 10)
        normalized_fitnesses = softmax(normalized_fitnesses)
        return chromosomes, normalized_fitnesses, fitnesses

    def select_parents(self, chromosomes: np.ndarray, nor_fits: np.ndarray) -> np.ndarray:
        parents_indexes = np.random.choice(np.arange(chromosomes.shape[0]), p=nor_fits, replace=True,
                                           size=self.population_size)
        parents = chromosomes[parents_indexes]
        return parents

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = np.float64(x - max(x))
        numerator = np.float64(np.exp(z))
        denominator = np.float64(np.sum(numerator))
        softmax = np.float64(numerator / denominator)
        return softmax

    def reproduce(self, parents: np.ndarray) -> np.ndarray:
        children = np.ndarray(shape=(self.population_size, 4),
                              dtype=int)
        k = 0
        for i in range(0, parents.shape[0] - 1, 2):
            child1, child2 = self.make_crossover(parents[i], parents[i + 1])
            children[k] = child1
            k += 1
            children[k] = child2
            k += 1
        return children

    def mutate(self, children: np.ndarray) -> np.ndarray:
        for idx, child in enumerate(children):
            num = np.random.uniform(0, 1, 1)
            if num <= self.mutation_rate:
                max_num = np.max(children)
                new_outlier = np.random.randint(0, max_num, dtype=int)
                while new_outlier in child:
                    new_outlier = np.random.randint(0, max_num, dtype=int)
                pos = np.random.randint(0, 4, dtype=int)
                children[idx][pos] = new_outlier
        return children

    def run_generation(self, chromosomes: np.ndarray) -> None:
        for i in range(self.num_gens):
            chromosomes, nor_fits, fitnesses = self.assess_fitness(chromosomes)
            parents = self.select_parents(chromosomes, nor_fits)
            children = self.reproduce(parents)
            children = self.mutate(children)
            chromosomes = np.array(copy.deepcopy(children))

            if i == self.num_gens - 1 or i % 5 == 0:
                chromosomes, nor_fits, fitnesses = self.assess_fitness(chromosomes)
                best_combination_idx = np.argmin(np.array(fitnesses))
                print(f'Best combination for {i + 1} generation: {chromosomes[best_combination_idx]}')


if __name__ == '__main__':
    gs = GeneticSolver(num_gens=20, dataset='stars', cook_distance=False)
    chromosomes = gs.make_chromosomes(string_length=4)
    gs.run_generation(chromosomes)
    pass