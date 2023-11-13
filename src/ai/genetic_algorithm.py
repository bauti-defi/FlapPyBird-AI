import numpy as np
from typing import List

from .bird import Bird
from .selection import mating_pool, roulette_wheel_selection
from .mutations import gaussian_mutation, model_mutate, mutate_weights, mutate_weights_v2
from .crossover import crossover


class GeneticAlgorithm:
    def __init__(self, config):
        self.config = config
        self.population = []
        self.fitness = []
        self.highscore = 0
        
    
    def set_population(self, new_population):
        self.population = new_population
        
    def get_population(self) -> List[Bird]:
        return self.population
        
    def calculate_fitness(self):
        """
        Calculates the fitness of each bird in the population.
        """
        for bird in self.population:
            bird.calculate_fitness()
            self.fitness.append(bird.get_fitness())
    
    def select_parents(self) -> List[Bird]:
        """
            Selecciona padres para la próxima generación utilizando el método de selección elitista.
            :return: Lista de padres seleccionados.
        """
        # Ordena la población por su aptitud (de mayor a menor)
        sorted_population = sorted(self.population, key=lambda bird: bird.get_fitness(), reverse=True)
        # Selecciona los mejores individuos para ser padres (por ejemplo, la mitad superior)
        num_parents = len(self.population) // 2
        
        parents = sorted_population[:num_parents]

        return parents
    
    def crossover(self, parent1: Bird, parent2: Bird):
        """
            Realiza el cruce de un punto en los pesos de dos redes neuronales.
            :param parent1: El primer modelo de red neuronal (padre).
            :param parent2: El segundo modelo de red neuronal (padre).
            :return: Dos nuevos conjuntos de pesos (para dos hijos).
        """

        # Obtener los pesos de los padres
        weights1 = parent1.model.get_weights()
        weights2 = parent2.model.get_weights()

        # Asegurarse de que los padres tengan la misma estructura de pesos
        assert len(weights1) == len(weights2)

        # Elegir un punto de cruce aleatorio
        crossover_point = np.random.randint(1, len(weights1))

        # Realizar el cruce de un punto
        child1_weights = weights1[:crossover_point] + weights2[crossover_point:]
        child2_weights = weights2[:crossover_point] + weights1[crossover_point:]

        return child1_weights, child2_weights

    def mutate(self, weights, mutation_rate=0.1, mutation_scale=0.1):
        """
        Aplica una mutación a los pesos de una red neuronal.
        :param weights: Los pesos de la red neuronal a mutar.
        :param mutation_rate: La probabilidad de que cada peso sea mutado.
        :param mutation_scale: La magnitud de las mutaciones.
        :return: Los pesos mutados.
        """
        mutated_weights = []
        for weight_matrix in weights:
            # Aplica una mutación a cada peso individualmente
            if np.random.rand() < mutation_rate:
                mutation = np.random.normal(loc=0.0, scale=mutation_scale, size=weight_matrix.shape)
                mutated_weight_matrix = weight_matrix + mutation
            else:
                mutated_weight_matrix = weight_matrix
            mutated_weights.append(mutated_weight_matrix)
        return mutated_weights

    def create_new_generation(self):
        # Paso 1: Calculamos el fitness
        self.calculate_fitness()
        
        # Paso 2: Seleccionar padres
        parents = self.select_parents()
        
        # Paso 3: Crear la próxima generación
        new_population = []
        
        while len(new_population) < len(self.population):
            # Selecciona dos padres al azar para el cruce
            parent1, parent2 = np.random.choice(parents, 2, replace=False)

            # Realiza el cruce para crear dos hijos
            child1_weights, child2_weights = crossover(parent1, parent2)

            # # Aplica la mutación a los pesos de los hijos
            mutated_child1_W1 = mutate_weights_v2(child1_weights[0], mutation_rate=0.1)
            mutated_child1_W2 = mutate_weights_v2(child1_weights[1], mutation_rate=0.1)
            mutated_child2_W1 = mutate_weights_v2(child2_weights[0], mutation_rate=0.1)
            mutated_child2_W2 = mutate_weights_v2(child2_weights[1], mutation_rate=0.1)
            
            # Crea nuevos modelos para los hijos y añádelos a la nueva población
            child1 = Bird(self.config)
            child1.model.set_weights(mutated_child1_W1, mutated_child1_W2)
            new_population.append(child1)

            child2 = Bird(self.config)
            child2.model.set_weights(mutated_child2_W1, mutated_child2_W2)

            new_population.append(child2)

        self.population = new_population
        
        
    def get_new_generation(self):
        self.fitness = []
        self.create_new_generation()