from typing import List

import numpy as np
from .bird import Bird

class GeneticAlgorithm:
    def __init__(self, population_size, config, mutation_rate=0, crossover_rate=0):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.config = config
        
        self.population = []
        self.fitness = []

        self.initialize_population(self.config)
        
        
    def initialize_population(self, config):
        """
        Initializes the population of birds for the genetic algorithm.

        Creates a list of Bird objects with a length equal to the population size.
        """
        self.population = [Bird(config) for _ in range(self.population_size)]
        # self.load_generation()

    def get_population(self):
        return self.population
    
    def set_population(self, population):
        self.population = population

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

    def select_parents_v2(self) -> List[Bird]:
        """
        Selecciona padres para la próxima generación utilizando el método de selección "Accept-Reject".
        :return: Lista de padres seleccionados.
        """
        fitness_scores = [bird.get_fitness() for bird in self.population]
        max_fitness = max(fitness_scores)
        num_parents = len(self.population) // 2
        selected_parents = []

        while len(selected_parents) < num_parents:
            # Elegir un individuo al azar
            candidate_idx = np.random.randint(len(self.population))
            # Generar un número aleatorio entre 0 y el máximo fitness
            threshold = np.random.uniform(0, max_fitness)

            # Aceptar el individuo si su fitness es mayor que el umbral
            if fitness_scores[candidate_idx] >= threshold:
                selected_parents.append(self.population[candidate_idx])

        return selected_parents

    def calculate_fitness(self):
        """
        Calculates the fitness of each bird in the population.
        """
        for bird in self.population:
            bird.calculate_fitness()
    

    def crossover(self, parent1: Bird, parent2: Bird):
        """
            Realiza el cruce de un punto en los pesos de dos redes neuronales.

            :param parent1: El primer modelo de red neuronal (padre).
            :param parent2: El segundo modelo de red neuronal (padre).
            :return: Dos nuevos conjuntos de pesos (para dos hijos).
        """
        
        # Obtener los pesos de los padres
        weights1 = parent1.model.get_model_weights()
        weights2 = parent2.model.get_model_weights()

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

    def model_mutate_v2(self, weights):
        for xi in range(len(weights)):
            for yi in range(len(weights[xi])):
                if np.random.uniform(0, 1) > 0.80: # 20% chance of mutation
                    change = np.random.uniform(-0.5,0.5)
                    weights[xi][yi] += change
        return weights

    def model_crossover_v2(self, model1, model2):
        weights1 = model1.model.get_model_weights()
        weights2 = model2.model.get_model_weights()
        new_weights1 = []
        new_weights2 = []
        for w1, w2 in zip(weights1, weights2):
            if len(w1.shape) == 2:  # If weight is for Dense layer
                idx = np.random.randint(0, w1.shape[0])
                new_w1 = np.vstack((w1[:idx, :], w2[idx:, :]))
                new_w2 = np.vstack((w2[:idx, :], w1[idx:, :]))
                new_weights1.append(new_w1)
                new_weights2.append(new_w2)
            else:  # Bias or other types of layers
                new_weights1.append(w1)
                new_weights2.append(w2)
        return new_weights1, new_weights2
        

    def generate_next_generation(self):
        """
        Genera la próxima generación a partir de la población actual.
        """
        
        # Calculo el fitness
        self.calculate_fitness()

        # Paso 1: Seleccionar padres
        parents = self.select_parents_v2()
        # Paso 2: Crear la próxima generación
        new_population = []
        
        while len(new_population) < len(self.population):
            # Selecciona dos padres al azar para el cruce
            parent1, parent2 = np.random.choice(parents, 2, replace=False)

            # Realiza el cruce para crear dos hijos
            child1_weights, child2_weights = self.model_crossover_v2(parent1, parent2)

            # Aplica la mutación a los pesos de los hijos
            child1_weights = self.model_mutate_v2(child1_weights)
            child2_weights = self.model_mutate_v2(child2_weights)
        
            # Crea nuevos modelos para los hijos y añádelos a la nueva población
            child1 = Bird(self.config)
            child1.model.set_model_weights(child1_weights)
            new_population.append(child1)

            child2 = Bird(self.config)
            child2.model.set_model_weights(child2_weights)
            
            new_population.append(child2)

        self.population = new_population
    
    def generate_new_population(self):
        self.generate_next_generation()
        self.save_generation()

    def save_generation(self):
        for index, bird in enumerate(self.population):
            bird.model.save_model(f"gen_{index}")
            
    def load_generation(self):
        for index, bird in enumerate(self.population):
            bird.model.load_model(f"gen_{index}")

    def migrate_population(self, bird, new_population):
        """
        Migrates the bird to the new population.

        Replaces the bird in the current population with the bird in the new population.
        [GAME LOOP]
        """
        self.population.remove(bird)
        new_population.append(bird)