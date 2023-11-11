from copy import copy

import numpy as np
from .bird import Bird

# TODO: el algoritmo debe aprender los pesos óptimos de la red neuronal para jugar el juego efectivamente.
class GeneticAlgorithm:
    def __init__(self, population_size, config, mutation_rate=0, crossover_rate=0):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness = []
        self.max_generations = 50  # Define cuántas generaciones correr
        
        self.config = config
        
        self.initialize_population(self.config)
        
        
    def initialize_population(self, config):
        """
        Initializes the population of birds for the genetic algorithm.

        Creates a list of Bird objects with a length equal to the population size.
        """
        self.population = [Bird(config) for _ in range(self.population_size)]

    def get_population(self):
        return self.population
    
    def set_population(self, population):
        self.population = population

    def evaluate_population(self):
        # Actualiza el fitness de cada pájaro en la población basado en el score obtenido
        for bird in self.population:
            bird.fitness = bird.score  # Asumiendo que 'score' ya ha sido actualizado

        # Si ninguna ave logró cruzar una tubería, entonces todas tienen fitness 0
        if all(bird.fitness == 0 for bird in self.population):
            # Podrías querer hacer algo especial aquí, como aumentar la tasa de mutación
            for bird in self.population:
                bird.fitness = 1  # Asignar un fitness mínimo para mantener la diversidad

    def select_parents(self):
        """
            Selecciona padres para la próxima generación utilizando el método de selección elitista.

            :param population: Lista de aves (instancias de la clase que contiene calculate_fitness y get_fitness).
            :return: Lista de padres seleccionados.
        """

        # Calcula la aptitud para cada ave en la población
        for bird in self.population:
            bird.calculate_fitness()

        # Ordena la población por su aptitud (de mayor a menor)
        sorted_population = sorted(self.population, key=lambda bird: bird.get_fitness(), reverse=True)

        # Selecciona los mejores individuos para ser padres (por ejemplo, la mitad superior)
        num_parents = len(self.population) // 2
        parents = sorted_population[:num_parents]

        return parents


    def calculate_fitness(self):
        """
        Calculates the fitness of each bird in the population.
        """
        for bird in self.population:
            bird.calculate_fitness()
    

    def crossover(self, parent1, parent2):
        """
            Realiza el cruce de un punto en los pesos de dos redes neuronales.

            :param parent1: El primer modelo de red neuronal (padre).
            :param parent2: El segundo modelo de red neuronal (padre).
            :return: Dos nuevos conjuntos de pesos (para dos hijos).
        """

        # Obtener los pesos de los padres
        weights1 = parent1.get_model_instance().get_model().get_weights()
        weights2 = parent2.get_model_instance().get_model().get_weights()

        # Asegurarse de que los padres tengan la misma estructura de pesos
        assert len(weights1) == len(weights2)

        # Elegir un punto de cruce aleatorio
        crossover_point = np.random.randint(1, len(weights1))

        # Realizar el cruce de un punto
        child1_weights = weights1[:crossover_point] + weights2[crossover_point:]
        child2_weights = weights2[:crossover_point] + weights1[crossover_point:]

        return child1_weights, child2_weights

    def mutate(self, weights, mutation_rate=0.01, mutation_scale=0.1):
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


    def generate_next_generation(self):
        """
        Genera la próxima generación a partir de la población actual.

        :param current_population: La población actual de modelos de red neuronal.
        :return: La nueva generación de modelos.
        """

        # Paso 1: Seleccionar padres
        parents = self.select_parents()

        # Paso 2: Crear la próxima generación
        new_population = []
        while len(new_population) < len(self.population):
            # Selecciona dos padres al azar para el cruce
            parent1, parent2 = np.random.choice(parents, 2, replace=False)

            # Realiza el cruce para crear dos hijos
            child1_weights, child2_weights = self.crossover(parent1, parent2)

            # Aplica la mutación a los pesos de los hijos
            child1_weights = self.mutate(child1_weights)
            child2_weights = self.mutate(child2_weights)
        
            # Crea nuevos modelos para los hijos y añádelos a la nueva población
            child1 = Bird(self.config)
            child1.get_model_instance().set_weights(child1_weights)
            new_population.append(child1)

            child2 = Bird(self.config)
            child2.get_model_instance().set_weights(child2_weights)
            
            new_population.append(child2)

        self.population = new_population
    
    def generate_new_population(self):
        self.calculate_fitness()
        self.generate_next_generation()


    def migrate_population(self, bird, new_population):
        """
        Migrates the bird to the new population.

        Replaces the bird in the current population with the bird in the new population.
        [GAME LOOP]
        """
        self.population.remove(bird)
        new_population.append(bird)