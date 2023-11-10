from copy import copy
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
        
        self.initialize_population(config)
        
        
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

    def selection(self):
        # Selecciona los mejores individuos para la reproducción
        pass


    def calculate_fitness(self):
        """
        Calculates the fitness of each bird in the population.
        """
        for bird in self.population:
            bird.calculate_fitness()
    
    
    def generate_new_population(self):
        self.calculate_fitness()


    def migrate_population(self, bird, new_population):
        """
        Migrates the bird to the new population.

        Replaces the bird in the current population with the bird in the new population.
        [GAME LOOP]
        """
        self.population.remove(bird)
        new_population.append(bird)