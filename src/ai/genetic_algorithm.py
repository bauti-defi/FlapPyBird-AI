from copy import copy
from .bird import Bird

# TODO: el algoritmo debe aprender los pesos óptimos de la red neuronal para jugar el juego efectivamente.
class GeneticAlgorithm:
    def __init__(self, population_size, config, mutation_rate=0, crossover_rate=0):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.scores = []
        self.max_generations = 50  # Define cuántas generaciones correr
        
        self.initialize_population(config)
        
    def initialize_population(self, config):
        """
        Initializes the population of birds for the genetic algorithm.

        Creates a list of Bird objects with a length equal to the population size.
        """
        self.population = [Bird(config) for _ in range(self.population_size)]

    def get_population(self):
        return copy(self.population)

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

    def generate_new_population(self, config):
        # Crea una nueva generación mediante cruzamiento y mutación
        self.population = [Bird(config) for _ in range(self.population_size)]
        return self.population
