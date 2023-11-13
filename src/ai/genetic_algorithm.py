import copy
import numpy as np
from typing import List

from .bird import Bird
from .mutations import  mutate_weights_v2
from .crossover import crossover

BIRD_NUMBER = 12 
class GeneticAlgorithm:
    def __init__(self, config):
        self.config = config
        self.population = []
        self.fitness = []

        self.allTimeBestBird = None
        
        self.highscore = 0
        self.generation = 1
        self.highgen = 0

        self.maxscore = 0
        
        self.globalFitness = 0.0
        self.respawn = False

        self.bestInputWeights = np.zeros((5, 3))
        self.bestHiddenWeights = np.zeros((3, 1))
        self.birdsToBreed = [] # Cantidad de pájaros a reproducir
        
    
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
        
    
    def parents_select(self):
        """
            Selecciona los padres para la próxima generación basándose en su aptitud (fitness).

            Esta función realiza los siguientes pasos:
            1. Pre-calcula la aptitud de cada pájaro y la almacena junto con su índice en la población.
            2. Ordena a los pájaros por su aptitud y selecciona los dos con mayor aptitud.
            3. Actualiza el puntaje alto (highscore) y el mejor pájaro de todos los tiempos si se encuentra un nuevo récord.
            4. Añade los pájaros seleccionados a la lista de pájaros para cría.
            5. Elimina los pájaros seleccionados de la población actual.
            6. Comprueba si es necesario reiniciar la población basándose en el progreso de las generaciones.

            Al final de la función, la población se ha reducido y se han seleccionado los candidatos para la cría de la próxima generación.

            Atributos:
                self.population (lista): La población actual de pájaros.
                self.birdsToBreed (lista): Los pájaros seleccionados para la cría.
                self.highscore (float): El mejor puntaje obtenido hasta ahora.
                self.highgen (int): La generación en la que se obtuvo el highscore.
                self.allTimeBestBird (Bird): El mejor pájaro de todos los tiempos.
                self.bestInputWeights (lista): Los pesos de entrada del mejor pájaro de todos los tiempos.
                self.bestHiddenWeights (lista): Los pesos ocultos del mejor pájaro de todos los tiempos.
                self.respawn (bool): Indicador para determinar si se reinicia la población.
                self.generation (int): El contador actual de generación.
        """
    
        # Pre-calcular la aptitud (fitness) de cada pájaro y almacenarla junto con su índice
        fitness_and_indices = [(i, bird.calculate_fitness()) for i, bird in enumerate(self.population)]

        # Ordenar por fitness y tomar los dos mejores
        sorted_by_fitness = sorted(fitness_and_indices, key=lambda x: x[1], reverse=True)[:2]

        for i, best_fitness in sorted_by_fitness:
            best_bird = self.population[i]

            if best_fitness >= self.highscore:
                self.highscore = best_fitness
                self.highgen = self.generation
                self.allTimeBestBird = best_bird
                self.bestInputWeights = copy.deepcopy(best_bird.model.inputWeights)
                self.bestHiddenWeights = copy.deepcopy(best_bird.model.hiddenWeights)

                print(f"highscore beaten {best_bird.model.inputWeights} - {best_bird.model.hiddenWeights} - Generation {self.generation}")

            self.birdsToBreed.append(best_bird)

        # Eliminar los pájaros seleccionados de la población
        for i, _ in sorted(sorted_by_fitness, reverse=True):
            self.population.pop(i)

        print(f"Best genes of this generation: {self.birdsToBreed[0].model.inputWeights}\n{self.birdsToBreed[0].model.hiddenWeights}")
        
        # Revisar si es necesario reiniciar la población
        print(f"GENERATION {self.generation} - HIGHGEN {self.highgen}")
        if (self.generation - self.highgen > 15):
            self.respawn = True

        self.generation += 1


    def breed(self):
        """
            Crea una nueva generación de pájaros basada en la actual piscina de cría.

            Esta función realiza los siguientes pasos:
            1. Conserva el mejor pájaro de la generación actual y el mejor de todos los tiempos sin mutación.
            2. Cría y muta a los dos mejores pájaros de la generación actual para un tercio de la nueva población.
            3. Cría y muta al mejor pájaro de la generación actual para otro tercio de la nueva población.
            4. Para el tercio final, cría al segundo mejor pájaro o reemplaza con nuevos pájaros basado en la bandera `respawn`.
            Si `respawn` es verdadero, indica genes malos y reemplaza un tercio de los pájaros con nuevos, reiniciando el contador de generaciones.
            
            La nueva población reemplaza a la población actual al final de la función.

            Atributos:
                BIRDS (int): Número total de pájaros en la población.
                third_of_birds (int): Un tercio del tamaño total de la población, utilizado para segmentos de cría.
                self.population (lista): La población actual de pájaros, que será reemplazada por la nueva generación.
                self.birdsToBreed (lista): Los mejores pájaros de la generación actual utilizados para la cría.
                self.bestInputWeights (lista): Los pesos de entrada del mejor pájaro de todos los tiempos.
                self.bestHiddenWeights (lista): Los pesos ocultos del mejor pájaro de todos los tiempos.
                self.respawn (bool): Indicador de si es necesario reemplazar genes malos en la población.
                self.generation (int): El conteo actual de generaciones, incrementado o reiniciado basado en `respawn`.
        """
        third_of_birds = int(BIRD_NUMBER / 3)
        
        self.population = []

        # Conservar el mejor pájaro de la generación y el mejor de todos los tiempos sin mutación
        best_of_generation = Bird(self.config)
        best_of_generation.model.set_weights(self.birdsToBreed[0].model.inputWeights,
                                            self.birdsToBreed[0].model.hiddenWeights)
        self.population.append(best_of_generation)

        best_of_all_time = Bird(self.config)
        best_of_all_time.model.set_weights(self.bestInputWeights, self.bestHiddenWeights)
        self.population.append(best_of_all_time)

        # Cría y mutación con los dos mejores pájaros de la generación
        for _ in range(third_of_birds):
            self.population.append(Bird(self.config, male=self.birdsToBreed[0].model,
                                        female=self.birdsToBreed[1].model))

        # Cría y mutación con el mejor pájaro de la generación
        for _ in range(third_of_birds):
            self.population.append(Bird(self.config, self.birdsToBreed[0].model))

        # Manejar los genes malos o cría con el segundo mejor pájaro
        for _ in range(third_of_birds):
            if self.respawn:
                self.population.append(Bird(self.config))
            else:
                self.population.append(Bird(self.config, male=self.birdsToBreed[1].model))

        if self.respawn:
            self.respawn = False
            self.generation = 0
            print(f"REPLACE BAD GENES - GENERATION {self.generation} {third_of_birds}")
            print("Due to natural selection - one third of birds receives new genes")
        else:
            self.generation += 1
        
    def get_new_generation_v2(self):
        self.birdsToBreed = []
        self.parents_select()
        self.breed()