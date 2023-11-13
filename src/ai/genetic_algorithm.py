import copy
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

        self.allTimeBestBird = None
        
        self.highscore = 0
        self.generation = 1
        self.highgen = 0

        self.maxscore = 0
        # self.score = 0
        
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
        
        
        
    def get_new_generation_v2(self):
        self.birdsToBreed = []
        
        # SELECTION
        for h in range(2): #Best two birds are taken
            bestBird = -1
            bestFitness = -10
            for i in range(len(self.population)): #Find the best bird
                bird = self.population[i]
                bird.calculate_fitness()
                if (bird.fitness > bestFitness):
                    bestFitness = bird.fitness
                    bestBird = i
                    if (bestFitness >= self.highscore):
                        self.highscore = bestFitness
            if (bestFitness >= self.highscore):
                #new highscore! Let's keep the bird and update our scores
                self.allTimeBestBird = self.population[bestBird]
                self.bestInputWeights =  copy.deepcopy(
                                        self.population[bestBird].model.inputWeights)
                self.bestHiddenWeights =  copy.deepcopy(
                                        self.population[bestBird].model.hiddenWeights)
                print("highscore beaten {}\n{} - Generation {}"
                                        .format(bird.model.inputWeights,
                                        bird.model.hiddenWeights, self.generation))
                self.highscore = bestFitness
                self.highgen = self.generation

            self.birdsToBreed.append(self.population[bestBird])
            self.population.pop(i)

        print("Best genes of this generation: {}\n{}".format(self.birdsToBreed[0].model.inputWeights, self.birdsToBreed[0].model.hiddenWeights))
    
        #If no progress was made in the last 50 generations - new genes.
        print(f"GENERATION {self.generation} - HIGHGEN {self.highgen}")
        if (self.generation-self.highgen > 25):
            self.respawn = True

        self.generation += 1
        self.breed()
        
        
    def breed(self):
        #Atleast one death happened
        multiPlayer = []
        BIRDS = 12
        #keep the best bird of generation without mutation
        _ = Bird(self.config)
        _.model.set_weights(self.birdsToBreed[0].model.inputWeights,
                    self.birdsToBreed[0].model.hiddenWeights)
        #print(f"BIRD 1: f{_.model.get_weights()}")
        multiPlayer.append(_)

        #also keep the best of all time alive without mutation
        _ = Bird(self.config)
        _.model.set_weights(self.bestInputWeights,
                    self.bestHiddenWeights)
        #print(f"BIRD 2: f{_.model.get_weights()}")
        multiPlayer.append(_)

        for _ in range(int(BIRDS/3)):
            #Breed and mutate the two generations best birds sometimes
            multiPlayer.append(Bird(self.config, male=self.birdsToBreed[0].model,
                                            female=self.birdsToBreed[1].model))
        for _ in range(int(BIRDS/3)):
            #Breed and mutate the generations best bird a couple of times
            multiPlayer.append(Bird(self.config, self.birdsToBreed[0].model))

        for _ in range(int(BIRDS/3)):
            if (self.respawn): #Bad genes - replace some.
                print(f"REPLACE BAD GENES - GENERATION {self.generation} {int(BIRDS/3)}")
                multiPlayer.append(Bird(self.config))
                # self.generation=0
            else:
                #Breed and mutate the generations second best bird asometimes
                multiPlayer.append(Bird(self.config, male=self.birdsToBreed[1].model))

        # if (ReplayBest): #Used to replay a very good bird
        #     multiPlayer[2].set_weights(startInputGenes, startHiddenGenes)

        if (self.respawn):
            self.respawn = False
            print("Due to natural selection - one third of birds " +
                    "receives new genes")
            
        self.population = multiPlayer