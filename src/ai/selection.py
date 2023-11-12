import numpy as np

def roulette_wheel_selection(population, fitnesses):
    """
    Este método asigna a cada individuo una probabilidad de ser seleccionado que es proporcional a su fitness. Los individuos con mayor fitness tienen una mayor probabilidad de ser elegidos.
    """
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    return np.random.choice(population, size=2, replace=False, p=probabilities)


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    En la selección por torneo, se eligen al azar varios individuos del conjunto de la población y el mejor de este grupo es seleccionado como padre. Este proceso se repite hasta seleccionar el número deseado de padres.
    """
    selected_parents = []
    for _ in range(2):  # Elegir dos padres
        participants = np.random.choice(population, tournament_size, replace=False)
        participant_fitnesses = [fitnesses[population.index(p)] for p in participants]
        winner = participants[np.argmax(participant_fitnesses)]
        selected_parents.append(winner)
    return selected_parents


def rank_selection(population, fitnesses):
    """
    En la selección por rango, los individuos se ordenan según su fitness, y luego se asignan probabilidades de ser seleccionados basadas en este rango. Esto ayuda a evitar que los individuos con fitness muy alto dominen la selección.
    """
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
    ranks = np.arange(1, len(sorted_population) + 1)
    total_rank = sum(ranks)
    probabilities = [r / total_rank for r in ranks]
    return np.random.choice(sorted_population, size=2, replace=False, p=probabilities)


def truncation_selection(population, fitnesses, top_percentage=0.5):
    """
    En la selección por truncamiento, solo los individuos con los mejores fitnesses (por ejemplo, el mejor 50%) son elegidos como padres. Los padres se eligen al azar de este subconjunto.
    """
    cutoff = int(len(population) * top_percentage)
    sorted_population = sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)
    top_individuals = [x for _, x in sorted_population[:cutoff]]
    return np.random.choice(top_individuals, size=2, replace=False)

def natural_selection(self):
    # Clear the list
    self.mating_pool = []

    max_fitness = 0
    for i in range(len(self.population)):
        if self.population[i].fitness > max_fitness:
            max_fitness = self.population[i].fitness

    # Based on fitness, each member will get added to the mating pool a certain number of times
    # a higher fitness = more entries to mating pool = more likely to be picked as a parent
    # a lower fitness = fewer entries to mating pool = less likely
    
    

def create_mating_pool(population, fitnesses, pool_size):
    mating_pool = []
    max_fitness = max(fitnesses)
    
    for individual, fitness in zip(population, fitnesses):
        # Normalizar el fitness y usarlo para determinar el número de veces que cada individuo
        # aparece en el mating pool
        num_entries = int((fitness / max_fitness) * pool_size)
        mating_pool.extend([individual] * num_entries)
    
    return mating_pool

def select_parents_from_mating_pool(mating_pool, num_parents=2):
    return np.random.choice(mating_pool, size=num_parents, replace=False)

