import numpy as np

def model_crossover(model_idx1, model_idx2):
    global current_pool
    weights1 = current_pool[model_idx1].get_weights()
    weights2 = current_pool[model_idx2].get_weights()
    weightsnew1 = weights1
    weightsnew2 = weights2
    weightsnew1[0] = weights2[0]
    weightsnew2[0] = weights1[0]
    return np.asarray([weightsnew1, weightsnew2])


def single_point_crossover(parent1, parent2):
    """
    Este método selecciona un punto al azar en la representación del genoma y cruza las secciones de los padres.

    """
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2


def two_point_crossover(parent1, parent2):
    """
    Similar al crossover de un punto, pero con dos puntos de corte.
    """
    crossover_point1 = np.random.randint(1, len(parent1) - 1)
    crossover_point2 = np.random.randint(crossover_point1 + 1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]])
    child2 = np.concatenate([parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]])
    return child1, child2


def uniform_crossover(parent1, parent2):
    """
    En este método, cada gen se elige al azar de uno de los padres.
    """
    mask = np.random.randint(0, 2, size=len(parent1)).astype(np.bool)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def arithmetic_crossover(parent1, parent2, alpha=0.5):
    """
    Este método es más común en problemas de optimización real y combina los genes de los padres mediante operaciones aritméticas.
    """
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2
