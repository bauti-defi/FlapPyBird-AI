import numpy as np

def crossover(parent1, parent2):
    """
    Realiza el cruce de un punto en los pesos de dos redes neuronales.
    :param parent1: El primer modelo de red neuronal (padre).
    :param parent2: El segundo modelo de red neuronal (padre).
    :return: Dos nuevos conjuntos de pesos (para dos hijos).
    """

    # Obtener los pesos de los padres
    W1_parent1, W2_parent1 = parent1.model.get_weights()
    W1_parent2, W2_parent2 = parent2.model.get_weights()

    # Elegir un punto de cruce aleatorio para cada conjunto de pesos
    crossover_point_W1 = np.random.randint(1, W1_parent1.shape[0])
    crossover_point_W2 = np.random.randint(1, W2_parent1.shape[0])

    # Realizar el cruce de un punto para W1
    new_W1_child1 = np.concatenate((W1_parent1[:crossover_point_W1, :], W1_parent2[crossover_point_W1:, :]), axis=0)
    new_W1_child2 = np.concatenate((W1_parent2[:crossover_point_W1, :], W1_parent1[crossover_point_W1:, :]), axis=0)

    # Realizar el cruce de un punto para W2
    new_W2_child1 = np.concatenate((W2_parent1[:crossover_point_W2, :], W2_parent2[crossover_point_W2:, :]), axis=0)
    new_W2_child2 = np.concatenate((W2_parent2[:crossover_point_W2, :], W2_parent1[crossover_point_W2:, :]), axis=0)

    return (new_W1_child1, new_W2_child1), (new_W1_child2, new_W2_child2)
    
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
