import numpy as np

def mutate_weights(weights, mutation_rate=0.1):
    """
    Aplica una mutación a un conjunto de pesos de la red neuronal.
    :param weights: Conjunto de pesos a mutar (ndarray).
    :param mutation_rate: La tasa de mutación, que determina la magnitud de los cambios.
    :return: Conjunto de pesos mutados.
    """
    mutation_mask = np.random.rand(*weights.shape) < mutation_rate
    mutation = np.random.normal(size=weights.shape)
    new_weights = weights + mutation_mask * mutation
    return new_weights

def mutate_weights_v2(weights, mutation_rate=0.1):
    mutation_mask = np.random.uniform(0, 1, weights.shape) < 0.10
    changes = np.random.uniform(-0.125, 0.125, weights.shape)
    return weights + mutation_mask * changes
    

def model_mutate(weights):
    for xi in range(len(weights)):
        for yi in range(len(weights[xi])):
            if np.random.uniform(0, 1) > 0.85:
                change = np.random.uniform(-0.125,0.125)
                weights[xi][yi] += change
    return weights

def gaussian_mutation(weights, mutation_rate=0.15, sigma=0.5):
    for i in range(len(weights)):
        if np.random.rand() < mutation_rate:
            weights[i] += np.random.normal(0, sigma)
    return weights


def gaussian_weight_mutation(network, mutation_rate, scale=0.1):
    """
    Esta técnica es comúnmente usada para mutar los pesos de una red neuronal. Consiste en añadir un pequeño cambio a cada peso, donde el cambio sigue una distribución gaussiana (normal).

    """
    for layer in network.layers:
        weights, biases = layer.get_weights()

        # Mutar los pesos
        if np.random.rand() < mutation_rate:
            weights += np.random.normal(0, scale, size=weights.shape)

        # Mutar los sesgos (opcional)
        if np.random.rand() < mutation_rate:
            biases += np.random.normal(0, scale, size=biases.shape)

        layer.set_weights([weights, biases])


def uniform_weight_mutation(network, mutation_rate, scale=0.1):
    """
    En esta técnica, cada peso se modifica sumándole un valor pequeño y aleatorio, seleccionado de una distribución uniforme.
    """
    for layer in network.layers:
        weights, biases = layer.get_weights()

        # Mutar los pesos
        if np.random.rand() < mutation_rate:
            weights += np.random.uniform(-scale, scale, size=weights.shape)

        # Mutar los sesgos (opcional)
        if np.random.rand() < mutation_rate:
            biases += np.random.uniform(-scale, scale, size=biases.shape)

        layer.set_weights([weights, biases])


def scale_weight_mutation(network, mutation_rate, scale_factor_range=(0.9, 1.1)):
    """
    Modifica los pesos multiplicándolos por un factor aleatorio cercano a 1. Este método es útil para realizar cambios proporcionales en la magnitud de los pesos.
    """
    for layer in network.layers:
        weights, biases = layer.get_weights()

        # Mutar los pesos
        if np.random.rand() < mutation_rate:
            scale_factor = np.random.uniform(*scale_factor_range)
            weights *= scale_factor

        # Mutar los sesgos (opcional)
        if np.random.rand() < mutation_rate:
            scale_factor = np.random.uniform(*scale_factor_range)
            biases *= scale_factor

        layer.set_weights([weights, biases])
