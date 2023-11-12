import numpy as np


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
