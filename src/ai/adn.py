from enum import Enum
import numpy as np

class GameAction(Enum):
    JUMP = 0
    DO_NOTHING = 1
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self):
        # Definimos el número de neuronas en cada capa
        self.inputSize = 5  # Número de entradas: bird_y_pos, y_dist_to_bot_pipe, y_dist_to_top_pipe, x_dist_to_pipe_pair, bird_y_vel
        self.outputSize = 1  # Salida: decisión de saltar o no saltar
        self.hiddenSize = 10  # Puedes ajustar el tamaño de la capa oculta

        # Inicialización de los pesos
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # pesos entre la capa de entrada y la capa oculta
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # pesos entre la capa oculta y la capa de salida

    def forward(self, inputs):
        # Propagación hacia adelante
        self.hidden = sigmoid(np.dot(inputs, self.W1))  # Activación de la capa oculta
        output = sigmoid(np.dot(self.hidden, self.W2))  # Activación de la capa de salida
        return output

    def predict(self):
        # Predice la salida
        action_probability = self.forward(self.inputs)
        action = GameAction.JUMP if action_probability > 0.5 else GameAction.DO_NOTHING
        return action
