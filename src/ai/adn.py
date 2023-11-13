import numpy as np
import math
from enum import Enum

class GameAction(Enum):
    JUMP = 0
    DO_NOTHING = 1

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size=5, hidden_size=3, output_size=1):
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # Inicialización de los pesos y bias
        # self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        # self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        
        self.W1 = np.random.normal(0, scale=0.1, size=(input_size, hidden_size))
        self.W2 = np.random.normal(0, scale=0.1, size=(hidden_size, output_size))
        
        
        # self.bias1 = np.random.uniform(-1, 1, self.hiddenSize)
        # self.bias2 = np.random.uniform(-1, 1, self.outputSize)

    def get_weights(self):
        return self.W1, self.W2
    
    def set_weights(self, w1, w2):
        self.W1 = w1
        self.W2 = w2
        
    def forward(self, inputs):
        
        # Propagación hacia adelante
        #self.hidden = sigmoid(np.dot(inputs, self.W1))  # Activación de la capa oculta
        
        #self.hidden = sigmoid(np.dot(inputs, self.W1))
        #output = sigmoid(np.dot(self.hidden, self.W2))  # Activación de la capa de salida
        #return output
        
        hidden_layer_in = np.dot(inputs, self.W1)
        hidden_layer_out = sigmoid(hidden_layer_in)
        output_layer_in = np.dot(hidden_layer_out, self.W2)
        prediction = sigmoid(output_layer_in)
        return prediction

    def predict(self, game_observation):
        BIAS = -0.5
        
        observation = game_observation.as_vector().reshape(1, -1)
        action_probability = self.forward(observation)
        action = GameAction.JUMP if (action_probability+BIAS > 0) else GameAction.DO_NOTHING
        
        print(f"Action probability: {action_probability} - Action: {action}")
        return action
