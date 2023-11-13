import random
import numpy as np
import math
from enum import Enum

class GameAction(Enum):
    JUMP = 0
    DO_NOTHING = 1

class NeuralNetwork:
    def __init__(self, input_size=5, hidden_size=3, output_size=1, male = None, female = None):
        """
            Inicializa una nueva instancia de la red neuronal.

            Si se proporcionan objetos 'male' y 'female', se utiliza su información genética
            para inicializar los pesos de la red. En caso contrario, los pesos se inicializan
            aleatoriamente. Si solo se proporciona 'male', se utiliza su información genética
            con mutación.

            Parámetros:
                input_size (int): Tamaño de la capa de entrada.
                hidden_size (int): Tamaño de la capa oculta.
                output_size (int): Tamaño de la capa de salida.
                male (NeuralNetwork, opcional): Red neuronal del 'padre'.
                female (NeuralNetwork, opcional): Red neuronal de la 'madre'.
        """

        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        #self.inputWeights = np.random.normal(0, scale=0.1, size=(input_size, hidden_size))
        m1 = np.array([[-0.49228218, -0.18985529, -0.85399669],
                    [ 1.29168001, -0.87615453, -1.22169777],
                    [-1.89356504, -2.98874179,  3.32321219],
                    [ 0.27552593, -1.57704918,  1.91817933],
                    [-2.01281056,  2.24059056, -1.50034911]])    
        m2 = np.array([[ 3.8136693 ],
            [ 0.92230492],
            [-0.01239667],])

        if (male == None): #New Bird, no parents
            #easy network
            self.inputWeights = m1
            self.hiddenWeights = m2
        elif (female == None): #Only one Parent (self mutate)
            self.inputWeights = male.inputWeights
            self.hiddenWeights = male.hiddenWeights
            self.mutate()
        else: # Two parents - Breed.
            self.inputWeights = np.random.normal(0, scale=0.1, size=(5, 3))
            self.hiddenWeights = np.random.normal(0, scale=0.1, size=(3, 1))
            self.breed(male, female)
        
    def get_weights(self):
        """
            Devuelve los pesos actuales de la red neuronal.

            Retorna:
                Tuple: Tupla conteniendo los pesos de la capa de entrada y oculta.
        """
        return self.inputWeights, self.hiddenWeights
    
    def set_weights(self, w1, w2):
        """
            Establece los pesos de la red neuronal.

            Parámetros:
                w1 (array): Pesos de la capa de entrada.
                w2 (array): Pesos de la capa oculta.
        """
        self.inputWeights = w1
        self.hiddenWeights = w2
        
    def forward(self, inputs):
        """
            Realiza la propagación hacia adelante (forward pass) de la red neuronal.

            Parámetros:
                inputs (array): Entrada a la red.

            Retorna:
                El resultado de la propagación hacia adelante.
        """
        
        hidden_layer_in = np.dot(inputs, self.inputWeights)
        hidden_layer_out = self.sigmoid(hidden_layer_in)
        
        output_layer_in = np.dot(hidden_layer_out, self.hiddenWeights)
        prediction = self.sigmoid(output_layer_in)

        return prediction

    def predict(self, game_observation):
        """
            Predice la acción a realizar basada en una observación del juego.

            Parámetros:
                game_observation: Observación del estado actual del juego.

            Retorna:
                GameAction: La acción a realizar (salto o no hacer nada).
        """
        BIAS = -0.5
        
        observation = game_observation.as_vector().reshape(1, -1)
        action_probability = self.forward(observation)
        action = GameAction.JUMP if (action_probability+BIAS > 0) else GameAction.DO_NOTHING
        
        # print(f"Action probability: {action_probability} - Action: {action}")
        return action

    def relu(self, x):
        """
        Función de activación ReLU.

        Parámetros:
            x (array): Valor(es) de entrada.

        Retorna:
            El resultado de aplicar la función ReLU.
        """
        return np.maximum(x, 0)

    def sigmoid(self, x):
        """
        Función de activación sigmoide.

        Parámetros:
            x (array): Valor(es) de entrada.

        Retorna:
            El resultado de aplicar la función sigmoide.
        """
        return 1 / (1 + np.exp(-x))

    def breed(self, male, female):
        """
        Genera una nueva red neuronal a partir de dos 'padres', promediando sus pesos
        y aplicando mutación posteriormente.

        Parámetros:
            male (NeuralNetwork): Red neuronal del 'padre'.
            female (NeuralNetwork): Red neuronal de la 'madre'.
        """
        for i in range(len(self.inputWeights)):
            self.inputWeights[i] = (male.inputWeights[i] + female.inputWeights[i]) / 2
        for i in range(len(self.hiddenWeights)):
            self.hiddenWeights[i] = (male.hiddenWeights[i] + female.hiddenWeights[i]) / 2
        self.mutate()
  
    def mutate(self):
        """
            Aplica mutaciones aleatorias a los pesos de la red neuronal.
        """
        for i in range(len(self.inputWeights)):
            for j in range(len(self.inputWeights[i])):
                self.inputWeights[i][j] = self.getMutatedGene(self.inputWeights[i][j])
        for i in range(len(self.hiddenWeights)):
            for j in range(len(self.hiddenWeights[i])):
                self.hiddenWeights[i][j] = self.getMutatedGene(self.hiddenWeights[i][j])


    def getMutatedGene(self, weight):
        """
            Aplica una mutación a un peso específico de la red.

            Parámetros:
                weight (float): Peso a mutar.

            Retorna:
                float: El peso mutado.
        """
        multiplier = 0
        learning_rate = random.randint(0, 25) * 0.015
        randBool = bool(random.getrandbits(1)) #adapt upwards or downwards?
        randBool2 = bool(random.getrandbits(1)) #or not at all?
        
        if (randBool and randBool2):
            multiplier = 1
        elif (not randBool and randBool2):
            multiplier = -1
        
        mutatedWeight = weight + learning_rate*multiplier
        
        return mutatedWeight