import random
import numpy as np
import math
from enum import Enum

class GameAction(Enum):
    JUMP = 0
    DO_NOTHING = 1

class NeuralNetwork:
    def __init__(self, input_size=5, hidden_size=3, output_size=1, male = None, female = None):
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        #self.inputWeights = np.random.normal(0, scale=0.1, size=(input_size, hidden_size))
        #self.hiddenWeights = np.random.normal(0, scale=0.1, size=(hidden_size, output_size))
        
        if (male == None): #New Bird, no parents
            #easy network
            self.inputWeights = np.random.normal(0, scale=0.1, size=(5, 3))
            self.hiddenWeights = np.random.normal(0, scale=0.1, size=(3, 1))
        elif (female == None): #Only one Parent (self mutate)
            self.inputWeights = male.inputWeights
            self.hiddenWeights = male.hiddenWeights
            self.mutate()
        else: # Two parents - Breed.
            self.inputWeights = np.random.normal(0, scale=0.1, size=(5, 3))
            self.hiddenWeights = np.random.normal(0, scale=0.1, size=(3, 1))
            self.breed(male, female)
        
    def get_weights(self):
        return self.inputWeights, self.hiddenWeights
    
    def set_weights(self, w1, w2):
        self.inputWeights = w1
        self.hiddenWeights = w2
        
    def forward(self, inputs):
        
        # Propagación hacia adelante
        #self.hidden = sigmoid(np.dot(inputs, self.inputWeights))  # Activación de la capa oculta
        
        #self.hidden = sigmoid(np.dot(inputs, self.inputWeights))
        #output = sigmoid(np.dot(self.hidden, self.hiddenWeights))  # Activación de la capa de salida
        #return output
        
        hidden_layer_in = np.dot(inputs, self.inputWeights)
        hidden_layer_out = self.sigmoid(hidden_layer_in)
        
        output_layer_in = np.dot(hidden_layer_out, self.hiddenWeights)
        prediction = self.sigmoid(output_layer_in)
        return prediction

    def predict(self, game_observation):
        BIAS = -0.5
        
        observation = game_observation.as_vector().reshape(1, -1)
        action_probability = self.forward(observation)
        action = GameAction.JUMP if (action_probability+BIAS > 0) else GameAction.DO_NOTHING
        
        # print(f"Action probability: {action_probability} - Action: {action}")
        return action

    def relu(self, x):
        """The relu actication function for the neural network
        INPUT: x - The value to apply the ReLu function on
        OUTPUT: The applied ReLus function value"""
        return np.maximum(x, 0)

    def sigmoid(self, x):
        """The sigmoid activation function for the neural net   
        INPUT: x - The value to calculate
        OUTPUT: The calculated result"""    
        return 1 / (1 + np.exp(-x))

    def breed(self, male, female):
        """Generate a new brain (neural network) from two parent birds
         	by averaging their brains and mutating them afterwards
        INPUT:  male - The male bird object (of class bird)
        		female - The female bird object (of class bird)
        OUTPUT:	None"""
        for i in range(len(self.inputWeights)):
            self.inputWeights[i] = (male.inputWeights[i] + female.inputWeights[i]) / 2
        for i in range(len(self.hiddenWeights)):
            self.hiddenWeights[i] = (male.hiddenWeights[i] + female.hiddenWeights[i]) / 2
        self.mutate()
  
    def mutate(self):
        """mutate (randomly apply the learning rate) the birds brain
        neural network) randomly changing the individual weights
        INPUT:  None
        OUTPUT:	None"""
        for i in range(len(self.inputWeights)):
            for j in range(len(self.inputWeights[i])):
                self.inputWeights[i][j] = self.getMutatedGene(self.inputWeights[i][j])
        for i in range(len(self.hiddenWeights)):
            for j in range(len(self.hiddenWeights[i])):
                self.hiddenWeights[i][j] = self.getMutatedGene(self.hiddenWeights[i][j])


    def getMutatedGene(self, weight):
        """mutate the input by -0.125 to 0.125 or not at all
        INPUT: weight - The weight to mutate
        OUTPUT: mutatedWeight - The mutated weight
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