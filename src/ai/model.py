from enum import Enum

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD

from .game_observation import GameObservation


class GameAction(Enum):
    JUMP = 0
    DO_NOTHING = 1

class Model:
    def __init__(self) -> None:
        # Create a simple sequential model
        self.model = Sequential(
            [
                layers.Dense(
                    10, activation="sigmoid", input_shape=(5,)
                ),  # Hidden layer with 10 neurons, and input shape of 5 (number of observation variables)
                layers.Dense(
                    1, activation="sigmoid"
                ),  # Output layer with 1 neuron for binary decision
            ]
        )

        # Compile the model with the Adam optimizer and a loss function for binary outcomes(0 or 1 - jump or don't jump)
        self.model.compile(
            optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
            loss="mse",
            metrics=["accuracy"],
        )

    def get_model(self):
        return self.model
    
    def set_model_weights(self, weights):
        self.model.set_weights(weights)
        
    def get_model_weights(self):
        self.model.get_weights()
        
    def save_model(self, filename):
        self.model.save_weights("Current_Model_Pool/model_new" + ".keras")
        
    def load_model(self, filename):
        self.model.load_weights("Current_Model_Pool/model_new" + ".keras")
    

    
    # Decides action based on game observation
    def predict(self, game_observation: GameObservation) -> GameAction:
        # Preprocess observation into the format the model expects (a batch of one observation)
        observation = game_observation.as_vector().reshape(
            1, -1
        )  # Reshape the observation to be a batch of one
        # Get the model's prediction
        action_probability = self.model.predict(observation)
        # Determine the action based on the probability
        action = GameAction.JUMP if action_probability > 0.5 else GameAction.DO_NOTHING
        
        return action
