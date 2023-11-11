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
                    10, activation="relu", input_shape=(5,)
                ),  # Hidden layer with 10 neurons, and input shape of 5 (number of observation variables)
                layers.Dense(
                    1, activation="sigmoid"
                ),  # Output layer with 1 neuron for binary decision
            ]
        )

        # Compile the model with the Adam optimizer and a loss function for binary outcomes(0 or 1 - jump or don't jump)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def get_model(self):
        """
            Returns the neural network model used by the AI agent.
        """
        return self.model
    
    def set_model_weights(self, weights):
        """
            Set the weights of the model.

            Args:
                weights (list): A list of numpy arrays representing the weights of the model.
        """
        self.model.set_weights(weights)
        
    def get_model_weights(self):
        """
            Returns the weights of the neural network model.
        """
        return self.model.get_weights()
        
    def save_model(self, filename):
        """
            Saves the weights of the model to a file.

            Args:
                filename (str): The name of the file to save the weights to.
        """
        self.model.save_weights(f"model_new_{filename}.keras", overwrite=True)
        
    def load_model(self, filename):
        self.model.load_weights(f"model_new_{filename}.keras")
    
    # Decides action based on game observation
    def predict(self, game_observation: GameObservation) -> GameAction:
        """
            Predicts the next action to take based on the current game observation.

            Args:
                game_observation (GameObservation): The current game observation.

            Returns:
                GameAction: The predicted action to take.
        """
        # Preprocess observation into the format the model expects (a batch of one observation)
        observation = game_observation.as_vector().reshape(
            1, -1
        )  # Reshape the observation to be a batch of one
        # Get the model's prediction
        action_probability = self.model.predict(observation)
        # Determine the action based on the probability
        action = GameAction.JUMP if action_probability > 0.5 else GameAction.DO_NOTHING
        
        return action
