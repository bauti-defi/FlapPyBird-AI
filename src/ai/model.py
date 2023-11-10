from enum import Enum

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        print(self.model.get_weights())

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
        
        #print(f"Action: {action.name}, Probability: {action_probability}")
        
        return action
