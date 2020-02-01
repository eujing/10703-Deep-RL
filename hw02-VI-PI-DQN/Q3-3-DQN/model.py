import keras.layers as layers
import keras.models as models
import gym

env = gym.make("CartPole-v0")

if __name__ == "__main__":
    hidden_units = 10

    # TODO: Determine from env
    input_shape = env.observation_space.shape
    output_dim = env.action_space.n

    model = models.Sequential([
        layers.Dense(hidden_units, activation="relu",
                     input_shape=input_shape),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dense(output_dim, activation="relu"),
        ])
    model.summary()
