import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class LinearModel:
    """ A linear regression model """

    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, X):
        # make sure X is N x D
        assert len(X.shape) == 2
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        # make sure X is N x D
        assert len(X.shape) == 2

        # the loss values are 2-D
        # normally we would divide by N only
        # but now we divide by N x K
        num_values = np.prod(Y.shape)

        # do one step of gradient descent
        # we multiply by 2 to get the exact gradient
        # (not adjusting the learning rate)
        # i.e. d/dx (x^2) --> 2x
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz["W"]
        self.b = npz["b"]

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


class DLModel:
    """ A deep learning model """

    def __init__(self, input_dim, n_action, learning_rate=0.01, momentum=0.9):
        self.model = tf.keras.Sequential(
            [layers.Dense(units=input_dim, input_shape=(input_dim,)), layers.Dense(units=n_action),]
        )

        self.model.compile(
            optimizer=tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
            loss='mean_squared_error')

    def predict(self, X):
        # make sure X is (N x state_size)
        assert len(X.shape) == 2
        action = self.model.predict(X)
        # make sure Y is (N x action_size)
        assert len(action.shape) == 2
        return action

    def sgd(self, X, Y):
        # make sure X is (N x state_size)
        assert len(X.shape) == 2
        # make sure Y is (N x action_size)
        assert len(Y.shape) == 2
        self.model.fit(X, Y, epochs=10)

    def load_weights(self, filepath):
        tf.keras.models.load_model(filepath)

    def save_weights(self, filepath):
        self.model.save(filepath)


class DQNAgent(object):
    def __init__(self, state_size, action_size, is_deep_learning:bool):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DLModel(state_size, action_size) if is_deep_learning else LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(
                self.model.predict(next_state), axis=1
            )

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
