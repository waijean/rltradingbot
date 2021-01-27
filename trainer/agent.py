import numpy as np
import pandas as pd

class LinearModel:
    """ A linear regression model """

    def __init__(self, input_dim, n_action, mmt, lr):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.learning_rate = lr
        self.momentum = mmt

        self.losses = []

    def predict(self, X):
        # make sure X is N x D
        assert len(X.shape) == 2
        return X.dot(self.W) + self.b

    def sgd(self, X, Y):
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
        self.vW = self.momentum * self.vW - self.learning_rate * gW
        self.vb = self.momentum * self.vb - self.learning_rate * gb

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
        #np.savez(filepath, W=self.W, b=self.b)
        savefile = pd.DataFrame({'w':self.W, 'b':self.b})
        savefile.to_csv(filepath)


class DQNAgent(object):
    def __init__(self, state_size, action_size, gamma, epsilon_decay, momentum, learnrate):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.model = LinearModel(state_size, action_size, momentum, learnrate)

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
