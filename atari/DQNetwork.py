import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


class DQNetwork:
    def __init__(self, actions, input_shape,
                 minibatch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.99,
                 dropout_prob=0.1,
                 load_path=None,
                 logger=None):

        # Parameters
        self.actions = actions  # Size of the network output
        self.discount_factor = discount_factor  # Discount factor of the MDP
        self.minibatch_size = minibatch_size  # Size of the training batches
        self.learning_rate = learning_rate  # Learning rate
        self.dropout_prob = dropout_prob  # Probability of dropout
        self.logger = logger
        self.training_history_csv = 'training_history.csv'

        if self.logger is not None:
            self.logger.to_csv(self.training_history_csv, 'Loss,Accuracy')

        self.model = Sequential()


        self.model.add(Conv2D(32, 8, strides=(4, 4),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        self.model.add(Conv2D(64, 4, strides=(2, 2),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        self.model.add(Conv2D(64, 3, strides=(1, 1),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        # Flatten the convolution output
        self.model.add(Flatten())


        self.model.add(Dense(512, activation='relu'))

        # Output layer
        self.model.add(Dense(self.actions, activation='softmax'))
        print(self.model.summary())

        # Load the network weights from saved model
        if load_path is not None:
            self.load(load_path)

        self.model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self, batch, DQN_target):
        x_train = []
        t_train = []


        for datapoint in batch:

            x_train.append(datapoint['source'].astype(np.float64))

            next_state = datapoint['dest'].astype(np.float64)
            next_state_pred = DQN_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            t = list(self.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + self.discount_factor * next_q_value
            t_train.append(t)

        # Prepare inputs and targets
        x_train = np.asarray(x_train).squeeze()
        t_train = np.asarray(t_train).squeeze()

        # Train the model for one epoch
        h = self.model.fit(x_train, t_train, batch_size=self.minibatch_size, epochs=1)


        if self.logger is not None:
            self.logger.to_csv(self.training_history_csv, [h.history['loss'][0], h.history['accuracy'][0]])

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None, append=''):
        f = ('model%s.h5' % append) if filename is None else filename
        if self.logger is not None:
            self.logger.log('Saving model as %s' % f)
        self.model.save_weights(self.logger.path + f)

    def load(self, path):
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.model.load_weights(path)