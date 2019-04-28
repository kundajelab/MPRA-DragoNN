from models.base_model import BaseModel

from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf

class DeepFactorizedModel(BaseModel):
    def __init__(self, config):
        super(DeepFactorizedModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()

        # sublayer 1
        self.model.add(Conv1D(48, 3, padding='same', activation='relu', input_shape=(self.config.input_sequence_length, 4)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Conv1D(64, 3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Conv1D(100, 3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Conv1D(150, 7, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Conv1D(300, 7, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(MaxPooling1D(3))

        # sublayer 2
        self.model.add(Conv1D(200, 7, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Conv1D(200, 3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Conv1D(200, 3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(MaxPooling1D(4))

        # sublayer 3
        self.model.add(Conv1D(200, 7, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(MaxPooling1D(4))

        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))
        self.model.add(Dense(self.config.number_of_outputs, activation='linear'))

        self.model.compile(
            loss= "mean_squared_error",
            optimizer=self.config.optimizer,
            # custom metrics in trainer
        )

