from models.base_model import BaseModel

from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf

class ConvModel(BaseModel):
    def __init__(self, config):
        super(ConvModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()

        self.model.add(Conv1D(120, 5, activation='relu', input_shape=(self.config.input_sequence_length, 4)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Conv1D(120, 5, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Conv1D(120, 5, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))

        self.model.add(Flatten())
        self.model.add(Dense(self.config.number_of_outputs, activation='linear'))

        self.model.compile(
            loss= "mean_squared_error",
            optimizer=self.config.optimizer,
            # custom metrics in trainer
        )

