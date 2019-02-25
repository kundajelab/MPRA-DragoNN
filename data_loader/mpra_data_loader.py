from keras.utils import Sequence
import h5py
import os


class MPRADataLoader(Sequence):
    def __init__(self, config, datatype):
        self.config = config
        self.batch_size = config.trainer.batch_size
        self.data = h5py.File(os.path.join(
            config.data_path, datatype + '.hdf5'))
        self.inputs = self.data['X']['sequence']
        self.labels = self.data['Y']['output']

    def __len__(self):
        return 10 #self.inputs.shape[0]//self.batch_size

    def __getitem__(self, idx):
        return self.inputs[idx:idx+self.batch_size], self.labels[idx:idx+self.batch_size]
    
