from keras.utils import Sequence
import h5py
import os


class MPRADataLoader(Sequence):
    def __init__(self, config, datatype):
        self.config = config
        self.batch_size = config.batch_size
        self.fname = os.path.join(config.data_path, datatype + '.hdf5')

        with h5py.File(self.fname, 'r') as hf:
            self.max_batches = hf['X']['sequence'].shape[0]//self.batch_size
        
        if self.config.max_batch_steps != -1:
            self.max_batches = min(self.config.max_batch_steps, self.max_batches)

    def __len__(self):
        return self.max_batches

    def __getitem__(self, idx):
        start_idx = self.batch_size*idx
        end_idx = start_idx + self.batch_size
        with h5py.File(self.fname, 'r') as hf:
            x, y = hf['X']['sequence'][start_idx:end_idx], hf['Y']['output'][start_idx:end_idx]
       
        return x,y 
    
