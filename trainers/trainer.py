import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from utils.metrics import CorrelationMetrics
import pdb

class Trainer():
    def __init__(self, model, train_data_loader, valid_data_loader, config):
        self.config = config
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        
        self.callbacks = []
        self.loss = []        
        self.val_loss = []        
        
        self.init_callbacks()

    def init_callbacks(self):
        # add pearson and spearman correlation metrics
        self.callbacks.append(CorrelationMetrics(self.valid_data_loader))

        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.experiment_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,  
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose, 
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.tensorboard_log_dir,
                write_graph=self.config.tensorboard_write_graph,
            )
        )
        
    
    def train(self):
        history = self.model.fit_generator(
            generator=self.train_data_loader,
            validation_data=self.valid_data_loader,
            epochs=self.config.num_epochs,
            verbose=self.config.verbose_training,
            callbacks=self.callbacks,
            use_multiprocessing=True,
            workers=self.config.multiprocessing_workers
        )
        
        self.loss.extend(history.history['loss'])        
        self.val_loss.extend(history.history['val_loss'])
        
