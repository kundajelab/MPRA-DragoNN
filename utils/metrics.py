import numpy as np
from keras.callbacks import Callback
from scipy.stats import spearmanr, pearsonr
import pdb


class CorrelationMetrics(Callback):
    def __init__(self, valid_data_loader):
        self.valid_data_loader = valid_data_loader

    def on_epoch_end(self, epoch, logs=None):
        preds = []
        labels = []

        for i in range(len(self.valid_data_loader)):
            x, y = self.valid_data_loader[i]
            p = self.model.predict(x, batch_size=x.shape[0])
            preds.append(p)
            labels.append(y)
        
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        taskwise_pearsonr = [pearsonr(labels[:,i], preds[:,i])[0] for i in range(preds.shape[1])]
        taskwise_spearmanr = [spearmanr(labels[:,i], preds[:,i])[0] for i in range(preds.shape[1])]

        mean_pearsonr = np.mean(taskwise_pearsonr)
        mean_spearmanr = np.mean(taskwise_spearmanr)

        print('Mean Pearson Correlation  : {:.4f}'.format(mean_pearsonr))
        print('Mean Spearman Correlation : {:.4f}'.format(mean_spearmanr))

        logs['val_mean_pearsonr']  = mean_pearsonr
        logs['val_mean_spearmanr'] = mean_spearmanr
        for i in range(len(taskwise_pearsonr)):
            logs['val_task_{}_pearsonr'.format(i)] = taskwise_pearsonr[i]
            logs['val_task_{}_spearmanr'.format(i)] = taskwise_spearmanr[i]
        