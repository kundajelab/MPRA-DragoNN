import numpy as np
from keras.callbacks import Callback
from scipy.stats import spearmanr, pearsonr
from utils.helper import predict_on_generator


def pearsonr_metric(y_true, y_pred):
    try:
        taskwise_pearsonr = [pearsonr(y_true[:, i], y_pred[:, i])[
            0] for i in range(y_pred.shape[1])]
        return taskwise_pearsonr, np.mean(taskwise_pearsonr)
    except:
        return pearsonr(y_true,y_pred)

def spearmanr_metric(y_true, y_pred):
    try:
        taskwise_spearmanr = [spearmanr(y_true[:, i], y_pred[:, i])[
            0] for i in range(y_pred.shape[1])]
        return taskwise_spearmanr, np.mean(taskwise_spearmanr)
    except:
        return spearmanr(y_true,y_pred)


class CorrelationMetrics(Callback):
    def __init__(self, valid_data_loader):
        self.valid_data_loader = valid_data_loader

    def on_epoch_end(self, epoch, logs=None):
        preds, labels = predict_on_generator(self.model, self.valid_data_loader)

        taskwise_pearsonr, mean_pearsonr = pearsonr_metric(labels, preds)
        taskwise_spearmanr, mean_spearmanr = spearmanr_metric(labels, preds)

        print('Mean Pearson Correlation  : {:.4f}'.format(mean_pearsonr))
        print('Mean Spearman Correlation : {:.4f}'.format(mean_spearmanr))

        logs['val_mean_pearsonr'] = mean_pearsonr
        logs['val_mean_spearmanr'] = mean_spearmanr
        for i in range(len(taskwise_pearsonr)):
            logs['val_task_{}_pearsonr'.format(i)] = taskwise_pearsonr[i]
            logs['val_task_{}_spearmanr'.format(i)] = taskwise_spearmanr[i]
