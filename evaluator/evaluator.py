import os
from utils.metrics import pearsonr_metric, spearmanr_metric
from utils.helper import predict_on_generator

class Evaluator():
    def __init__(self, model, test_data_loader, config):
        self.config = config
        self.model = model
        self.test_data_loader = test_data_loader

    def evaluate(self):
        preds, labels = predict_on_generator(self.model, self.test_data_loader)

        taskwise_pearsonr, mean_pearsonr = pearsonr_metric(labels, preds)
        taskwise_spearmanr, mean_spearmanr = spearmanr_metric(labels, preds)

        print('Mean Pearson Correlation  : {:.4f}'.format(mean_pearsonr))
        print('Mean Spearman Correlation : {:.4f}'.format(mean_spearmanr))

        print('Taskwise Pearson Correlation  : {}'.format(taskwise_pearsonr))
        print('Taskwise Spearman Correlation : {}'.format(taskwise_spearmanr))
