import numpy as np

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    # predict on generator, return predictions and labels
    def predict_on_generator(self, generator):
        preds = []
        labels = []

        for i in range(len(generator)):
            x, y = generator[i]
            p = self.model.predict(x, batch_size=x.shape[0])
            preds.append(p)
            labels.append(y)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        return preds, labels

    def build_model(self):
        raise NotImplementedError