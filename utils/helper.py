import numpy as np

# predict on generator, return predictions and labels
def predict_on_generator(model, generator):
    preds = []
    labels = []

    for i in range(len(generator)):
        x, y = generator[i]
        p = model.predict(x, batch_size=x.shape[0])
        preds.append(p)
        labels.append(y)

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels
