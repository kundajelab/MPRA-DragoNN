# MPRA

@TODO Rajiv add description 

## Data

All associated data can be downloaded from [here](http://mitra.stanford.edu/kundaje/projects/mpra/).

@TODO Rajiv describe data

## Model Training 

@Rajiv mention what the model takes as inputs and predicts. Loss funciton etc. Evaluation criteria also. 

The models have been implemented in Keras with a Tensorflow backend. To start trianing the model:

```bash
python main.py --data_path /path/to/data
```

To resume training from an existing checkpoint:

```bash
python main --data_path /path/to/data --pretrained_model_checkpoint /path/to/checkpoint/model
```

During training, the model produces logs in the `experiments` directory, which can be visualized using tensorboard as:

```bash
tensorboard --logdir /path/to/log/dir/in/experiments
```

To evaluate on test set, pass the `--evaluate 1` flag in addition to resuming the model from the checkpoint.

For other inputs, such as hyperparameters, refer

```bash
python main.py --help
```

We have provided a pretrained model in the `pretrained_models` directory.

## Citation

If you use this code for your research, please cite our paper.

<!--- add citation --->
