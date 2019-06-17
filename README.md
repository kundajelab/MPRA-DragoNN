# MPRA-DragoNN: Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays

This project applies convolutional neural networks to predict output from massively parallel reporter assays (MPRAs), with the aim of systematically decoding regulatory sequence patterns and identifying noncoding variants that may affect gene expression.

## Data

This project uses the Sharpr-MPRA dataset from Ernst et al. 2016 (https://www.nature.com/articles/nbt.3678). The raw data were minimally processed, as described in the paper and below. The prepared training, validation, and testing datasets in hdf5 format can be downloaded from [here](http://mitra.stanford.edu/kundaje/projects/mpra/).

Technical details:

The raw data for each of the Sharpr-MPRA experiments was downloaded from the Gene Expression Omnibus, accession number GSE71279. These raw files are hosted at the following [Dropbox link](https://www.dropbox.com/sh/wh7b30dauxuajcw/AABQsvfmG65knGbFv0UsIcv1a?dl=0). 

The raw counts from the experiments were processed by (1) computing log2(RNA+1 / DNA+1) for each 145bp sequence in each of the 12 tasks (described below); (2) column-wise z-score normalization of the log fold-changes (i.e., each task's output values had mean 0 and variance 1); (3) including the reverse complement of each sequence as another datapoint with the same activity values.

Each datapoint was then converted to a 145x4 NumPy array corresponding to the one-hot encoding of the sequence's ACGT representation. The label for each sequence was a length-12 array containing the normalized activity values for the 12 tasks. The data was split as follows: sequences on chr8 for validation (~30K), chr18 for testing (~20K), and the remaining chromosomes for training (~900K); the resulting hdf5 files are the ones at [the data link](http://mitra.stanford.edu/kundaje/projects/mpra/).

Task description:

1. "k562_minp_rep1": K562 cell line, minimal promoter, replicate 1
1. "k562_minp_rep2": K562 cell line, minimal promoter, replicate 2
1. "k562_minp_avg": K562 cell line, minimal promoter, average\*
1. "k562_sv40p_rep1": K562 cell line, strong SV40 promoter, replicate 1
1. "k562_sv40p_rep1": K562 cell line, strong SV40 promoter, replicate 2
1. "k562_sv40p_avg": K562 cell line, strong SV40 promoter, average\*
1. "hepg2_minp_rep1": HepG2 cell line, minimal promoter, replicate 1
1. "hepg2_minp_rep2": HepG2 cell line, minimal promoter, replicate 2
1. "hepg2_minp_avg": HepG2 cell line, minimal promoter, average\*
1. "hepg2_sv40p_rep1": HepG2 cell line, strong SV40 promoter, replicate 1
1. "hepg2_sv40p_rep1": HepG2 cell line, strong SV40 promoter, replicate 2
1. "hepg2_sv40p_rep1": HepG2 cell line, strong SV40 promoter, average\*

\*The "average" tasks are computed by pooling counts between replicates, i.e. computing log2(RNA_Rep1 + RNA_Rep2 + 1) - log(DNA + 1).


## Model Training 

The inputs to our model are shape (145, 4) NumPy arrays corresponding to one-hot encoded 145 base-pair DNA sequences. The outputs are 12 continuous values corresponding to normalized activity levels of the sequence in different cellular contexts (described above).

The neural network model used for MPRA activity prediction is a fairly standard convolutional architecture for genomics. We use three convolution layers (ReLU activation), each containing 120 filters of width 5, followed by a single fully connected layer (linear activation) to predict the 12 tasks. Our model uses task-wise mean squared error loss and our primary evaluation criteria (for validation/testing) is the Spearman correlation (robust to outliers, unlike Pearson).

The models have been implemented in Keras with a Tensorflow backend. To train the model:

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

## Prediction and Interpretation 

We have provided pretrained models in `kipoi/ConvModel` and `kipoi/DeepFactorizedModel` directories. We provide support for prediction and interpretation through [Kipoi](https://github.com/kipoi/kipoi). The model json and yaml files are also available in the `kipoi` directory. The models can be loaded as:

```python
import kipoi

model = kipoi.get_model("SNPpet/ConvModel")    # or "SNPpet/DeepFactorizedModel"
```

Follow the instructions on Kipoi to make predictions on arbitrary sequences and for interpreting the model. 

## Help

Feel free to direct questions about this project to Rajiv Movva: rmovva at mit dot edu, or open an Issue.

## Citation

If you use this code for your research, please cite our paper: Movva R, Greenside P, Marinov GK, Nair S, Shrikumar A, Kundaje A (2019). Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays. PLoS ONE 14(6): e0218073. https://doi.org/10.1371/journal.pone.0218073
