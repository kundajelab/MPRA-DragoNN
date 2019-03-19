# MPRA

This project applies convolutional neural networks to predict output from massively parallel reporter assays (MPRAs), with the aim of systematically decoding regulatory sequence patterns and identifying noncoding variants that may affect gene expression (thus affecting disease risk).

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

We also provide support for [Kipoi](https://github.com/kipoi/kipoi). We have provided a pretrained model at `kipoi/pretrained.hdf5` directory.

## Citation

If you use this code for your research, please cite our paper.

<!--- add citation --->
