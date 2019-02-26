import argparse
import os
import time

def fetch_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-en', '--experiment_name', type=str, default='MPRA', help='Name of experiment')
    parser.add_argument('-dp', '--data_path', type=str, default='./data/', help='Path to training and validation hdf5 data')
    
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='Learning Rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('-ep', '--num_epochs', type=int, default=20, help='Total Number of Epochs')
    parser.add_argument('-ms', '--max_batch_steps', type=int, default=-1, help='Maximum number of steps in a batch, by default goes through entire batch')
    parser.add_argument('-op', '--optimizer', type=str, default='adam', help='Keras optimizer')
    parser.add_argument('-vt', '--verbose_training', type=int, default=1, help='Verbose Training')

    parser.add_argument('-mw', '--multiprocessing_workers', type=int, default=6, help='Number of workers for data loading')

    # for callbacks
    parser.add_argument('-mn', '--checkpoint_monitor', type=str, default= 'val_mean_spearmanr', help='Primary training metric')
    parser.add_argument('-md', '--checkpoint_mode', type=str, default='max', help='Primary training metric condition')
    parser.add_argument('-sb', '--checkpoint_save_best_only', type=int, default=1, help='Save only best models')
    parser.add_argument('-sv', '--checkpoint_save_weights_only', type=int, default=1, help='Save weights only')
    parser.add_argument('-vc', '--checkpoint_verbose', type=int, default=1, help='Verbose checkpointing')
    parser.add_argument('-tg', '--tensorboard_write_graph', type=int, default=1, help='Write tensorboard output')

    args = parser.parse_args()
    
    # processing
    args.tensorboard_log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), args.experiment_name, "logs/")
    args.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), args.experiment_name, "checkpoints/")

    return args
