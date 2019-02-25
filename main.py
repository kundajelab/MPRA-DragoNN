from data_loader.mpra_data_loader import MPRADataLoader as DataLoader
from models.conv_model import ConvModel as Model
from trainers.trainer import Trainer as Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    train_data_loader = DataLoader(config, 'train')
    valid_data_loader = DataLoader(config, 'valid')

    print('Create the model.')
    model = Model(config)

    print('Create the trainer')
    trainer = Trainer(model.model, train_data_loader, valid_data_loader, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()