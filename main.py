from data_loader.mpra_data_loader import MPRADataLoader as DataLoader
from models.deep_factorized_model import DeepFactorizedModel as Model
from trainers.trainer import Trainer as Trainer
from evaluator.evaluator import Evaluator as Evaluator
from utils.dirs import create_dirs
from utils.fetch_args import fetch_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = fetch_args()

    # create the experiments dirs
    create_dirs([config.tensorboard_log_dir, config.checkpoint_dir])

    print('Create the data generator.')
    train_data_loader = DataLoader(config, 'train')
    valid_data_loader = DataLoader(config, 'valid')

    print('Create the model.')
    model = Model(config)
    if config.pretrained_model_checkpoint is not None:
        model.load(config.pretrained_model_checkpoint)
    
    if config.evaluate:
        print('Predicting on test set.')
        test_data_loader = DataLoader(config, 'test')
        evaluator = Evaluator(model.model, test_data_loader, config)
        evaluator.evaluate()
        exit(0)

    print('Create the trainer')
    trainer = Trainer(model.model, train_data_loader, valid_data_loader, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()