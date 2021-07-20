from dataloaders.data_loader import DataLoader
from dataloaders.data_generator import DataGenerator
from models.fno_2d import FNO2D
from trainers.trainer import ModelTrainer
from utils.args import get_args
from utils.config import process_config


if __name__ == "__main__":
    args = get_args()  # parse args
    config = process_config(args.config)  # load config
    try:
        args = get_args() #parse args
        config = process_config(args.config) #load config
    except FileNotFoundError:
        print("File {} don't exists".format(args.config))
        exit(0)
    except Exception:
        print(("Missing or invalid arguments"))
        exit(0)
    dataloader = DataLoader(config)
    train_gen = DataGenerator(config, dataloader, subset='training')
    test_gen = DataGenerator(config, dataloader, subset='test')
    model = FNO2D(config)
    trainer = ModelTrainer(config, model)
    trainer.train(train_gen, test_gen)
    trainer.test(test_gen)



