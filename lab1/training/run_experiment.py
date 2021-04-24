"""Experiment-running framework."""
import argparse # for sending command-line arguments for training
import importlib 

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

# loads the lit_models directory which has base.py and __init__.py
# __init__.py runs: from .base import BaseLitModel
# so it automatically loads our BaseLitModel class without running base.py
from text_recognizer import lit_models


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1) # splits into 2 elements at "."
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name) # gives us model.class_name attribute (ex: jacques = Person(), jacques.age -> 28)
    return class_ # the underscore after is part of PEP8 so that you don't use a built-in name like class or set


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--data_class", type=str, default="MNIST")      # data argument
    parser.add_argument("--model_class", type=str, default="MLP")       # model argument
    parser.add_argument("--load_checkpoint", type=str, default=None)    # load previous checkpoint (?)

    # Get the data and model classes, so that we can add their specific arguments
    # (?) This seems like it 
    temp_args, _ = parser.parse_known_args() # output: Namespace(data_class='MNIST', load_checkpoint=None, model_class='MLP')
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")      # temp_args.data_class -> MNIST
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")  # temp_args.model_class -> MLP

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args") # add_argument_group creates a group to make it easier for users
    data_class.add_to_argparse(data_group)              # add_to_argparse is a method in the classes

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_recognizer.data.{args.data_class}")       # argument group for data_class created in _setup_parse()
                                                                                # args for data_class: MNIST; later: EMNIST
    model_class = _import_class(f"text_recognizer.models.{args.model_class}")   # argument group for model_class created in _setup_parse()
                                                                                # args for model_class: MLP; later: CNNLSTM, CNNTRANSFORMER
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_models.BaseLitModel

    # Loading from a checkpoint after a training run. How is this defined?? What is the checkpoint string?
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)

    # How do you use this?
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")

    # pylint: disable=no-member
    # Why don't I need to pass train_dataloader, val_dataloaders?
    # Does this not do anything unless I pass --auto_lr_find and/or --auto_scale_batch_size?
    # It seems there's an error when we use both auto_lr_fin AND auto_scale_batch_size
    # Perhaps we need to run them separately: https://github.com/PyTorchLightning/pytorch-lightning/issues/5374
    # With MLP + MNIST: model had 10% accuracy with auto_lr_find (landed on 0.28 while default is 1e-3) and 50% accuracy with auto_scale_batch_size. However, +97% without them.
    # I expect that they both work well for transfer learning, but not for a simple untrained MLP.
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate
                                              # tune: can find best batch size and learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member



if __name__ == "__main__":
    main()
